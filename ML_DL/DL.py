import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.utils import scatter
import torch.nn.functional as F
import os
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from torch_geometric.data import Batch
from rdkit import Chem
from rdkit.Chem import GetAdjacencyMatrix
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define atom and bond feature extraction functions
def one_hot_encoding(x, permitted_list):
    if x not in permitted_list:
        x = permitted_list[-1]
    binary_encoding = [int(boolean_value) for boolean_value in list(map(lambda s: x == s, permitted_list))]
    return binary_encoding

def get_atom_features(atom, use_chirality=True, hydrogens_implicit=True):
    permitted_list_of_atoms = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I',
                               'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'Li', 'Ge',
                               'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'Unknown']
    if hydrogens_implicit == False:
        permitted_list_of_atoms = ['H'] + permitted_list_of_atoms
    atom_type_enc = one_hot_encoding(str(atom.GetSymbol()), permitted_list_of_atoms)
    n_heavy_neighbors_enc = one_hot_encoding(int(atom.GetDegree()), [0, 1, 2, 3, 4, "MoreThanFour"])
    formal_charge_enc = one_hot_encoding(int(atom.GetFormalCharge()), [-3, -2, -1, 0, 1, 2, 3, "Extreme"])
    hybridisation_type_enc = one_hot_encoding(str(atom.GetHybridization()),
                                              ["S", "SP", "SP2", "SP3", "SP3D", "SP3D2", "OTHER"])
    is_in_a_ring_enc = [int(atom.IsInRing())]
    is_aromatic_enc = [int(atom.GetIsAromatic())]
    vdw_radius_scaled = [float((Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum()) - 1.5) / 0.6)]
    covalent_radius_scaled = [float((Chem.GetPeriodicTable().GetRcovalent(atom.GetAtomicNum()) - 0.64) / 0.76)]

    atom_feature_vector = atom_type_enc + n_heavy_neighbors_enc + formal_charge_enc + hybridisation_type_enc + is_in_a_ring_enc + is_aromatic_enc + vdw_radius_scaled + covalent_radius_scaled

    if use_chirality == True:
        chirality_type_enc = one_hot_encoding(str(atom.GetChiralTag()),
                                              ["CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW",
                                               "CHI_OTHER"])
        atom_feature_vector += chirality_type_enc
    if hydrogens_implicit == True:
        n_hydrogens_enc = one_hot_encoding(int(atom.GetTotalNumHs()), [0, 1, 2, 3, 4, "MoreThanFour"])
        atom_feature_vector += n_hydrogens_enc

    return np.array(atom_feature_vector)


def get_bond_features(bond, use_stereochemistry=True):
    permitted_list_of_bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
    bond_type_enc = one_hot_encoding(bond.GetBondType(), permitted_list_of_bond_types)
    bond_is_conj_enc = [int(bond.GetIsConjugated())]
    bond_is_in_ring_enc = [int(bond.IsInRing())]
    bond_feature_vector = bond_type_enc + bond_is_conj_enc + bond_is_in_ring_enc
    if use_stereochemistry == True:
        stereo_type_enc = one_hot_encoding(str(bond.GetStereo()), ["STEREOZ", "STEREOE", "STEREOANY", "STEREONONE"])
        bond_feature_vector += stereo_type_enc
    return np.array(bond_feature_vector)

def create_pytorch_geometric_graph_data_list_from_smiles_and_labels(x_smiles, y):
    data_list = []
    for (smiles, y_val) in zip(x_smiles, y):
        mol = Chem.MolFromSmiles(smiles)
        n_nodes = mol.GetNumAtoms()
        n_edges = 2 * mol.GetNumBonds()
        unrelated_smiles = "O=O"
        unrelated_mol = Chem.MolFromSmiles(unrelated_smiles)
        n_node_features = len(get_atom_features(unrelated_mol.GetAtomWithIdx(0)))
        n_edge_features = len(get_bond_features(unrelated_mol.GetBondBetweenAtoms(0, 1)))
        X = np.zeros((n_nodes, n_node_features))
        for atom in mol.GetAtoms():
            X[atom.GetIdx(), :] = get_atom_features(atom)
        X = torch.tensor(X, dtype=torch.float)
        (rows, cols) = np.nonzero(GetAdjacencyMatrix(mol))
        torch_rows = torch.from_numpy(rows.astype(np.int64)).to(torch.long)
        torch_cols = torch.from_numpy(cols.astype(np.int64)).to(torch.long)
        E = torch.stack([torch_rows, torch_cols], dim=0)
        EF = np.zeros((n_edges, n_edge_features))
        for (k, (i, j)) in enumerate(zip(rows, cols)):
            EF[k] = get_bond_features(mol.GetBondBetweenAtoms(int(i), int(j)))
        EF = torch.tensor(EF, dtype=torch.float)
        y_tensor = torch.tensor(np.array([y_val]), dtype=torch.float)
        data_list.append(Data(x=X, edge_index=E, edge_attr=EF, y=y_tensor))
    return data_list




script_dir = os.path.dirname(os.path.abspath(__file__))


data = pd.read_csv(os.path.join(script_dir, "结果.csv"))
def create_batch(data_list):
    return Batch.from_data_list(data_list)
# 使用样本ID或索引进行数据划分
indices = list(range(len(data)))
train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42)

# 对数值数据进行处理
numerical_train = data.select_dtypes(include=["float64", "int64"]).iloc[train_indices]
numerical_test = data.select_dtypes(include=["float64", "int64"]).iloc[test_indices]

X_train = numerical_train.drop("Standard Value", axis=1, errors='ignore').values
y_train = numerical_train["Standard Value"].values
X_test = numerical_test.drop("Standard Value", axis=1, errors='ignore').values
y_test = numerical_test["Standard Value"].values

# 标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 转换数据到 PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train).to(device)
y_train_tensor = torch.LongTensor(y_train).to(device)
X_test_tensor = torch.FloatTensor(X_test).to(device)
y_test_tensor = torch.LongTensor(y_test).to(device)

# 对图数据进行处理
df_train = data.iloc[train_indices].dropna(subset=['Standard Value'])
df_test = data.iloc[test_indices].dropna(subset=['Standard Value'])

df_train['Standard Value'] = np.where(df_train['Standard Value'] > 0, 1, 0)
x_smiles_train = df_train["Smiles"].tolist()
y_train_graph = df_train["Standard Value"].values
train_data = create_pytorch_geometric_graph_data_list_from_smiles_and_labels(x_smiles_train, y_train_graph)

df_test['Standard Value'] = np.where(df_test['Standard Value'] > 0, 1, 0)
x_smiles_test = df_test["Smiles"].tolist()
y_test_graph = df_test["Standard Value"].values
test_data = create_pytorch_geometric_graph_data_list_from_smiles_and_labels(x_smiles_test, y_test_graph)
input_dim1 = train_data[0].x.shape[1]
print(input_dim1)
hidden_dim1=256
output_dim1=64
class AttentionPooling(nn.Module):
    def __init__(self, node_dim):
        super(AttentionPooling, self).__init__()

        # Define the attention mechanism
        self.attention_mechanism = nn.Linear(node_dim, 1)

    def forward(self, node_features, batch_index):
        # Get attention weights
        attention_weights = self.attention_mechanism(node_features)
        attention_weights = F.softmax(attention_weights, dim=0)

        # Weighted sum of node features according to the batch_index
        pooled_representation = scatter(node_features * attention_weights, batch_index, dim=0, reduce="sum")

        return pooled_representation

class GCNWithAttentionPooling(nn.Module):
    def __init__(self, input_dim1, hidden_dim, output_dim1):
        super(GCNWithAttentionPooling, self).__init__()
        self.conv1 = GCNConv(input_dim1, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.pooling = AttentionPooling(hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = self.pooling(x, batch)
        return x

class MLP1(nn.Module):
    def __init__(self, input_dim):
        super(MLP1, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

    def forward(self, x):
        return self.layers(x)
# 定义合并输出的 MLP 模型，输入维度为两个网络输出的维度之和
class CombinedMLP(nn.Module):
    def __init__(self, input_dim):
        super(CombinedMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(320, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # 二分类
        )
    def forward(self, x):
        return self.layers(x)
#gcn_model = GCN(input_dim1, hidden_dim1,output_dim1).to(device)
gcn_model2 = GCNWithAttentionPooling(input_dim1, hidden_dim1,output_dim1).to(device)

input_dim = X_train.shape[1]
mlp1 = MLP1(input_dim).to(device)
combined_mlp_model = CombinedMLP(64 + 64).to(device)


# 1. 前向传播
def forward(data_batch_list, numerical_features):
    # Convert list of Data objects to a single Batch
    batch_data = Batch.from_data_list(data_batch_list)

    gcn_output2 = gcn_model2(batch_data)
    mlp_output = mlp1(numerical_features)

    # Apply weight to GCN output
    gcn_weight = nn.Parameter(torch.tensor([5.0]))
    gcn_weight = gcn_weight.to(gcn_output2.device)# 设置GCN权重
    gcn_output2_weighted = torch.mul(gcn_output2, gcn_weight)

    combined_features = torch.cat((gcn_output2_weighted, mlp_output), dim=1)  # 注意dim=1，因为我们现在处理的是批数据
    final_output = combined_mlp_model(combined_features)
    return final_output

# 2. 损失函数
criterion = nn.CrossEntropyLoss()

# 3. 优化器
optimizer = torch.optim.Adam(
    list(gcn_model2.parameters()) + list(mlp1.parameters()) + list(combined_mlp_model.parameters()), lr=0.0001)

# 4. 训练循环
epochs = 350
batch_size = 1
num_batches = len(train_data) // batch_size
for epoch in range(epochs):

    total_loss = 0.0
    correct_predictions = 0
    for batch_idx in range(num_batches):
        # 获取一个批次的数据
        batch_data = train_data[batch_idx * batch_size: (batch_idx + 1) * batch_size]
        batch_numerical_features = X_train_tensor[batch_idx * batch_size: (batch_idx + 1) * batch_size]
        batch_labels = y_train_tensor[batch_idx * batch_size: (batch_idx + 1) * batch_size]

        # 将数据移到相同的设备上
        batch_data = [data.to(device) for data in batch_data]  # 注意，这里我们使用列表推导式，因为batch_data是一个Data对象的列表
        batch_numerical_features = batch_numerical_features.to(device)
        batch_labels = batch_labels.to(device)
        # 将数据传递给模型
        outputs = forward(batch_data, batch_numerical_features)
        outputs = forward(batch_data, batch_numerical_features)
        loss = criterion(outputs, batch_labels)
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    outputs = forward(batch_data, batch_numerical_features)
    loss = criterion(outputs, batch_labels)
    _, predicted = torch.max(outputs, 1)  # 获取每个样本的最大值的索引
    correct_predictions += (predicted == batch_labels).sum().item()
    accuracy = correct_predictions / len(train_data)  # 计算准确率
    gcn_model2.eval()
    mlp1.eval()
    combined_mlp_model.eval()

    correct_predictions_test = 0
    with torch.no_grad():  # 在评估模式下，我们不计算梯度
        for i, data in enumerate(test_data):
            #outputs = forward([data.to(device)], X_test_tensor[i].to(device))
            single_test_sample = X_test_tensor[i].unsqueeze(0).to(device)
            outputs = forward([data.to(device)], single_test_sample)
            predicted = outputs.argmax()
            correct_predictions_test += (predicted == y_test_tensor[i].to(device)).sum().item()
            batch_data_sample = train_data[:batch_size]
            batch_numerical_features_sample = X_train_tensor[:batch_size]
            batch_labels_sample = y_train_tensor[:batch_size]
            outputs_debug = forward(batch_data_sample, batch_numerical_features_sample)
            _, predicted_class_debug = torch.max(outputs_debug, 1)
            #print("Predicted classes:", predicted_class_debug.cpu().numpy())
            #print("Actual classes:   ", batch_labels_sample.cpu().numpy())
            # 计算训练集中1和0的数量
            train_zeros = len(y_train) - np.count_nonzero(y_train)
            train_ones = np.count_nonzero(y_train)

            # 计算测试集中1和0的数量
            test_zeros = len(y_test) - np.count_nonzero(y_test)
            test_ones = np.count_nonzero(y_test)

            # 计算每个类别的占比
            train_zeros_percentage = train_zeros / len(y_train) * 100
            train_ones_percentage = train_ones / len(y_train) * 100

            test_zeros_percentage = test_zeros / len(y_test) * 100
            test_ones_percentage = test_ones / len(y_test) * 100

            # 打印结果
            #print(f"Training set: 0s: {train_zeros_percentage:.2f}%, 1s: {train_ones_percentage:.2f}%")
           # print(f"Test set: 0s: {test_zeros_percentage:.2f}%, 1s: {test_ones_percentage:.2f}%")

    test_accuracy = correct_predictions_test / len(test_data)
    #print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_data)}, Accuracy: {accuracy:.4f}")
   # print( f"Epoch {epoch + 1}/{epochs}, Train Loss: {total_loss / len(train_data)} Test Accuracy: {test_accuracy:.4f}")


    def compute_accuracy(combined_mlp_model , data_set, features, labels):
        correct_predictions = 0

        with torch.no_grad():
            for i, data in enumerate(data_set):
                sample_features = features[i].unsqueeze(0).to(device)
                outputs = forward([data.to(device)], sample_features)
                predicted = outputs.argmax(dim=1)
                correct_predictions += (predicted == labels[i].to(device)).sum().item()

        accuracy = correct_predictions / len(data_set)
        return accuracy


    # 在每个训练循环结束后或在需要的时候，您可以这样调用上述函数来得到训练和测试的准确度：
    train_accuracy = compute_accuracy(combined_mlp_model , train_data, X_train_tensor, y_train_tensor)
    test_accuracy = compute_accuracy(combined_mlp_model , test_data, X_test_tensor, y_test_tensor)
    print(epoch)
    print(f"total_loss :{total_loss}")
    print(f"Train Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")