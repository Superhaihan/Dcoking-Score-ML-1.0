import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import time

script_dir = os.path.dirname(os.path.abspath(__file__))

# 读取数据
data = pd.read_csv(os.path.join(script_dir, "duijie.csv"))
label = pd.read_csv(os.path.join(script_dir, "huoxing.csv"))

label.rename(columns={"Molecule ChEMBL ID": "Title"}, inplace=True)
label["Title"] = label["Title"].str.strip()  # 去掉空格

# 合并数据
data_merge = pd.merge(data, label, on='Title')

# 数据预处理
data_merge["Standard Value"] = data_merge["Standard Value"].apply(pd.to_numeric, errors='coerce').fillna(0.0)

# 查看活性分布
#sns.distplot(data_merge["Standard Value"].values)

# 计算中值
median_value = data_merge['Standard Value'].median()
print(median_value)
# 分箱处理
data_merge["Standard Value"] = data_merge["Standard Value"].apply(lambda x: 1 if x > median_value else 0)

# 删除不需要的特征
data_merge.drop(['Title'], axis=1, inplace=True)
data_merge.dropna(axis=0, how='any', inplace=True)

# 数据集划分
x = data_merge.drop(['Standard Value'], axis=1)
y = data_merge['Standard Value']
X_train_split, X_val, y_train_split, y_val = train_test_split(x, y, test_size=0.1)

# 数据重采样
ros = RandomOverSampler(random_state=0)
X_resampled, y_resampled = ros.fit_resample(X_train_split, y_train_split)
X_train_split = X_resampled
y_train_split = y_resampled

# 特征标准化
sc = StandardScaler()
norm = StandardScaler().fit(X_train_split)
sx_train = norm.transform(X_train_split)
sx_val = norm.transform(X_val)

# 训练和测试函数
# 训练和测试函数
def TrainandTest(X_train, y_train, X_test, y_test, algorithm, modelname):
    start = time.time()
    model = algorithm
    model.fit(X_train, y_train)
    end = time.time()
    prediction_train = model.predict(X_train)
    prediction_test = model.predict(X_test)

    train_score = round((accuracy_score(y_train, prediction_train) * 100), 2)
    test_score = round((accuracy_score(y_test, prediction_test) * 100), 2)

    cm_train = confusion_matrix(y_train, prediction_train)
    cm_test = confusion_matrix(y_test, prediction_test)

    print("\nTraining Set - Model Score:", train_score, "%")
    print("Training Set - Precision:", precision_score(y_train, prediction_train, average="micro"))
    print("Training Set - Recall:", recall_score(y_train, prediction_train, average='micro'))
    print("Training Set - F1 score:", f1_score(y_train, prediction_train, average='micro'))
    print("Training Set - Confusion Matrix:\n", cm_train)

    print("\nTesting Set - Model Score:", test_score, "%")
    print("Testing Set - Precision:", precision_score(y_test, prediction_test, average="micro"))
    print("Testing Set - Recall:", recall_score(y_test, prediction_test, average='micro'))
    print("Testing Set - F1 score:", f1_score(y_test, prediction_test, average='micro'))
    print("Testing Set - Confusion Matrix:\n", cm_test)

    print("Spends time:", end - start)
    print()

    ### plt.show()

    model_info = {}
    model_info['Algorithm'] = modelname
    model_info['Training Set Model Score'] = str(train_score) + "%"
    model_info['Testing Set Model Score'] = str(test_score) + "%"
    model_info['Training Set Precision'] = round(precision_score(y_train_split, prediction_train, average="micro"), 2)
    model_info['Testing Set Precision'] = round(precision_score(y_test, prediction_test, average="micro"), 2)
    model_info['Training Set Recall'] = round(recall_score(y_train_split, prediction_train, average="micro"), 2)
    model_info['Testing Set Recall'] = round(recall_score(y_test, prediction_test, average="micro"), 2)
    model_info['Training Set F1 score'] = round(f1_score(y_train, prediction_train, average="micro"), 2)
    model_info['Testing Set F1 score'] = round(f1_score(y_test, prediction_test, average="micro"), 2)
    model_info["Spend time"] = end - start
    return model_info


DataModels = pd.DataFrame()

DataModels = pd.DataFrame()

# 模型选择
DataModels = []

# 模型选择
algorithms = {
    "Random Forest": RandomForestClassifier(n_estimators=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Logistic Regression": LogisticRegression(multi_class="multinomial", solver="newton-cg"),
    "MLP": MLPClassifier(solver='sgd', activation='relu', alpha=1e-4, max_iter=10000, hidden_layer_sizes=(64, 32),
                         random_state=1)
}

for model_name, model_algorithm in algorithms.items():
    print("Algorithm Performance: {}".format(model_name))
    model_info = TrainandTest(sx_train, y_train_split, sx_val, y_val, model_algorithm, model_name)
    DataModels.append(model_info)

DataModels = pd.DataFrame(DataModels)
