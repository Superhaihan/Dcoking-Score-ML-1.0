#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


import pandas as pd
import os
import glob


# In[2]:


for file in glob.glob("Down*.csv"):
    print (file)


# In[3]:


with open(file, 'r', encoding='utf8') as f:
   content = f.read()
content2 = content.replace(';', ',')
with open('download2.csv', 'w', encoding='utf8') as f:
   f.write(content2)


# In[4]:


df = pd.read_csv("download2.csv")


# In[5]:


df.head()


# In[6]:


cols = ['Molecule ChEMBL ID','Smiles' ]


# In[7]:


df = pd.read_csv("download2.csv")


# In[8]:


df.head()


# In[9]:


cols = ['Molecule ChEMBL ID','Smiles' ]


# In[10]:


df[cols].head()


# In[11]:


df[cols].to_csv("huahewu.csv",encoding='utf8',index=False)


# In[12]:



df = pd.read_csv("download2.csv")
cols = ['Molecule ChEMBL ID','Standard Value']
df[cols].to_csv("huoxing.csv",encoding='utf8',index=False)


# In[ ]:





# In[ ]:





# In[ ]:




