import pandas as pd
import numpy as num
import matplotlib.pyplot as plt
import seaborn as sns
import copy
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

training_data = pd.read_csv("/home/icts/practice-datasets/Big-Mart-Sales/train.csv")
testing_data = pd.read_csv("/home/icts/practice-datasets/Big-Mart-Sales/test.csv")
training_data['label']='train'
testing_data['label']='test'
data=pd.concat([training_data, testing_data], ignore_index=True)

#work with data and later split into train and test sets to avoid repeated computations.
data.apply(lambda x:sum(x.isnull()))
data.apply(lambda x:len(x.unique()))

cat_column_names = data.select_dtypes(include=['object']).copy()
cat_column_names = cat_column_names.drop(['label', 'Item_Identifier', 'Outlet_Identifier'], axis=1)

for i in cat_column_names:
	data[i].value_counts()

avg_wt = data.pivot_table(values='Item_Weight', index='Item_Identifier')
_bool = data["Item_Weight"].isnull()

data.loc[_bool, "Item_Weight"] = data.loc[_bool, "Item_Identifier"].apply(lambda x: avg_wt[x])mo

"""
training_data['Item_Weight'].fillna(training_data['Item_Weight'].mean())
testing_data['Item_Weight'].fillna(testing_data['Item_Weight'].mean())

sns.relplot(x="Item_MRP", y="Item_Outlet_Sales", data=training_data)
#plt.show()

sns.catplot(y="Item_Outlet_Sales", x="Item_Type", kind="violin", hue="Item_Fat_Content", data=training_data)
plt.show()



corr=training_data.corr()
sns.heatmap(corr)
#plt.show()
"""