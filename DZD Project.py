#Import packages
from urllib import request
from urllib.request import urlopen

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sys

from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

from cleverminer import cleverminer

#Import Datasets
url1 = 'https://raw.githubusercontent.com/dikozhevnikov/DZD_Colab/main/Telco_customer_churn_location-2.csv'
location = pd.read_csv(url1)
url2 = 'https://raw.githubusercontent.com/dikozhevnikov/DZD_Colab/main/Customer_Churn.csv'
churn = pd.read_csv(url2)

print(location)
print(churn)

#Join Location and Churn datasets
df = churn.join(location)
df.City.nunique()
df["Zip Code"].nunique()
df["Zip_cluster"]= df["Zip Code"].map(lambda x: int(x)//50*50)
df["Zip_cluster"].nunique()


#Integer Datatype reassessment
df['IsFemale'] = (df['gender'] == 'Female').astype(int)
df['Leave'] = (df['Churn'] == 'Yes').astype(int)
df['Partner'] = (df['Partner'] == 'Yes').astype(int)
df['PaperlessBilling'] = (df['PaperlessBilling'] == 'Yes').astype(int)
df = df.astype({"Leave": bool})

#Show Attribute Names and Count Unique

cols=pd.DataFrame(df.columns)
cols["c"] = cols[0].map(lambda x: df[x].nunique())
cols = cols.sort_values("c",ascending=False)
cols

#Creation Bins - Quartile
df['quartile_MonCharg'] = pd.qcut(df['MonthlyCharges'], q=4)
print(df.quartile_MonCharg.unique())

#4ft Miner - dependency of the Monthly Charge on the loyalty

clm = cleverminer(df=df,proc='4ftMiner',
               quantifiers= {'conf':0.8, 'Base':500},
               ante ={
                    'attributes':[
                        {'name': 'quartile_MonCharg', 'type': 'subset', 'minlen': 1, 'maxlen': 1}
                    ], 'minlen':1, 'maxlen':1, 'type':'con'},
               succ ={
                    'attributes':[
                        {'name': 'Leave', 'type': 'subset', 'minlen': 1, 'maxlen': 1}
                    ], 'minlen':1, 'maxlen':1, 'type':'con'},
               )

clm.print_summary()
clm.print_rulelist()
clm.print_rule(1)

print(df.quartile_MonCharg.unique())

#4ft Miner

clm = cleverminer(df=df,proc='4ftMiner',
               quantifiers= {'conf':0.2, 'Base':100},
               ante ={
                    'attributes':[
                        {'name': 'PaperlessBilling', 'type': 'subset', 'minlen': 1, 'maxlen': 1}
                    ], 'minlen':1, 'maxlen':1, 'type':'con'},
               succ ={
                    'attributes':[
                        {'name': 'PaymentMethod', 'type': 'subset', 'minlen': 1, 'maxlen': 1}
                    ], 'minlen':1, 'maxlen':1, 'type':'con'},
               )

clm.print_summary()
clm.print_rulelist()
clm.print_rule(1)


#4ft Miner

clm = cleverminer(df=df,proc='4ftMiner',
               quantifiers= {'conf':0.8, 'Base':100},
               ante ={
                    'attributes':[
                        {'name': 'Leave', 'type': 'subset', 'minlen': 1, 'maxlen': 1}
                    ], 'minlen':1, 'maxlen':1, 'type':'con'},
               succ ={
                    'attributes':[
                        {'name': 'Contract', 'type': 'subset', 'minlen': 1, 'maxlen': 1}
                    ], 'minlen':1, 'maxlen':1, 'type':'con'},
               )

clm.print_summary()
clm.print_rulelist()
clm.print_rule(1)
