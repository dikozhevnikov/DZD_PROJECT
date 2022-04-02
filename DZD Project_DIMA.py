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
url1 = 'https://raw.githubusercontent.com/dikozhevnikov/DZD_PROJECT/master/Telco_customer_churn_location-2.csv'
location = pd.read_csv(url1)
url2 = 'https://raw.githubusercontent.com/dikozhevnikov/DZD_PROJECT/master/Customer_Churn.csv'
churn = pd.read_csv(url2)

print(location)
print(churn)

#Join Location and Churn datasets
df = churn.join(location)

#create clasters for zip_codes
df["Zip_cluster"]= df["Zip Code"].map(lambda x: int(x)//50*50)

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


#Create tenure Bins
def ten_exp(x):
  lowers = [-1,3,6,12,24,60,120]
  for i,lower in enumerate(lowers):
    if x<=lower: return i #f"{lowers[i-1]+1} - {lower}"
df["tenure_exp"] = df["tenure"].map(ten_exp)
df[["tenure", "tenure_exp"]]
df["tenure_exp"].value_counts()



#Creation Bins - Quartile
df['quartile_MonCharg'] = pd.qcut(df['MonthlyCharges'], q=4)
print(df.quartile_MonCharg.unique())


#4ft Miner - dependency of the churn on the type of services used

clm = cleverminer(df=df,proc='4ftMiner',
               quantifiers= {'conf':0.7, 'Base':100},
               ante ={
                    'attributes':[
                        {'name': 'Leave', 'type': 'subset', 'minlen': 1, 'maxlen': 1}
                    ], 'minlen':1, 'maxlen':1, 'type':'con'},
               succ ={
                    'attributes':[
                        {'name': 'PhoneService', 'type': 'subset', 'minlen': 1, 'maxlen': 1},
                        {'name': 'MultipleLines', 'type': 'subset', 'minlen': 1, 'maxlen': 1},
                        {'name': 'InternetService', 'type': 'subset', 'minlen': 1, 'maxlen': 1},
                        {'name': 'OnlineSecurity', 'type': 'subset', 'minlen': 1, 'maxlen': 1},
                        {'name': 'DeviceProtection', 'type': 'subset', 'minlen': 1, 'maxlen': 1},
                        {'name': 'TechSupport', 'type': 'subset', 'minlen': 1, 'maxlen': 1},
                        {'name': 'StreamingTV', 'type': 'subset', 'minlen': 1, 'maxlen': 1},
                        {'name': 'StreamingMovies', 'type': 'subset', 'minlen': 1, 'maxlen': 1},
                        {'name': 'OnlineBackup', 'type': 'subset', 'minlen': 1, 'maxlen': 1},
                    ], 'minlen':1, 'maxlen':3, 'type':'con'}
               )

clm.print_summary()
clm.print_rulelist()
clm.print_rule(2)
clm.print_rule(3)
clm.print_rule(4)
clm.print_rule(5)


#4ft Miner - dependency of the churn on the type of services used

clm = cleverminer(df=df,proc='4ftMiner',
               quantifiers= {'conf':0.7, 'Base':100},
               ante ={
                    'attributes':[
                        {'name': 'Leave', 'type': 'subset', 'minlen': 1, 'maxlen': 1}
                    ], 'minlen':1, 'maxlen':1, 'type':'con'},
               succ ={
                    'attributes':[
                        {'name': 'Contract', 'type': 'subset', 'minlen': 1, 'maxlen': 1},
                        {'name': 'PaymentMethod', 'type': 'subset', 'minlen': 1, 'maxlen': 1},
                        {'name': 'quartile_MonCharg', 'type': 'seq', 'minlen': 1, 'maxlen': 1},
                        {'name': 'PaperlessBilling', 'type': 'subset', 'minlen': 1, 'maxlen': 1},
                    ], 'minlen':1, 'maxlen':3, 'type':'con'}
               )

clm.print_summary()
clm.print_rulelist()
clm.print_rule(1)
clm.print_rule(2)

#4ft Miner - dependency of the churn on the demografics

clm = cleverminer(df=df,proc='4ftMiner',
               quantifiers= {'conf':0.8, 'Base':100},
               ante ={
                    'attributes':[
                        {'name': 'Leave', 'type': 'subset', 'minlen': 1, 'maxlen': 1}
                    ], 'minlen':1, 'maxlen':1, 'type':'con'},
               succ ={
                    'attributes':[
                        {'name': 'Dependents', 'type': 'subset', 'minlen': 1, 'maxlen': 1},
                        {'name': 'Partner', 'type': 'subset', 'minlen': 1, 'maxlen': 1},
                        {'name': 'SeniorCitizen', 'type': 'subset', 'minlen': 1, 'maxlen': 1},
                        {'name': 'gender', 'type': 'subset', 'minlen': 1, 'maxlen': 1},
                        {'name': 'tenure_exp', 'type': 'seq', 'minlen': 1, 'maxlen': 1},
                    ], 'minlen':1, 'maxlen':3, 'type':'con'}
               )

clm.print_summary()
clm.print_rulelist()
clm.print_rule(1)
clm.print_rule(2)


#4ft Miner - dependency of the churn on the geography (nefunguje a je tu nekonečný proces)

clm = cleverminer(df=df,proc='4ftMiner',
               quantifiers= {'conf':0.8, 'Base':100},
               ante ={
                    'attributes':[
                        {'name': 'Leave', 'type': 'subset', 'minlen': 1, 'maxlen': 1}
                    ], 'minlen':1, 'maxlen':1, 'type':'con'},
               succ ={
                    'attributes':[
                        {'name': 'Zip_cluster', 'type': 'subset', 'minlen': 1, 'maxlen': 40}
                    ], 'minlen':1, 'maxlen':1, 'type':'con'}
               )

clm.print_summary()
clm.print_rulelist()