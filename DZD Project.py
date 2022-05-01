#Import packages
from urllib import request
from urllib.request import urlopen

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
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
df["Zip_cluster"]= df["Zip Code"].map(lambda x: int(x)//100)

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

# Create new column Services
def services(x):
  res = ""
  if x.PhoneService == "Yes": res += "Phone "
  if x.InternetService != "No": res += "Internet "
  if x.StreamingTV != "No" or x.StreamingMovies != "No": res += "TV "
  return res

df["Service"] = df.apply(services, axis=1)
df.Service.value_counts()


#Creation Bins - Quartile
df['Quartile_MonCharg'] = pd.qcut(df['MonthlyCharges'], q=4)
print(df.Quartile_MonCharg.unique())


#4ft Miner - dependency of the churn on the type of services used

clm = cleverminer(df=df,proc='4ftMiner',
               quantifiers= {'conf':0.85, 'Base':1400},
               ante ={
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
                    ], 'minlen':1, 'maxlen':4, 'type':'con'},
               succ ={
                    'attributes':[
                        {'name': 'Leave', 'type': 'subset', 'minlen': 1, 'maxlen': 1}
                    ], 'minlen':1, 'maxlen':1, 'type':'con'}
               )

clm.print_summary()
clm.print_rulelist()
clm.print_rule(1)
clm.print_rule(132)


#4ft Miner - dependency of the churn on the contract conditions

clm = cleverminer(df=df,proc='4ftMiner',
               quantifiers= {'conf':0.9, 'Base':1000},
               ante ={
                    'attributes':[
                        {'name': 'Contract', 'type': 'subset', 'minlen': 1, 'maxlen': 1},
                        {'name': 'PaymentMethod', 'type': 'subset', 'minlen': 1, 'maxlen': 1},
                        {'name': 'Quartile_MonCharg', 'type': 'seq', 'minlen': 1, 'maxlen': 1},
                        {'name': 'PaperlessBilling', 'type': 'subset', 'minlen': 1, 'maxlen': 1}
                    ], 'minlen':1, 'maxlen':4, 'type':'con'},
               succ ={
                    'attributes':[
                        {'name': 'Leave', 'type': 'subset', 'minlen': 1, 'maxlen': 1}
                    ], 'minlen':1, 'maxlen':1, 'type':'con'}
               )

clm.print_summary()
clm.print_rulelist()
clm.print_rule(1)
clm.print_rule(2)

#4ft Miner - dependency of the churn on the demografic characteristics

clm = cleverminer(df=df,proc='4ftMiner',
               quantifiers= {'conf':0.9, 'Base':1000},
               ante ={
                    'attributes':[
                        {'name': 'Dependents', 'type': 'subset', 'minlen': 1, 'maxlen': 1},
                        {'name': 'Partner', 'type': 'subset', 'minlen': 1, 'maxlen': 1},
                        {'name': 'SeniorCitizen', 'type': 'subset', 'minlen': 1, 'maxlen': 1},
                        {'name': 'gender', 'type': 'subset', 'minlen': 1, 'maxlen': 1},
                        {'name': 'tenure_exp', 'type': 'seq', 'minlen': 1, 'maxlen': 1},
                    ], 'minlen':1, 'maxlen':5, 'type':'con'},
               succ ={
                    'attributes':[
                        {'name': 'Leave', 'type': 'subset', 'minlen': 1, 'maxlen': 1}
                    ], 'minlen':1, 'maxlen':1, 'type':'con'}
               )

clm.print_summary()
clm.print_rulelist()
clm.print_rule(1)
clm.print_rule(2)
clm.print_rule(3)



# CFMiner- Payment Method
his= df.PaymentMethod.hist()
# 'Bank transfer (automatic)', 'Credit card (automatic)', 'Electronic check', 'Mailed check'

# Condition  : PhoneService(Yes ) & InternetService(No ) & gender(Male ) & tenure_exp(1 2 )

#Histogram [15, 17, 15, 148]
clm = cleverminer(df=df.copy(),target='PaymentMethod',proc='CFMiner',
               quantifiers= {'RelMax':0.75, 'Base':100},
               cond ={
                    'attributes':[

                        {'name': 'PhoneService', 'type': 'subset', 'minlen': 1, 'maxlen': 1},
                        {'name': 'InternetService', 'type': 'subset', 'minlen': 1, 'maxlen': 1},
                        {'name': 'gender', 'type': 'subset', 'minlen': 1, 'maxlen': 1},
                        {'name': 'SeniorCitizen', 'type': 'subset', 'minlen': 1, 'maxlen': 1},
                        {'name': 'tenure_exp', 'type': 'seq', 'minlen': 1, 'maxlen': 3},
                        {'name': 'Zip_cluster', 'type': 'seq', 'minlen': 1, 'maxlen': 4}
                    ], 'minlen':1, 'maxlen':4, 'type':'con'}
               )


#clm.print_summary()
clm.print_rulelist()
clm.print_rule(4)
print(clm.result)


# CFMiner Payment Method
# InternetService(No ) & gender(Male ) & tenure_exp(1 2 )
clm = cleverminer(df=df.copy(),target='PaymentMethod',proc='CFMiner',
               quantifiers= {'RelMax':0.75, 'Base':100},
               cond ={
                    'attributes':[
                        {'name': 'InternetService', 'type': 'subset', 'minlen': 1, 'maxlen': 1},
                        {'name': 'gender', 'type': 'subset', 'minlen': 1, 'maxlen': 1},
                        {'name': 'tenure_exp', 'type': 'seq', 'minlen': 1, 'maxlen': 3},
                        {'name': 'Zip_cluster', 'type': 'seq', 'minlen': 1, 'maxlen': 4}
                    ], 'minlen':1, 'maxlen':3, 'type':'con'}
               )


clm.print_summary()
clm.print_rulelist()
clm.print_rule(1)
print(clm.result)

#Predikce

#prehodnoceni dat na integer
df['IsFemale'] = (df['gender'] == 'Female').astype(int)
df['Leave'] = (df['Churn'] == 'Yes').astype(int)
df['Partner'] = (df['Partner'] == 'Yes').astype(int)
df['PaperlessBilling'] = (df['PaperlessBilling'] == 'Yes').astype(int)
df

#pretypovani dat na boolean (TRUE/FALSE)
df = df.astype({'Leave': bool})

#vytvoreni nových sloupců rozdělením
df = pd.get_dummies(df, columns=['PaymentMethod'], drop_first=True)
df = pd.get_dummies(df, columns=['MultipleLines'], drop_first=True)
df = pd.get_dummies(df, columns=['PhoneService'], drop_first=True)
df = pd.get_dummies(df, columns=['InternetService'], drop_first=True)
df = pd.get_dummies(df, columns=["StreamingTV"], drop_first=True)
df

#rozdeleni trenovacich a testovacich dat - kontrola klasifikatorů
predictors = ['IsFemale','SeniorCitizen','Partner','tenure','PaymentMethod_Credit card (automatic)','PaymentMethod_Electronic check','PaymentMethod_Mailed check', 'MultipleLines_No phone service', 'MultipleLines_Yes','PhoneService_Yes','InternetService_Fiber optic','InternetService_No','StreamingTV_No internet service', 'StreamingTV_Yes' ]
X = df[predictors].values
y = df['Leave'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)
np.unique(y)

#Logisticka regrese

listC = list(np.power(10.0, np.arange(-4, 4)))
listC
model_lr = LogisticRegressionCV(cv=10,Cs=listC,scoring = "accuracy", random_state=40, max_iter=1000)
model_lr.fit(X_train, y_train)
model_lr.score(X_train,y_train)
model_lr.score(X_test, y_test)

#decision tree > hyperparameter tuning

model_dt_default = DecisionTreeClassifier(random_state = 40)
model_dt_default.fit(X_train, y_train)
model_dt_default.score(X_train, y_train)
model_dt_default.score(X_test, y_test)

#hyperparameter tuning
parametergrid= {"criterion" : ("gini", "entropy"),"max_depth":(2, 3, 4, 5, 6, 7),"min_samples_leaf":(1,2)
}
clf = GridSearchCV(DecisionTreeClassifier(random_state=40), parametergrid)
clf.fit(X_train, y_train)
clf.best_estimator_
clf.best_params_
clf.best_score_

#po tuningu
model_dt = clf
model_dt.score(X_train,y_train)
#zlepseni uspesnosti testovaci mnoziny
model_dt.score(X_test,y_test)

#model evaluation

#Logisticka regrese
plot_confusion_matrix(model_lr, X_test, y_test)
model_lr.score(X_test,y_test)

#Decision tree
plot_confusion_matrix(model_dt, X_test, y_test)
model_dt.score(X_test,y_test)

#zakresleni Decision Tree
decision_tree = DecisionTreeClassifier(max_depth = 3)
treemodel = decision_tree.fit(X, y)
plt.figure(figsize=(18,18))
plot_tree(treemodel,feature_names = predictors,
               class_names=["left","stayed"],
               filled = True,fontsize=10)
plt.savefig('tree.png')

#ROC curve
#Logisticka regrese
y_test_probs = model_lr.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_test_probs)
roc_auc = roc_auc_score(y_test, y_test_probs)

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.5f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

#decision tree
y_test_probs = model_dt.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_test_probs)
roc_auc = roc_auc_score(y_test, y_test_probs)

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.3f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()