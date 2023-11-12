## EXP NO. 06
# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn
### Date : 12.10.23

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2. Upload and read the dataset.
3. Check for any null values using the isnull() function.
4. From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
5. Find the accuracy of the model and predict the required values by importing the required module from sklearn. 

## Program:
```

Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: D. Vinitha Naidu
RegisterNumber: 212222230175

import pandas as pd
data=pd.read_csv('/content/Employee.csv')

data.head()

data.info()

data.isnull().sum()

data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

data["salary"]=le.fit_transform(data["salary"])
data.head()

x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y=data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:
### Data Head:
![image](https://github.com/ShanmathiShanmugam/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/121243595/365b56a8-1c8d-4322-bc84-46d08fa30a9d)

### Dataset Info:
![image](https://github.com/ShanmathiShanmugam/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/121243595/ee654ffa-c3e2-42a9-b93a-9a82f7bab8d2)

### Null dataset:
![image](https://github.com/ShanmathiShanmugam/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/121243595/21178019-f435-4bd6-9cbd-e4d6cbbc1c6c)

### Values Count in Left Column:
![image](https://github.com/ShanmathiShanmugam/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/121243595/43a7745e-2388-44a5-9ed8-113106ac0bb8)

### Dataset transformed head:
![image](https://github.com/ShanmathiShanmugam/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/121243595/cc57beb6-a992-49ee-901e-9727e1cf45e8)

### x.head():
![image](https://github.com/ShanmathiShanmugam/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/121243595/80c5b353-e69f-4dbf-a74a-170e21527735)

### Accuracy: 
![image](https://github.com/ShanmathiShanmugam/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/121243595/2d2d5943-14e4-490d-90b8-78dc9c8d5891)

### Data Prediction:
![image](https://github.com/ShanmathiShanmugam/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/121243595/64782b05-022a-4fcd-8073-c7baa23346c8)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
