# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.import pandas module and import the required data set.

2.Find the null values and count them.

3.Count number of left values.

4.From sklearn import LabelEncoder to convert string values to numerical values.

5.From sklearn.model_selection import train_test_split.

6.Assign the train dataset and test dataset.

7.From sklearn.tree import DecisionTreeClassifier.

8.Use criteria as entropy.

9.From sklearn import metrics.

10.Find the accuracy of our model and predict the require values. 

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: POOJA SRI P
RegisterNumber: 212224230197
*/
import pandas as pd

data = pd.read_csv("Employee.csv")

data.head()

data.info()

data.isnull().sum()

data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

data["salary"] = le.fit_transform(data["salary"])
data.head()

x=data[["satisfaction_level","last_evaluation","number_project", "average_montly_hours",
"time_spend_company", "Work_accident","promotion_last_5years","salary"]]
x.head()

y = data["left"]

from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn. tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt. predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)

accuracy
dt.predict([[0.5,0.8,9,260, 6,0,1,2]])
```

## Output:
![image](https://github.com/user-attachments/assets/c79a678c-a912-461d-becd-97b2ceaa36c2)

![image](https://github.com/user-attachments/assets/a9ff42d6-2d70-45d5-80bc-70d0da6470d7)

![image](https://github.com/user-attachments/assets/5632e8d8-b4c1-41a6-99d4-c902461960aa)

![image](https://github.com/user-attachments/assets/49aab963-7037-4a3d-a988-88e181a670d4)

![image](https://github.com/user-attachments/assets/20c4f7ae-0410-4a18-9a15-c0b7d818c7fe)

![image](https://github.com/user-attachments/assets/0ce20d2b-8fe9-4c58-8a5d-703017b76b4e)

![image](https://github.com/user-attachments/assets/6740a99c-56c5-4f48-b9c8-feabd29feab9)

![image](https://github.com/user-attachments/assets/ab52a77a-a2f4-4c2d-ad1a-1c45dd9c4db9)










## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
