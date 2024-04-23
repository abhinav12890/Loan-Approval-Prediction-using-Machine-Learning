import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm

df = pd.read_csv('loan.csv')
print(df.head())
print(df.info())
print(df.isnull().sum())

df['loanAmount_log'] = np.log(df['LoanAmount'])
df['loanAmount_log'].hist(bins=20)
Display the histogram
plt.show()
df['TotalIncome']=df['ApplicantIncome']+ df['CoapplicantIncome']
df['TotalIncome_log'] = np.log(df['TotalIncome'])
df['TotalIncome_log'].hist(bins=20)
plt.show()

df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
df['Married'].fillna(df['Married'].mode()[0], inplace=True)
df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)
df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)

df.LoanAmount = df.LoanAmount.fillna(df.LoanAmount.mean())
# df.loanAmount_log = df.loanAmount_log.fillna(df.loanAmount_log.mean())
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)
df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)
# print('Checking if null values are fixed or not')
# print(df.isnull().sum())

x = df.iloc[:, np.r_[1:6, 8:11, 11:13]].values

# Assuming the target variable is 'Loan_Status'
y = df['Loan_Status'].values

print(x)
print(y)

print('per of missing gender is %2f%%' %((df['Gender'].isnull().sum()/df.shape[0])*100))
print('number of people who take loan as group by gender:')
print(df['Gender'].value_counts())
sns.countplot(x='Gender', data=df, hue='Gender', palette='Set1', legend=False)
plt.show()

print('number of people who take loan as group by marital status:')
print(df['Married'].value_counts())
sns.countplot(x='Married', data=df, hue='Married', palette='Set1', legend=False)
plt.show()

print('number of people who take loan as group by dependents:')
print(df['Dependents'].value_counts())
sns.countplot(x='Dependents', data=df, hue='Dependents', palette='Set1', legend=False)
plt.show()

print('number of people who take loan as group by self employed:')
print(df['Self_Employed'].value_counts())
sns.countplot(x='Self_Employed', data=df, hue='Self_Employed', palette='Set1', legend=False)
plt.show()

print('number of people who take loan as group by self loanamount:')
print(df['LoanAmount'].value_counts())
sns.countplot(x='LoanAmount', data=df, hue='LoanAmount', palette='Set1', legend=False)
plt.show()

print('number of people who take loan as group by Credit History:')
print(df['Credit_History'].value_counts())
sns.countplot(x='Credit_History', data=df, hue='Credit_History', palette='Set1', legend=False)
plt.show()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
from sklearn.preprocessing import LabelEncoder

Labelencoder_x = LabelEncoder()

for i in range(0, 5):
    X_train[:, i] = Labelencoder_x.fit_transform(X_train[:, i])
    X_train[:, 7] = Labelencoder_x.fit_transform(X_train[:, 7])

# print(X_train)

Labelencoder_y= LabelEncoder()
y_train= Labelencoder_y.fit_transform(y_train)
# print(y_train)


for i in range(0, 5):
    X_test[:, i] = Labelencoder_x.fit_transform(X_test[:, i])
    X_test[:, 7] = Labelencoder_x.fit_transform(X_test[:, 7])

# print(X_test)

Labelencoder_y= LabelEncoder()
y_test= Labelencoder_y.fit_transform(y_test)
# print(y_test)

from sklearn.preprocessing import StandardScaler
ss= StandardScaler()
X_train = ss.fit_transform(X_train)
x_test = ss.fit_transform(X_test)

from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier()
rf_clf.fit(X_train,y_train)

from  sklearn import  metrics
y_pred = rf_clf.predict(x_test)
# print("acc of random forest clf is", metrics.accuracy_score(y_pred,y_test))
# print(y_pred)

from sklearn.naive_bayes import GaussianNB
nb_clf= GaussianNB()
nb_clf.fit(X_train,y_train)
y_pred= nb_clf.predict(X_test)
print('acc of naive bayes is', metrics.accuracy_score(y_pred,y_test))