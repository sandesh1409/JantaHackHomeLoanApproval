import pandas as pd


path = '/home/sandynote/Desktop/JantaHack3/train_ctrUa4K.csv'
df = pd.read_csv(path)

df.drop(['Loan_ID'], axis = 1, inplace=True)
df['Dependents'] = df['Dependents'].str.replace(r'\D','').astype(float)

df.Married.fillna(method='bfill', inplace=True)
cat_df = df.select_dtypes(include=['object'])
cat_df = cat_df.fillna('Unknown')
Gender = pd.get_dummies(cat_df.Gender)
Married = pd.get_dummies(cat_df.Married)
Education = pd.get_dummies(cat_df.Education)
Self_Employed = pd.get_dummies(cat_df.Self_Employed)
Property_Area = pd.get_dummies(cat_df.Property_Area)
#Loan_Status = pd.get_dummies(cat_df.Loan_Status)

catvar_df = pd.concat([Gender,Married,Education,Self_Employed,Property_Area], axis=1)


df.Dependents.fillna(0, inplace=True)
df.ApplicantIncome.fillna(df.ApplicantIncome.mean, inplace=True)
df.LoanAmount.fillna(method='bfill', inplace=True)
df.Loan_Amount_Term.fillna(df.Loan_Amount_Term.mean, inplace=True)
df.Credit_History.fillna(1, inplace=True)
numeric_df = df.select_dtypes(exclude=['object'])

df1 = pd.concat([numeric_df,catvar_df], axis=1)
y = cat_df['Loan_Status']
x = df1

from sklearn.model_selection  import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=41)

from sklearn.svm import SVC
kernal = input("Enter a kernal (linear/rbf/poly/sigmoid) : ")
c = 10
g = 5

if kernal == 'linear':
    clf = SVC(kernel='linear',C=c, gamma=g)
elif kernal == 'rbf':
    clf = SVC(kernel='rbf',C=c, gamma=g)
elif kernal == 'poly':
    clf = SVC(kernel='poly',C=c, gamma=g)
elif kernal == 'sigmoid': 
    clf = SVC(kernel='sigmoid',C=c, gamma=g)


clf.fit(x_train,y_train)

from sklearn.metrics import accuracy_score
y_pred1 = clf.predict(x_train)
print(accuracy_score(y_train, y_pred1))

y_pred2 = clf.predict(x_test)
print(accuracy_score(y_test, y_pred2))









df_test = pd.read_csv('/home/sandynote/Desktop/JantaHack3/test_lAUu6dG.csv')
Loan_ID = df_test.Loan_ID
df_test.drop(['Loan_ID'], axis = 1, inplace=True)
df_test['Dependents'] = df_test['Dependents'].str.replace(r'\D','').astype(float)

cat_df_test = df_test.select_dtypes(include=['object'])
cat_df_test = cat_df_test.fillna('Unknown')

Gender_test = pd.get_dummies(cat_df_test.Gender)
Married_test = pd.get_dummies(cat_df_test.Married)
Education_test = pd.get_dummies(cat_df_test.Education)
Self_Employed_test = pd.get_dummies(cat_df_test.Self_Employed)
Property_Area_test = pd.get_dummies(cat_df_test.Property_Area)

catvar_df_test = pd.concat([Gender_test,Married_test,Education_test,Self_Employed_test,Property_Area_test], axis=1)

df_test.Dependents.fillna(0, inplace=True)
df_test.LoanAmount.fillna(method='bfill', inplace=True)
df_test.Loan_Amount_Term.fillna(df.Loan_Amount_Term.mean, inplace=True)
df_test.Credit_History.fillna(1, inplace=True)
numeric_df_test = df_test.select_dtypes(exclude=['object'])

df2 = pd.concat([numeric_df_test,catvar_df_test], axis=1)

y_pred = clf.predict(df2)
y_pred = pd.DataFrame(y_pred)
y_pred.columns = ['Loan_Status']

answer = pd.concat([Loan_ID,y_pred],axis=1)

#answer.to_csv(r'/home/sandynote/Desktop/JantaHack3/predicted_Loan_Status.csv', index = False)




