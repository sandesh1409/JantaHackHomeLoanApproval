#Problem
#Company wants to automate the loan eligibility process (real time) 
#based on customer detail provided while filling online application form. These details are Gender, 
#Marital Status, Education, Number of Dependents, Income, Loan Amount, Credit History and others. 
#To automate this process, they have given a problem to identify the customers segments, those 
#are eligible for loan amount so that they can specifically target these customers. Here they 
#have provided a partial data set.


import pandas as pd
import numpy as np


path = '/home/sandynote/Desktop/JantaHack3/train_ctrUa4K.csv'

df = pd.read_csv(path)

df.drop(['Loan_ID'], axis = 1, inplace=True)
df['Dependents'] = df['Dependents'].str.replace(r'\D','').astype(float)

cat_df = df.select_dtypes(include=['object'])
cat_df = cat_df.fillna('Unknown')


df.Dependents.fillna(0, inplace=True)
df.ApplicantIncome.fillna(df.ApplicantIncome.mean, inplace=True)
df.LoanAmount.fillna(method='bfill', inplace=True)
df.Loan_Amount_Term.fillna(df.Loan_Amount_Term.mean, inplace=True)
df.Credit_History.fillna(1, inplace=True)
numeric_df = df.select_dtypes(exclude=['object'])

df1 = pd.concat([numeric_df,cat_df], axis=1)
y = df1['Loan_Status']
x = df1.drop(['Loan_Status'],axis=1)

from sklearn.model_selection  import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=41)


cat_features_indices = np.where(x_train.dtypes == 'object')[0]

#importing library and building model
from catboost import CatBoostClassifier
#model=CatBoostClassifier(iterations=1000, depth=3,  loss_function='Logloss')
#model.fit(x_train, y_train,cat_features=cat_features_indices)

params = {'loss_function':'Logloss', # objective function
          'eval_metric':'Accuracy', # metric
          'verbose': 200, # output to stdout info about training process every 200 iterations
          'random_seed': 2,
          'cat_features': cat_features_indices,
          'learning_rate':0.005,
          'iterations' :400,
          'depth':7,
          'task_type': 'CPU',
          'border_count': 32
         }

model = CatBoostClassifier(**params)
model.fit(x_train, y_train, # data to train on (required parameters, unless we provide X as a pool object, will be shown below)
          eval_set=(x_test, y_test), # data to validate on
          use_best_model=True, # True if we don't want to save trees created after iteration with the best validation score
          plot=True # True for visualization of the training process (it is not shown in a published kernel - try executing this code)
         );
          
from sklearn.metrics import accuracy_score
y_pred1 = model.predict(x_train)
print(accuracy_score(y_train, y_pred1))

y_pred2 = model.predict(x_test)
print(accuracy_score(y_test, y_pred2))







df_test = pd.read_csv('/home/sandynote/Desktop/JantaHack3/test_lAUu6dG.csv')
Loan_ID = df_test.Loan_ID
df_test.drop(['Loan_ID'], axis = 1, inplace=True)
df_test['Dependents'] = df_test['Dependents'].str.replace(r'\D','').astype(float)

cat_df_test = df_test.select_dtypes(include=['object'])
cat_df_test = cat_df_test.fillna('Unknown')

df_test.Dependents.fillna(0, inplace=True)
df_test.LoanAmount.fillna(method='bfill', inplace=True)
df_test.Loan_Amount_Term.fillna(df.Loan_Amount_Term.mean, inplace=True)
df_test.Credit_History.fillna(1, inplace=True)
numeric_df_test = df_test.select_dtypes(exclude=['object'])

df2 = pd.concat([numeric_df_test,cat_df_test], axis=1)


y_pred = model.predict(df2)
y_pred = pd.DataFrame(y_pred)
y_pred.columns = ['Loan_Status']

answer = pd.concat([Loan_ID,y_pred],axis=1)





#df.to_csv(r'Path where you want to store the exported CSV file\File Name.csv', index = False)
#answer.to_csv(r'/home/sandynote/Desktop/JantaHack3/predicted_Loan_Status.csv', index = False)
