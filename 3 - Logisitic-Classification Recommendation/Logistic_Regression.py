import pandas as pd
import numpy as np

from pandas import Series,DataFrame
from sklearn.linear_model import LogisticRegression

bank_full = pd.read_csv('bank_full_w_dummy_vars.csv')
print(bank_full.head())
'''
   age           job  marital  education default  balance housing loan  \
0   58    management  married   tertiary      no     2143     yes   no   
1   44    technician   single  secondary      no       29     yes   no   
2   33  entrepreneur  married  secondary      no        2     yes  yes   
3   47   blue-collar  married    unknown      no     1506     yes   no   
4   33       unknown   single    unknown      no        1      no   no 
   contact  day              ...              job_unknown                   \
0  unknown    5              ...                                         0   
1  unknown    5              ...                                         0   
2  unknown    5              ...                                         0   
3  unknown    5              ...                                         0   
4  unknown    5              ...                                         1   

There are 37 columns in total. The first 8 columns are known values. 
The remaining columns are dummy values.
'''

# Getting a better view of the dataset
print(bank_full.info())
'''
Data columns (total 37 columns):
age                             45211 non-null int64
job                             45211 non-null object
marital                         45211 non-null object
education                       45211 non-null object
default                         45211 non-null object
balance                         45211 non-null int64
housing                         45211 non-null object
loan                            45211 non-null object
contact                         45211 non-null object
day                             45211 non-null int64
month                           45211 non-null object
duration                        45211 non-null int64
campaign                        45211 non-null int64
pdays                           45211 non-null int64
previous                        45211 non-null int64
poutcome                        45211 non-null object
y                               45211 non-null object
y_binary                        45211 non-null int64
housing_loan                    45211 non-null int64
credit_in_default               45211 non-null int64
personal_loans                  45211 non-null int64
prev_failed_to_subscribe        45211 non-null int64
prev_subscribed                 45211 non-null int64
job_management                  45211 non-null int64
job_tech                        45211 non-null int64
job_entrepreneur                45211 non-null int64
job_bluecollar                  45211 non-null int64
job_unknown                     45211 non-null int64
job_retired                     45211 non-null int64
job_services                    45211 non-null int64
job_self_employed               45211 non-null int64
job_unemployed                  45211 non-null int64
job_maid                        45211 non-null int64
job_student                     45211 non-null int64
married                         45211 non-null int64
single                          45211 non-null int64
divorced                        45211 non-null int64

The columns age...poutcome have known values. The column 'y' is our required output and 
'y_binary' is the binary version of it. The remaining columns have dummy values.

Using 'y_binary' as the trained variable to determine if a new user will subscribe to a term deposit
upon marketing contact based on the given user attributes.
(If the model predicts the user will not subscribe, then he should not be contacted with the marketing
offer.)
'''

'''
Picking columns housing_loan...divorced for our logisitic analysis with indices 18..36 and 
picking column y_binary with index 17 as our y value (target)
'''
X = bank_full.ix[:,(18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36)].values
y = bank_full.ix[:,(17)].values


#Instantiating a logistic regression object and fitting our model to the given data
LogReg = LogisticRegression()
LogReg.fit(X,y)
print(LogReg)
'''
LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
          penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
          verbose=0, warm_start=False)
'''

'''
Each attribute/characteristic describing a new user is given by binary values 0(no) and 1(yes).
So for our new user, the binary-valued list is [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,1] for each and
every attribute. 
This means that the new user is working in job management, is single and divorced.
The attribute list is columns with indices 18-36 from the table above (output of bank_full.info())
'''

#Running the Logistic regression prediction function onto our new user's binary-valued list.
new_user = [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,1]
y_pred = LogReg.predict(new_user)
print(y_pred)
'''
array([0],dtype=int64)
The predicted answer is 0(no) meaning that if this user has a marketing contact, he would not subscribe
to a term deposit.
'''
