'''
Evaluating a given recommendation using the two metrics: precision and recall.

Precision = number of items I liked that were recommended to me
			__________________________________________________
				number of items that were recommended to me

Recall = number of items I liked that were recommended to me
		 ___________________________________________________
		 			number of items I liked

Basically, precision: model of relevancy and recall: model of completeness.

Evalutaing the logistic regression classification model. 
'''

import numpy as np
import pandas as pd

from pandas import Series,DataFrame
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

bank_full = pd.read_csv('bank_full_w_dummy_vars.csv')
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
'''

X = bank_full.ix[:, (18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36)].values
y = bank_full.ix[:,17].values

LogReg = LogisticRegression()
LogReg.fit(X,y)

y_pred = LogReg.predict(X)

print(classification_report(y,y_pred))
'''
             precision    recall  f1-score   support

          0       0.90      0.99      0.94     39922
          1       0.67      0.17      0.27      5289

avg / total       0.87      0.89      0.86     45211
'''

'''
Precision: 87% meaning 87% of the items that were recommended were then liked.
Recall: 89% meaning 89% of the items that were liked were then recommended.
'''
