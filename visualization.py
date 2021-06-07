import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder

df= pd.read_csv('loan_data_set.csv')
df.head()
df.columns
df.info()
df.shape # Dataset Consists of 614 rows and 13 columns 
df.describe()

#UNIVARIATE ANALYSIS
graduated = df['Education'].value_counts()['Graduate']
not_graduated =df['Education'].value_counts()['Not Graduate']
print("Total graduates are : ",graduated)
print("Total non graduates are: ",not_graduated)  
sns.countplot(x='Education',data=df)

approved_loan = df['Loan_Status'].value_counts()['Y']
not_approved_loan =df['Loan_Status'].value_counts()['N']
print("Total approved loans are : ",approved_loan)
print("Total non approved loans are: ",not_approved_loan)  
sns.countplot(x="Loan_Status",data=df)

print(df['Married'].value_counts())
sns.countplot(x='Married',data=df)
#Near about 65% of the applicants in the dataset are married.

print(df['Property_Area'].value_counts())
sns.countplot(x='Property_Area',data=df)

print(df['Self_Employed'].value_counts()) # Only 16.5 % people are self employed
sns.countplot(x='Self_Employed',data=df)

print(df['Credit_History'].value_counts()) 
sns.countplot(x='Credit_History',data=df)

sns.countplot(x='Gender',data=df) # Approx 100 customers are female and rest of them are male

#BIVARIATE ANALYSIS
sns.countplot(x="Loan_Status", hue="Gender", data=df) 
sns.countplot(x="Loan_Status", hue="Education", data=df) #Graduate people are more eligible for loan 
sns.countplot(x="Loan_Status",hue="Married",data=df) #The proportion of married gettting the loan approved is more
pd.crosstab(df ['Credit_History'], df ['Loan_Status'], margins=True) #people with a credit history as 1 likely to get their loans approved.
sns.countplot(x="Loan_Status",hue="Self_Employed",data=df) 
sns.countplot(x="Loan_Status",hue="Property_Area",data=df)

# Justifying the correlation between diffrent numeric attributes
cr= df.corr()
cr

sns.heatmap(cr)
df.groupby('Loan_Status')['ApplicantIncome'].mean().plot.bar()
sns.boxplot(x=df['ApplicantIncome'])

# Finding outliers in the ApllicantIncome
q1 =df['ApplicantIncome'].quantile(0.25)
q3=df['ApplicantIncome'].quantile(0.75)
IQR= q3-q1
lower_Range = q1 -(1.5 * IQR)
upper_Range = q3 + (1.5 * IQR)
df[df['ApplicantIncome']>upper_Range] # several outliers as expected from the above boxplot
df[df['ApplicantIncome']<lower_Range] # There are none
#  Hiatogram of the LoanAmount column 
df['LoanAmount'].hist(bins=50)
df.boxplot(column='LoanAmount', by = 'Gender')
df.boxplot(column='ApplicantIncome', by = 'Education')