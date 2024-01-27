#!/usr/bin/env python
# coding: utf-8

# ## Importing Libraries:

# In[1]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Busines Case:

# In[2]:


df = pd.read_csv('loan_approved.csv')
df.head()


# ### Basic Checks

# ### Loading and Exploring Data:

# In[3]:


df.tail()


# In[4]:


df.info()


# In[5]:


df.describe()


# In[6]:


df.isnull().sum()


# ### Exploratory Data Analysis
# - Uni-variate
# - Bi-variate
# - Multi-variate

# In[7]:


get_ipython().system('pip install sweetviz')


# In[8]:


import sweetviz as sv
my_report = sv.analyze(df)
my_report.show_html()


# ## Data Preprocessing:

# In[10]:


df1 = df[['Gender','Married','Dependents','Education','Self_Employed','Property_Area']]
df2 = df[['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History']]


# In[11]:


plt.figure(figsize=(20,25),facecolor='white')
plotnumber=1
for column in df1:
    if plotnumber<=16:
        ax=plt.subplot(4,4,plotnumber)
        sns.countplot(x=df1[column],hue=df['Loan_Status (Approved)'])
        plt.xlabel(column,fontsize=20)
        plt.ylabel('Loan Status',fontsize =20)
    plotnumber+=1
plt.tight_layout()


# In[12]:


plt.figure(figsize=(20,25),facecolor='white')
plotnumber=1
for column in df2:
    if plotnumber<=16:
        ax=plt.subplot(4,4,plotnumber)
        sns.histplot(x=df2[column],hue=df['Loan_Status (Approved)'])
        plt.xlabel(column,fontsize=20)
        plt.ylabel('Loan Status',fontsize =20)
    plotnumber+=1
plt.tight_layout()


# In[13]:


sns.countplot(x='Gender' ,hue='Loan_Status (Approved)',data=df)


# In[14]:


df.Gender.value_counts()


# ### Data Preprocesing Pipline
# - Missing Values
# - Outliers
# - Balancing Values

# In[15]:


df.loc[df.Gender.isnull()== True, 'Gender']='Male'
df


# In[16]:


df.isnull().sum()


# In[17]:


sns.countplot(x='Dependents' ,hue='Loan_Status (Approved)', data=df)


# In[18]:


df.loc[df['Dependents'].isnull()==True]


# In[19]:


df.Dependents.value_counts()


# In[20]:


df.loc[df['Dependents'].isnull()==True,'Dependents']='3+'
df


# In[21]:


df.isnull().sum()


# In[22]:


df.Dependents.value_counts()


# In[23]:


sns.countplot(x='Married' ,hue='Loan_Status (Approved)', data=df)


# In[24]:


df.loc[df['Married'].isnull()==True]


# In[25]:


df.Married.value_counts()


# In[26]:


df.loc[df['Married'].isnull()==True,'Married']='Yes'
df


# In[27]:


df.isnull().sum()


# In[28]:


sns.countplot(x='Self_Employed',data=df)


# In[29]:


df.Self_Employed.value_counts()


# In[30]:


df.loc[df['Self_Employed'].isnull()==True,'Self_Employed']='No'
df


# In[31]:


df.isnull().sum()


# In[32]:


sns.boxplot(x='LoanAmount' , data=df)


# In[33]:


df.LoanAmount


# In[34]:


IQR=168-100
IQR


# In[35]:


upb1=168+1.5*IQR
upb1


# In[36]:


lob1=100-1.5*IQR
lob1


# In[37]:


df.loc[df.LoanAmount > upb1,'LoanAmount'].count()/df.LoanAmount.count()*100


# In[38]:


df.loc[df['LoanAmount'].isnull()==True,'LoanAmount']=np.median(df.LoanAmount.dropna(axis=0))


# In[39]:


df.loc[df['Loan_Amount_Term'].isnull()==True,'Loan_Amount_Term']=np.median(df.Loan_Amount_Term.dropna(axis=0))


# In[40]:


df.loc[df['Credit_History'].isnull()==True,'Credit_History']=0.0


# In[41]:


df.isnull().sum()


# ### Encoding
# - Rename The Target
# - Encoding The Data(ordinal)
# - Heat maps

# In[42]:


df.rename(columns={"Loan_Status (Approved)":'Loan_Status'},inplace=True)


# In[43]:


from numpy import asarray
from sklearn.preprocessing import OrdinalEncoder
Oc=OrdinalEncoder()
pd.get_dummies(df['Gender'],prefix='Gender')


# In[44]:


pd.get_dummies(df['Gender'],prefix='Gender',drop_first=True)


# In[45]:


df1=pd.get_dummies(df['Gender'],prefix='Gender',drop_first=True)
df=pd.concat([df,df1],axis=1).drop(['Gender'],axis=1)


# In[46]:


df1=pd.get_dummies(df['Married'],prefix='Married',drop_first=True)
df=pd.concat([df,df1],axis=1).drop(['Married'],axis=1)


# In[47]:


df1=pd.get_dummies(df['Education'],prefix='Education',drop_first=True)
df=pd.concat([df,df1],axis=1).drop(['Education'],axis=1)


# In[48]:


df1=pd.get_dummies(df['Property_Area'],prefix='Property_Area',drop_first=True)
df=pd.concat([df,df1],axis=1).drop(['Property_Area'],axis=1)


# In[49]:


df1=pd.get_dummies(df['Dependents'],prefix='Dependents',drop_first=True)
df=pd.concat([df,df1],axis=1).drop(['Dependents'],axis=1)


# In[50]:


df1=pd.get_dummies(df['Self_Employed'],prefix='Self_Employed',drop_first=True)
df=pd.concat([df,df1],axis=1).drop(['Self_Employed'],axis=1)


# In[51]:


df.head()


# In[52]:


df_corr=df.corr()


# In[53]:


dict={'Y':1,'N':0}
df.Loan_Status=df.Loan_Status.map(dict)


# In[54]:


sns.heatmap(df_corr,annot=True,cmap='Reds')
plt.show()


# In[55]:


df.head()


# ### Model Creation

# In[56]:


X=df.loc[:,['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
       'Loan_Amount_Term', 'Credit_History','Gender_Male',
       'Married_Yes', 'Education_Not Graduate', 'Property_Area_Semiurban',
       'Property_Area_Urban', 'Dependents_1', 'Dependents_2', 'Dependents_3+',
       'Self_Employed_Yes']]
y=df.Loan_Status


# In[57]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X, y)


# In[58]:


from sklearn.svm import SVC
model = SVC()
model.fit(X_train, y_train)


# In[59]:


y_pred=model.predict(X_train)
y_pred


# In[60]:


from sklearn.metrics import accuracy_score,classification_report, f1_score
accuracy_score(y_train,y_pred)


# In[61]:


print(classification_report(y_train,y_pred))


# ### Scaling the Data (min , max )

# In[67]:


from sklearn.preprocessing import MinMaxScaler
scale=MinMaxScaler()
df[['ApplicantIncome','CoapplicantIncome','LoanAmount']]=scale.fit_transform(df[['ApplicantIncome','CoapplicantIncome',
                          'LoanAmount']])


# In[65]:


y_train.value_counts()


# ### Cross Validation

# In[68]:


get_ipython().system('pip install imblearn')


# In[69]:


from imblearn.over_sampling import SMOTE
sm = SMOTE()


# In[70]:


y_train.value_counts()


# In[71]:


X_sm, y_sm = sm.fit_resample(X_train,y_train)


# In[72]:


y_sm.value_counts()


# In[73]:


model1 = SVC()
model1.fit(X_sm, y_sm)


# In[74]:


y_pred1=model1.predict(X_test)
accuracy_score(y_pred1,y_test)


# In[75]:


print(classification_report(y_pred1,y_test))


# In[76]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(model,X,y,cv=3, scoring = 'f1')
print(scores)
print("Cross validation Score:",scores.mean())
print("Std :",scores.std())


# In[ ]:




