#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import numpy as np
import pandas as pd
import seaborn as sns
import warnings
import matplotlib.ticker as mtick
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from IPython.display import display 
from matplotlib.pylab import * #importing numpu and pyplot
from sklearn.model_selection import train_test_split
from json import dumps
from imblearn.combine import SMOTEENN


warnings.filterwarnings("ignore")

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data = pd.read_csv(r"C:\Users\DELL\Desktop\Capstone project\CS train (2).csv")


# In[3]:


data.head()


# In[4]:


data.shape


# In[5]:


data.columns


# In[6]:


data.dtypes


# In[7]:


data.describe()


# In[8]:


data['Churn'].value_counts().plot(kind='barh', figsize=(8, 6))
plt.xlabel("Count", labelpad=14)
plt.ylabel("Target Variable", labelpad=14)
plt.title("Count of TARGET Variable per category", y=1.02);


# In[9]:


100*data['Churn'].value_counts()/len(['Churn'])


# In[10]:


data['Churn'].value_counts()


# In[11]:


data.info(verbose = True) 


# In[12]:


missing = pd.DataFrame((data.isnull().sum())*100/data.shape[0]).reset_index()
plt.figure(figsize=(16,5))
ax = sns.pointplot('index',0,data=missing)
plt.xticks(rotation =90,fontsize =7)
plt.title("Percentage of Missing values")
plt.ylabel("PERCENTAGE")
plt.show()


# In[13]:


data1 = data.copy()


# In[14]:


data1.TotalCharges = pd.to_numeric(data1.TotalCharges, errors='coerce')
data1.isnull().sum()


# In[15]:


data1.loc[data1 ['TotalCharges'].isnull() == True]


# In[16]:


data1.dropna(how = 'any', inplace = True)


# In[17]:


print(data1['tenure'].max())


# In[18]:


labels = ["{0} - {1}".format(i, i + 11) for i in range(1, 72, 12)]

data1['tenure_group'] = pd.cut(data1.tenure, range(1, 80, 12), right=False, labels=labels)


# In[19]:


data1['tenure_group'].value_counts()


# In[20]:


#drop column customerID and tenure
data1.drop(columns= ['customerID','tenure'], axis=1, inplace=True)


# In[21]:


data1.head()


# In[22]:


for i, predictor in enumerate(data1.drop(columns=['Churn', 'TotalCharges', 'MonthlyCharges'])):
    plt.figure(i)
    sns.countplot(data=data1, x=predictor, hue='Churn')


# In[23]:


data1['Churn'] = np.where(data1.Churn == 'Yes',1,0)


# In[24]:


data1.head()


# In[25]:


data1_dummies = pd.get_dummies(data1)
data1_dummies.head()


# In[26]:


sns.lmplot(data=data1_dummies, x='MonthlyCharges', y='TotalCharges', fit_reg=False)


# In[27]:


Mth = sns.kdeplot(data1_dummies.MonthlyCharges[(data1_dummies["Churn"] == 0) ],
                color="Red", shade = True)
Mth = sns.kdeplot(data1_dummies.MonthlyCharges[(data1_dummies["Churn"] == 1) ],
                ax =Mth, color="Blue", shade= True)
Mth.legend(["No Churn","Churn"],loc='upper right')
Mth.set_ylabel('Density')
Mth.set_xlabel('Monthly Charges')
Mth.set_title('Monthly charges by churn')


# In[28]:


Tot = sns.kdeplot(data1_dummies.TotalCharges[(data1_dummies["Churn"] == 0) ],
                color="Red", shade = True)
Tot = sns.kdeplot(data1_dummies.TotalCharges[(data1_dummies["Churn"] == 1) ],
                ax =Tot, color="Blue", shade= True)
Tot.legend(["No Churn","Churn"],loc='upper right')
Tot.set_ylabel('Density')
Tot.set_xlabel('Total Charges')
Tot.set_title('Total charges by churn')


# In[29]:


plt.figure(figsize=(20,8))
data1_dummies.corr()['Churn'].sort_values(ascending = False).plot(kind='bar')


# In[30]:


plt.figure(figsize=(12,12))
sns.heatmap(data1_dummies.corr(), cmap="Paired")


# In[31]:


new_df1_target0=data1.loc[data1["Churn"]==0]
new_df1_target1=data1.loc[data1["Churn"]==1]


# In[32]:


def uniplot(df,col,title,hue =None):
    
    sns.set_style('whitegrid')
    sns.set_context('talk')
    plt.rcParams["axes.labelsize"] = 20
    plt.rcParams['axes.titlesize'] = 22
    plt.rcParams['axes.titlepad'] = 30
    
    
    temp = pd.Series(data = hue)
    fig, ax = plt.subplots()
    width = len(df[col].unique()) + 7 + 4*len(temp.unique())
    fig.set_size_inches(width , 8)
    plt.xticks(rotation=45)
    plt.yscale('log')
    plt.title(title)
    ax = sns.countplot(data = df, x= col, order=df[col].value_counts().index,hue = hue,palette='bright') 
        
    plt.show()


# In[33]:


uniplot(new_df1_target1,col='Partner',title='Distribution of Gender for Churned Customers',hue='gender')
uniplot(new_df1_target0,col='Partner',title='Distribution of Gender for Non Churned Customers',hue='gender')
uniplot(new_df1_target1,col='PaymentMethod',title='Distribution of PaymentMethod for Churned Customers',hue='gender')
uniplot(new_df1_target1,col='Contract',title='Distribution of Contract for Churned Customers',hue='gender')
uniplot(new_df1_target1,col='TechSupport',title='Distribution of TechSupport for Churned Customers',hue='gender')
uniplot(new_df1_target1,col='SeniorCitizen',title='Distribution of SeniorCitizen for Churned Customers',hue='gender')



# In[34]:


data1_dummies.to_csv('tel_churn.csv')


# In[35]:


df=pd.read_csv("tel_churn.csv")
df.head()


# In[36]:


df=df.drop('Unnamed: 0',axis=1)


# In[37]:


x=df.drop('Churn',axis=1)
x


# In[38]:


y=df['Churn']
y


# In[39]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


# In[40]:


from sklearn.linear_model import LogisticRegression
  
model_lr = LogisticRegression(random_state = 18).fit(x_train, y_train)


# In[41]:


model_lr.fit(x_train,y_train)


# In[42]:


y_pred=model_lr.predict(x_test)
y_pred


# In[43]:


model_lr.score(x_test,y_test)


# In[44]:


print(classification_report(y_test, y_pred, labels=[0,1]))


# In[45]:


model_dt=DecisionTreeClassifier(criterion = "gini",random_state = 100,max_depth=6, min_samples_leaf=8)


# In[46]:


model_dt.fit(x_train,y_train)


# In[47]:


y_pred=model_dt.predict(x_test)
y_pred


# In[48]:


model_dt.score(x_test,y_test)


# In[49]:


print(classification_report(y_test, y_pred, labels=[0,1]))


# In[50]:


from sklearn.ensemble import RandomForestClassifier

model_rf=RandomForestClassifier(n_estimators=100, criterion='gini', random_state = 100,max_depth=6, min_samples_leaf=8)


# In[51]:


model_rf.fit(x_train,y_train)


# In[52]:


y_pred=model_rf.predict(x_test)
model_rf.score(x_test,y_test)


# In[53]:


print(classification_report(y_test, y_pred, labels=[0,1]))


# In[54]:


y_predict = model_lr.predict(x_test)


# In[55]:


model_score_r1 = model_lr.score(x_test, y_test)


# In[56]:


print(model_score_r1)
print(metrics.classification_report(y_test, y_predict))


# In[57]:


print(metrics.confusion_matrix(y_test, y_predict))


# In[58]:


# Applying PCA
from sklearn.decomposition import PCA
pca = PCA(0.9)
x_train_pca = pca.fit_transform(x_train)
x_test_pca = pca.transform(x_test)
explained_variance = pca.explained_variance_ratio_


# In[59]:


X = data.iloc[:, :1].values

model_lr = LogisticRegression(random_state = 18).fit(x_train, y_train)


# In[60]:


model_lr.fit(x_train_pca,y_train)


# In[61]:


y_predict_pca = model_lr.predict(x_test_pca)


# In[62]:


model_score_r_pca = model_lr.score(x_test_pca, y_test)


# In[63]:


print(model_score_r_pca)
print(metrics.classification_report(y_test, y_predict_pca))


# With PCA, we couldn't see any better results, hence let's finalise the model which was created by Logistic Regression,  

# In[64]:


import pickle


# In[65]:


filename = 'model_churn.pkl'


# In[66]:


pickle.dump(model_rf, open(filename, 'wb'))


# In[67]:


load_model = pickle.load(open(filename, 'rb'))


# In[68]:


model_score_r1 = load_model.score(x_test, y_test)


# In[69]:


model_score_r1


# In[ ]:





# In[ ]:




