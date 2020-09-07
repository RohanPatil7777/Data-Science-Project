#!/usr/bin/env python
# coding: utf-8

# NAME - ROHAN RAHUL PATIL
# 

# ## Loading Dataset

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


df=pd.read_csv("odi.csv")


# ## Using 'Groupsby' operation to find the average number of runs, scored by each country. 

# In[3]:


team_avg={}
for team,frame in df.groupby("bat_team"):
    team_total=sum(frame["total"])
    avg=team_total/len(frame)
    print(team+"  : "+str(avg))
    team_avg[team]=avg


# In[4]:


avg_series=pd.Series(team_avg)


# In[5]:


avg_df=pd.DataFrame(avg_series).reset_index().rename(columns={'index':'Team',0:'average runs'})


# In[6]:


avg_df


# In[7]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# ## Representing it in a Bar graph 

# In[8]:


plt.barh(avg_df["Team"],avg_df["average runs"],align='center',)
plt.figure(figsize=(80,60))
plt.show()


# ## Finding Null values

# In[9]:


df.info()


# There are no null values.

# # Dropping columns that do not contribute much to the Total

# In[10]:


df=df.drop(["mid"],axis=1)


# Date and Mid represents the same information. So, we can drop one of them.

# In[11]:


df


# In[12]:


import datetime as dt
df["date"]=pd.to_datetime(df["date"])
df["date"]=df["date"].map(dt.datetime.toordinal)


# In[13]:


df


# ### Let's check weather date is related to match total by checking only for each match instead of checking for each ball in given data

# In[14]:


df_50=df[df["overs"]==0.1]
#these way we took total and date for each match only for co-relation between total and date


# In[15]:


plt.plot(df_50["total"],df_50["date"])
plt.xlabel("total")
plt.ylabel("date")
df_50["total"].corr(df_50["date"])
#as pearson coefficient for date and total is less so better to drop date column
df=df.drop(["date"],axis=1)


# Batsman and Bowler at that ball(over) in connection to the total runs..seems far fetched. And it will decrease the dummy columns while One-Hot encoding drastically if we drop them.
# 

# In[16]:


df=df.drop(["batsman","bowler"],axis=1)


# ## Venue check

# In[17]:


venue_avg=df_50.groupby("venue").total.mean()
venues=df_50["venue"].unique()
plt.scatter(venue_avg,venues)

#some venues have high average so we cannot drop it


# # Feature Scaling using MinMaxScaler

# In[18]:


from sklearn.preprocessing import MinMaxScaler
scalar=MinMaxScaler(feature_range=(0,1))
runs=pd.DataFrame(df["runs"])
wickets=pd.DataFrame(df["wickets"])
overs=pd.DataFrame(df["overs"])
runs_last=pd.DataFrame(df["runs_last_5"])
wickets_last=pd.DataFrame(df["wickets_last_5"])
striker=pd.DataFrame(df["striker"])
non_striker=pd.DataFrame(df["non-striker"])
df["runs"]=scalar.fit_transform(runs)
df["wickets"]=scalar.fit_transform(wickets)
df["overs"]=scalar.fit_transform(overs)
df["runs_last_5"]=scalar.fit_transform(runs_last)
df["wickets_last_5"]=scalar.fit_transform(wickets_last)
df["striker"]=scalar.fit_transform(striker)
df["non-striker"]=scalar.fit_transform(non_striker)
df


# # One-Hot encoding

# In[19]:


df_new=pd.get_dummies(df,columns=["bat_team","bowl_team","venue"])
df2=df.drop(["venue"],axis=1)
df_predic=pd.get_dummies(df2,columns=["bat_team","bowl_team"]) #for predictions on a new dataset


# In[20]:


df_new


# ## Training data

# In[21]:


X=df_new.drop(["total"],axis=1)
Y=df_new["total"]
X1=df_predic.drop(["total"],axis=1)   #for predictions on a new dataset
Y1=df_predic["total"]

X1


# # Train-Test-Split

# In[22]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.33,random_state=50)
X_train_new,X_test_new,Y_train_new,Y_test_new=train_test_split(X1,Y1,test_size=0.33,random_state=50) #for predictions on a new dataset


# # Linear Regression

# In[23]:


from sklearn.linear_model import LinearRegression
linear=LinearRegression()
linear.fit(X_train,Y_train)
y_pred=linear.predict(X_test)
y_pred


# In[24]:


linear.score(X_test,Y_test)


# # Decision Tree Regression

# In[25]:


from sklearn.tree import DecisionTreeRegressor
decision=DecisionTreeRegressor()
decision.fit(X_train,Y_train)
y_pred2=decision.predict(X_test)
y_pred2


# In[26]:


decision.score(X_test,Y_test)


# In[27]:


#X_test.columns


# # Random Forest Regression

# In[28]:


from sklearn.ensemble import RandomForestRegressor
random=RandomForestRegressor(random_state=15325)
random.fit(X_train,Y_train)
y_pred3=random.predict(X_test)
y_pred3


# In[29]:


random.score(X_test,Y_test)


# ### Random Forest Regression has an accuracy of 97.59% which is the highest

# ## Creation of the Dataset

# Since we have used feature scaling, we have to fill the dataset keeping in mind that the features are scaled. Scaled values in MinMaxScaler are calculated by dividing that value with the difference in maximum and minimum of that value. Min, Max values of the scaled columns are : <br>
# runs:(0,444),wickets:(0,10),overs:(0,49.6),runs_last_5:(0,101),wickets_last_5:(0,7),striker:(0,264),non-striker:(0,149)<br>
# 
# Let's create a dataset using the match scenarios described below
# 

# Match Scenarios :<br>
#     1. India vs Pakistan : 
#       India is batting first and the score stands 286-4 in 42 overs. Runs in the last five overs are 40 and the wickets in           the last five overs are 2. Striker has made 82 and the non-striker has made 41.
#     2. Australia vs England :    
#       Australia is batting first and the score stands 244-6 in 38 overs. Runs in the last five overs are 45 and the wickets          in the last five overs are 3. Striker has made 44 and the non-striker has made 18.
#     3. West Indies vs Newzealand :   
#       Newzealand is batting first and the score stands 260-8 in 36 overs. Runs in the last five overs are 45 and the wickets          in the last five overs are 4. Striker has made 8 and the non-striker has made 6.      

# ### Final Dataset

# In[30]:


pred_test=pd.read_csv("odi_test.csv")
pred_test


# ## Prediction on a new dataset

# A new dataset has been created in order to predict on it. As mentioned in the comments on some of the code above, we are predicting on a model where we don't include venue in the features. The reason is that, by including venue column the created dataset has to have 185 columns. Creating such a dataset is a tedious task and hence we are predicting by removing venue, which requires 49 columns. Since, Random Forest Regression has the highest accuracy, let's predict using that model

# In[31]:


from sklearn.ensemble import RandomForestRegressor
random=RandomForestRegressor(random_state=15325)
random.fit(X_train_new,Y_train_new)
Total=random.predict(pred_test)
Total


# Let's round the Totals to their nearest integers.

# ## Final Totals of the aforementioned matches 

# 1. India ends up with a total of 381
# 2. Australia puts up 309 on the board
# 3. NewZealand finishes on 327

# In[ ]:




