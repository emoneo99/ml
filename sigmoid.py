
# coding: utf-8

# In[30]:


path = "C:/User/vishwa/Desktop/Remedial/Remedial/Breast Cancer Wisconsin (Diagnostic) Data Set"


# In[33]:


file_name = "/wdbc.data"


# In[35]:


import pandas


# In[51]:


s=pandas.read_csv

columns = ["case number","diagnosis"]+["col_"+str(y) for y in xrange(1,31)]


# In[52]:


dataframe=s(path+file_name,names = columns)


# In[53]:


from sklearn.svm import SVC


# In[54]:


from sklearn.model_selection import train_test_split


# In[75]:


dataframe.head(n=5)


# In[69]:


target=dataframe["diagnosis"]


# In[70]:


changed_df=dataframe.drop(["diagnosis"],axis=1)


# In[73]:


X,X1,Y,Y1=train_test_split(changed_df,target,test_size=0.2)


# In[76]:


s=SVC(kernel="sigmoid")


# In[78]:


s.fit(X,Y)


# In[84]:


predictions=s.predict(X1)


# In[87]:


diff=[1 if s==v else 0 for s,v in zip(predictions,Y1)]


# In[89]:


(diff.count(1)+0.0)/len(diff)

