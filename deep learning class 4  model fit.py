#!/usr/bin/env python
# coding: utf-8

# In[8]:


#pip install tensorflow


# In[10]:


import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler , LabelEncoder


# In[ ]:


df = pd.read_csv("C:\\Users\\91895\\Downloads\\Churn_modelling.csv")


# In[ ]:


df.head()


# In[11]:


x = df.iloc[: , 3:-1]
x


# In[12]:


y = df.iloc[: , -1]


# In[13]:


y


# In[ ]:





# In[15]:


lb = LabelEncoder()


# In[18]:


x['Geography'] = lb.fit_transform(x['Geography'])
x['Gender'] = lb.fit_transform(x['Gender'])


# In[20]:


x.corr()


# In[21]:


x


# In[22]:


x_train , x_test , y_train , y_test = train_test_split(x,y, test_size = 0.2 , random_state = 42)


# In[24]:


sc = StandardScaler()


# In[25]:


x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# In[30]:


from keras.models import Sequential


# In[33]:


model = tf.keras.Sequential([
    tf.keras.layers.Dense(64 , activation = 'relu', input_dim = x_train.shape[1]) , #input layer
    tf.keras.layers.Dense(64 , activation = 'relu') , #hidden layer
    tf.keras.layers.Dense(1 , activation = 'sigmoid')
])


# In[34]:


#define the loss function and metrics


# In[35]:


loss_fn = tf.keras.losses.BinaryCrossentropy()
metrics = ['accuracy']


# In[38]:


#define the optimizer with momentum

learning_rate = 0.001
momentum = 0.9
optimizer = tf.keras.optimizers.SGD(learning_rate = learning_rate , momentum = momentum)


# In[40]:


#compile the model 
model.compile(optimizer = optimizer , loss = loss_fn , metrics = metrics)


# In[42]:


#train the model
epochs = 10
batch_size = 32


# In[43]:


model.fit(x_train , y_train , batch_size = batch_size , epochs = epochs , validation_split = 0.2)


# In[44]:


#evaluate the model
test_loss , test_acc = model.evaluate(x_test , y_test)


# In[45]:


print("test loss" , test_loss)
print("test acc" , test_acc)


# In[ ]:




