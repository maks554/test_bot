#!/usr/bin/env python
# coding: utf-8

# In[10]:


import joblib
import telebot
from sklearn.neighbors import BallTree
from sklearn.base import BaseEstimator
import numpy as np


# In[2]:


token = '1608570088:AAHGWdarXJrIlowm6shDywfWLOBdpGU6oUo'


# In[6]:


def softmax(x):
    proba = np.exp(-x)
    return proba / sum(proba)

class NeighborSampler(BaseEstimator):
    
    def __init__(self, k=5, temperature=1.0):
        self.k = k
        self.temperature = temperature
    def fit(self, X, y):
        self.tree_ = BallTree(X)
        self.y_ = np.array(y)
    def predict(self, X, random_state=None):
        distances, indices = self.tree_.query(X, return_distance=True,
                                             k=self.k)
        result = []
        for distance, index in zip(distances, indices):
            result.append(np.random.choice(index,
                                          p=softmax(distance*self.temperature)))
        return self.y_[result]    


# In[7]:


pipe = joblib.load('pipe.pkl')


# In[14]:


bot = telebot.TeleBot(token)

@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    bot.reply_to(message, f'Привет, {message.from_user.first_name}')

@bot.message_handler(func=lambda message:True)
def send_message(message):
    bot.reply_to(message, pipe.predict([message.text.lower()])[0])


# In[15]:


bot.polling()


# In[ ]:




