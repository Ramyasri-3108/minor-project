
# coding: utf-8

#### DataSet

# In[1]:

import requests


# In[2]:

from bs4 import BeautifulSoup


# In[3]:

reviews = []
for i in range(92):
    r = requests.get('https://www.scoi.com/about-us/patient-reviews?page={}'.format(i))
    soup = BeautifulSoup(r.text, 'html.parser')
    divs = soup.findAll(class_='views-field views-field-nothing review-text')
    for div in divs:
        reviews.append(div.find('span').text)


# In[4]:

import pandas as pd
import numpy as np


# In[5]:

df=pd.DataFrame(np.array(reviews),columns=['review'])


# In[6]:

len(df['review'])


# In[7]:

df['word_count'] = df['review'].apply(lambda x: len(x.split()))


# In[8]:

df['char_count'] = df['review'].apply(lambda x: len(x))


# In[9]:

df


# In[10]:

def average_words(x):
    words = x.split()
    return sum(len(word) for word in words)/len(words)


# In[11]:

df['average_word_length']=df['review'].apply(lambda x: average_words(x))


# In[12]:

from nltk.corpus import stopwords


# In[13]:

stop_words=stopwords.words('english')


# In[14]:

df['stopword_count']=df['review'].apply(lambda x: len([word for word in x.split() if word.lower() in stop_words]))


# In[15]:

df['stopword_rate']=df['stopword_count']/df['word_count']


# In[16]:

df['lowercase']=df['review'].apply(lambda x: " ".join(word.lower() for word in x.split()))


# In[17]:

df['punctuation']=df['lowercase'].str.replace('[^\w\s]','')


# In[18]:

df['stopwords'] = df['punctuation'].apply(lambda x: " ".join(word for word in x.split() if word not in stop_words))


# In[19]:

#pd.Series(" ".join(df['stopwords']).split()).value_counts()[:30]


# In[20]:

df


# In[21]:

df['cleanreview']=df['stopwords']


# In[22]:

df


#### Sentiment Analysis using TextBlob

# In[23]:

from textblob import TextBlob


# In[24]:

df['polarity']=df['cleanreview'].apply(lambda x: TextBlob(x).sentiment[0])


# In[25]:

df['subjectivity']=df['cleanreview'].apply(lambda x: TextBlob(x).sentiment[1])


# In[26]:

df


# In[27]:

df=df.drop(['lowercase','punctuation','average_word_length','stopwords','subjectivity'],axis=1)


# In[28]:

df


# In[29]:

type(df['polarity'])


# In[30]:

ser=pd.Series(df['polarity'])


# In[31]:

#ser


# In[34]:

l=[]
for ele in ser:
    l.append(ele)
for e in l:
    if(e> 0):
        print('positive')
    elif(e< 0):
        print('negative')
    else:
        print('neutral')


# In[ ]:




# In[75]:

if __name__ == "__main__" :
    pos_count=0
    neg_count=0
    neu_count=0
    for e in l:
        if(e > 0):
           #print("Positive")
            pos_count=pos_count+1
        elif(e < 0):
           #print("Negative")
            neg_count=neg_count+1
        else:
           #print("Neutral") 
           neu_count=neu_count+1
    tot_count=pos_count+neg_count+neu_count
    print("total no of reviews",tot_count)
    print("total no of positive reviews",pos_count)
    print("total no of negative reviews",neg_count)
    print("total no of neutral reviews",neu_count)
    
    pospercent=(pos_count/tot_count)*100
    negpercent=(neg_count/tot_count)*100
    neupercent=(neu_count/tot_count)*100
    print("postive percent ",pospercent)
    print("negative percent ",negpercent)
    print("neutral percent ",neupercent)


# In[80]:

import matplotlib.pyplot as plt
#plt.figure(figsize=(5,5))
labels=["Positive","Negative","Neutral"]
values=[pospercent,negpercent,neupercent]
colors = ['yellowgreen', 'lightcoral','lightskyblue']
explode=(0.05,0.02,0)
plt.pie(values,labels=labels,colors=colors,shadow=True,autopct="%.1f%%",explode=explode)
plt.title("Analysis of Patient Opinions On Southern California Orthopedic Institute using TEXTBLOB")
plt.show()


# In[37]:




# In[ ]:



