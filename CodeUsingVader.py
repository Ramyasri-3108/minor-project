
# coding: utf-8

### #Extracting Data From A Website Using BeautifulSoup

### import requests

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

df[200:300]


# In[7]:




# In[8]:

df['word_count'] = df['review'].apply(lambda x: len(x.split()))


# In[9]:

df['char_count'] = df['review'].apply(lambda x: len(x))


# In[9]:




# In[10]:

df.to_csv('/home/ramya/Desktop/reveiws.csv', index=False)


# In[11]:

def average_words(x):
    words = x.split()
    return sum(len(word) for word in words)/len(words)


# In[12]:

df['average_word_length']=df['review'].apply(lambda x: average_words(x))


# In[13]:

from nltk.corpus import stopwords


# In[14]:

stop_words=stopwords.words('english')


# In[15]:

df['stopword_count']=df['review'].apply(lambda x: len([word for word in x.split() if word.lower() in stop_words]))


# In[16]:

df['stopword_rate']=df['stopword_count']/df['word_count']


# In[17]:

df['lowercase']=df['review'].apply(lambda x: " ".join(word.lower() for word in x.split()))


# In[18]:

df['punctuation']=df['lowercase'].str.replace('[^\w\s]','')


# In[19]:

df['stopwords'] = df['punctuation'].apply(lambda x: " ".join(word for word in x.split() if word not in stop_words))


# In[20]:

#pd.Series(" ".join(df['stopwords']).split()).value_counts()[:30]


# In[20]:




# In[21]:

df['cleanreview']=df['stopwords']


# In[22]:

df


# In[23]:

from IPython.display import display
with pd.option_context('display.max_rows', 914):
    display(df['cleanreview'])


# In[24]:

df['cleanreview'].to_csv('/home/ramya/Desktop/reviews1.csv', index=False)


# In[24]:




### Sentiment Analysis Using VADER Lexicon-Based Approach 

# In[25]:

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# In[26]:

sid_obj = SentimentIntensityAnalyzer()
# sentiment_dict = sid_obj.polarity_scores(sentence)
df['vader_analysis']=df['cleanreview'].apply(lambda x: sid_obj.polarity_scores(x))
df.head()


# In[27]:

df=df.drop(['lowercase','punctuation','average_word_length','stopwords'],axis=1)


# In[28]:

df


# In[29]:

#type(df['vader_analysis'])


# In[38]:

ser=pd.Series(df['vader_analysis'])


# In[39]:

ser1=pd.Series(['compound'])


# In[48]:

l=[]
for ele in ser:
    l.append(ele['compound'])
for e in l:
    if(e>=0.05):
        print('positive')
    elif(e<=-0.05):
        print('negative')
    else:
        print('neutral')


# In[37]:




# In[37]:

if __name__ == "__main__" :
    pos_count=0
    neg_count=0
    neu_count=0
    for e in l:
        if(e >= 0.05):
           #print("Positive")
            pos_count=pos_count+1
        elif(e <= - 0.05):
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


# In[38]:

import matplotlib.pyplot as plt
#plt.figure(figsize=(5,5))
labels=["Positive","Negative","Neutral"]
values=[pospercent,negpercent,neupercent]
colors = ['yellowgreen', 'lightcoral', 'lightskyblue']
explode=(0.05,0.02,0)
plt.pie(values,labels=labels,colors=colors,shadow=True,autopct="%.1f%%",explode=explode)
plt.title("Analysis of Patient Opinions On Southern California Orthopedic Institute using VADER")
plt.show()


# In[33]:




# In[ ]:



