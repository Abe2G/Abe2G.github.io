

### Airline Sentiment Analysis Project

#### Project Objective
*    Analysing data to visualize airline trends
        - What most posetive or negative tweets for each airline service talks about?
        - Does time of flight affect quality service?
        - Which reason commonly tweeted by customers for bad service?
        - Counting for retweeted negative tweets to shows which service is highly affecting.
*    Classifying whether the sentiment of the tweets is positive, neutral, or negative using Machine Learning Techniques, then categorizing negative tweets for their reason.

#### Data Analysis


```python
import pandas as pd ## for reading and undestanding data
import matplotlib.pyplot as plt ## for plotting data
import seaborn as sns ## another library to visualize data features
import numpy as np ## for numerical array processing
```


```python
##reading data
data=pd.read_csv('twitter-airline/Tweets.csv')
```


```python
data[['airline_sentiment','negativereason','airline','retweet_count','tweet_created']].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>airline_sentiment</th>
      <th>negativereason</th>
      <th>airline</th>
      <th>retweet_count</th>
      <th>tweet_created</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>neutral</td>
      <td>NaN</td>
      <td>Virgin America</td>
      <td>0</td>
      <td>2015-02-24 11:35:52 -0800</td>
    </tr>
    <tr>
      <th>1</th>
      <td>positive</td>
      <td>NaN</td>
      <td>Virgin America</td>
      <td>0</td>
      <td>2015-02-24 11:15:59 -0800</td>
    </tr>
    <tr>
      <th>2</th>
      <td>neutral</td>
      <td>NaN</td>
      <td>Virgin America</td>
      <td>0</td>
      <td>2015-02-24 11:15:48 -0800</td>
    </tr>
    <tr>
      <th>3</th>
      <td>negative</td>
      <td>Bad Flight</td>
      <td>Virgin America</td>
      <td>0</td>
      <td>2015-02-24 11:15:36 -0800</td>
    </tr>
    <tr>
      <th>4</th>
      <td>negative</td>
      <td>Can't Tell</td>
      <td>Virgin America</td>
      <td>0</td>
      <td>2015-02-24 11:14:45 -0800</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 14640 entries, 0 to 14639
    Data columns (total 8 columns):
    tweet_id                        14640 non-null int64
    text                            14640 non-null object
    airline_sentiment               14640 non-null object
    airline_sentiment_confidence    14640 non-null float64
    negativereason                  9178 non-null object
    airline                         14640 non-null object
    retweet_count                   14640 non-null int64
    tweet_created                   14640 non-null object
    dtypes: float64(1), int64(2), object(5)
    memory usage: 915.1+ KB
    


```python
semtiments=pd.crosstab(data.airline, data.airline_sentiment)
semtiments
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>airline_sentiment</th>
      <th>negative</th>
      <th>neutral</th>
      <th>positive</th>
    </tr>
    <tr>
      <th>airline</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>American</th>
      <td>1960</td>
      <td>463</td>
      <td>336</td>
    </tr>
    <tr>
      <th>Delta</th>
      <td>955</td>
      <td>723</td>
      <td>544</td>
    </tr>
    <tr>
      <th>Southwest</th>
      <td>1186</td>
      <td>664</td>
      <td>570</td>
    </tr>
    <tr>
      <th>US Airways</th>
      <td>2263</td>
      <td>381</td>
      <td>269</td>
    </tr>
    <tr>
      <th>United</th>
      <td>2633</td>
      <td>697</td>
      <td>492</td>
    </tr>
    <tr>
      <th>Virgin America</th>
      <td>181</td>
      <td>171</td>
      <td>152</td>
    </tr>
  </tbody>
</table>
</div>




```python
negative_tweet=data[(data['airline_sentiment']=='negative')]
negative_tweet[['airline','negativereason','text']].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>airline</th>
      <th>negativereason</th>
      <th>text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>Virgin America</td>
      <td>Bad Flight</td>
      <td>@VirginAmerica it's really aggressive to blast...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Virgin America</td>
      <td>Can't Tell</td>
      <td>@VirginAmerica and it's a really big bad thing...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Virgin America</td>
      <td>Can't Tell</td>
      <td>@VirginAmerica seriously would pay $30 a fligh...</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Virgin America</td>
      <td>Late Flight</td>
      <td>@VirginAmerica SFO-PDX schedule is still MIA.</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Virgin America</td>
      <td>Bad Flight</td>
      <td>@VirginAmerica  I flew from NYC to SFO last we...</td>
    </tr>
  </tbody>
</table>
</div>




```python
negative_tweet.airline.value_counts() #counts number of negative rate for each airline to identify worest airway of 2015
```




    United            2633
    US Airways        2263
    American          1960
    Southwest         1186
    Delta              955
    Virgin America     181
    Name: airline, dtype: int64



Most common words in negative tweets


```python
from wordcloud import WordCloud
def plotWords(words):
    wordcloud=WordCloud(width=1200, height=600, random_state=21,max_font_size=110).generate(words)
    plt.figure(figsize=(10,7))
    plt.imshow(wordcloud,interpolation="bilinear")
    plt.axis('off')
    plt.show()
```


```python
neg_tweet_words=negative_tweet.text.values.tolist()
neg_words=' '.join([text for text in neg_tweet_words])
plotWords(neg_words)
```


![png](neg_tweet.png)


The plot is showing wich airline service is more tweeted for negative sentiment and reason for negativity.

Lets look at posetive comments to understand services on which customers are more satisfied.


```python
posetive_tweet=data[(data['airline_sentiment']=='positive')]
pos_tweet_words=posetive_tweet.text.values.tolist()
pos_words=' '.join([text for text in pos_tweet_words])
plotWords(pos_words)
```


![png](pos_tweet.png)


appreciate, good, thanks, really, great, amazing, best, nice, happy, ... shows services on which customers are ok with airlines.


```python
def plot_bar(title,x_label,y_label,data):
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.set_ylabel(y_label , fontsize=12)
    ax.set_title(title, fontsize=15, fontweight='bold')
    _=data.plot(kind='bar')
```


```python
reason_count=negative_tweet['negativereason'].value_counts()
_=reason_count.plot(kind='bar')
```


![png](reason_cnt.png)



```python
airline_neg_reason=negative_tweet.groupby('airline')['negativereason'].value_counts()
```


```python
def plot_sns(x,y,data):
    sns.set(rc={'figure.figsize':(10,10)})
    ax=sns.countplot(y=y,hue=x,data=data)
    for p in ax.patches:
        patch_height = p.get_height()
        if np.isnan(patch_height):
            patch_height = 0
        ax.annotate('{}'.format(int(patch_height)), (p.get_x()+0.01, patch_height+0.5),ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
    plt.title("Distribution of negative reason for each airline")
    plt.show()
plot_sns('negativereason','airline',negative_tweet)


![png](neg_service.png)


The plot and table above interstingly depicts the United, US, and American airlines has worest service than Delta, Virgin America, and Southwest airlines. Except, Delta and Virgin America airways, the rest four has no good customer handling and United and US airways also mostly late on flight time. Comaratively, Virgin America is good than other and then Delta is next choise.

# Does flight time has relation to negative reason?

We will focus on top three airlines with negative sentiment


```python
#time based analysis
data['tweet_created']=data['tweet_created'].astype('datetime64[ns]') ## conversion of data type to datetime
data[['airline','airline_sentiment','tweet_created','negativereason']].tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>airline</th>
      <th>airline_sentiment</th>
      <th>tweet_created</th>
      <th>negativereason</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>14635</th>
      <td>American</td>
      <td>positive</td>
      <td>2015-02-22 20:01:01</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>14636</th>
      <td>American</td>
      <td>negative</td>
      <td>2015-02-22 19:59:46</td>
      <td>Customer Service Issue</td>
    </tr>
    <tr>
      <th>14637</th>
      <td>American</td>
      <td>neutral</td>
      <td>2015-02-22 19:59:15</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>14638</th>
      <td>American</td>
      <td>negative</td>
      <td>2015-02-22 19:59:02</td>
      <td>Customer Service Issue</td>
    </tr>
    <tr>
      <th>14639</th>
      <td>American</td>
      <td>neutral</td>
      <td>2015-02-22 19:58:51</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
data['tweet_created_date']=data.tweet_created.dt.date
data['tweet_created_weekday_name']=data.tweet_created.dt.weekday_name
data['tweet_created_hour']=data.tweet_created.dt.hour
data[['airline','airline_sentiment','tweet_created_weekday_name','tweet_created_hour']].tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>airline</th>
      <th>airline_sentiment</th>
      <th>tweet_created_weekday_name</th>
      <th>tweet_created_hour</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>14635</th>
      <td>American</td>
      <td>positive</td>
      <td>Sunday</td>
      <td>20</td>
    </tr>
    <tr>
      <th>14636</th>
      <td>American</td>
      <td>negative</td>
      <td>Sunday</td>
      <td>19</td>
    </tr>
    <tr>
      <th>14637</th>
      <td>American</td>
      <td>neutral</td>
      <td>Sunday</td>
      <td>19</td>
    </tr>
    <tr>
      <th>14638</th>
      <td>American</td>
      <td>negative</td>
      <td>Sunday</td>
      <td>19</td>
    </tr>
    <tr>
      <th>14639</th>
      <td>American</td>
      <td>neutral</td>
      <td>Sunday</td>
      <td>19</td>
    </tr>
  </tbody>
</table>
</div>



Negative reason of tweet vs day of the week. Which day flight has most negative indicator?


```python
negative_tweet=data[(data['airline_sentiment']=='negative')]
neg_by_wkday = negative_tweet.groupby(['tweet_created_weekday_name']).negativereason.value_counts()
```


```python
neg_by_wkday = neg_by_wkday.unstack().plot(kind='line',figsize=(10,5),rot=0,title="Negetive Reasons by Day of Week")
neg_by_wkday.set_xlabel("Day of Week")
neg_by_wkday.set_ylabel("Negative Reason")
```




    Text(0, 0.5, 'Negative Reason')




![png](week_day.png)


The plot clearly depicts expect Friday, Saturday, Thursady and Wednesday flights are comaratively good. Monday, Sunday and Tuesday flights has customer service problem and are mostly late (the green lines also shows that probability of cancelation of flights by Monday, Sunday and Tuesday is high).


```python
neg_by_time = negative_tweet.groupby(['tweet_created_hour']).negativereason.value_counts()

neg_by_time = neg_by_time.unstack().plot(kind='line',figsize=(10, 5),title="Negetive Reasons by Hour")
neg_by_time.set_xlabel("Time")
neg_by_time.set_ylabel("Negative Reason")
```




    Text(0, 0.5, 'Negative Reason')




![png](time.png)


Time based analysis is showing something good look to optimize airline service.
<ul>
    <li>Relatively good customer sutisfaction period (6 A.M to 10 A.M) </li>
    <li>7:AM to 9:AM less customer service issue </li>
    <li>1:pm to 7:pm almost no late flight</li>
    <li>9:AM no cancelled flight</li>
  </ul>

<b>Flights at time range 0:00 A.M -03:00 A.M and 04:00 PM - 06:00 PM are with high customer dististfaction.</b>
