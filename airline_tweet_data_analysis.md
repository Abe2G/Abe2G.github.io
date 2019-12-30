
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
data.head()
```

```python
data=data[['tweet_id','text','airline_sentiment','airline_sentiment_confidence','negativereason','airline','retweet_count','tweet_created']]
```


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
| airline | negative | neutral | positive |
| ------- | -------- | ------- | -------- |
| American | 1960 | 463 | 336 |
| Delta | 955 | 723 | 544 |
| Southwest | 1186 | 664 | 570 |
| US Airways | 2263 | 381 | 269 |
| United | 2633 | 697 | 492 |
| Virgin America | 181 | 171 | 152 |   


```python
negative_tweet=data[(data['airline_sentiment']=='negative')]
negative_tweet.head()
```

Most common words in negative tweets


```python
negative_tweet.airline.value_counts() #counts number of negative rate for each airline to identify worse airway of 2015
```




    United            2633
    US Airways        2263
    American          1960
    Southwest         1186
    Delta              955
    Virgin America     181
    Name: airline, dtype: int64




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


![png](output_12_0.png)


The plot is showing wich airline service is more tweeted for negative sentiment and reason for negativity.

Lets look at posetive comments to understand services on which customers are more satisfied.


```python
posetive_tweet=data[(data['airline_sentiment']=='positive')]
pos_tweet_words=posetive_tweet.text.values.tolist()
pos_words=' '.join([text for text in pos_tweet_words])
plotWords(pos_words)
```


![png](output_14_0.png)


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


![png](output_17_0.png)



```python
airline_neg_reason=negative_tweet.groupby('airline')['negativereason'].value_counts()
airline_neg_reason.unstack()
```
|Airlines|Bad Flight|Can't Tell|Cancelled Flight|Customer Service Issue|Damaged Luggage|Flight Attendant Complaints|Flight Booking Problems|Late Flight|Lost Luggage|longlines|
|---|---|---|---|---|---|---|---|---|---|---|
|American|87|198|246|768|12|87|130|249|149|34|
|Delta|64|186|51|199|11|60|44|269|57|14|
|Southwest|90|159|162|391|14|38|61|152|90|29|
|US Airways|104|246|189|811|11|123|122|453|154|50|
|United|216|379|181|681|22|168|144|525|269|48|
|Virgin America|19|22|18|60|4|5|28|17|5|3|
 


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

# plt.figure(figsize=(6, 8))
# splot = sns.barplot(data=df, x = 'sex', y = 'total_bill', ci = None)
# for p in splot.patches:
# splot.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()),
#                ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
```


![png](output_19_0.png)


The plot and table above interstingly depicts the United, US, and American airlines has worest service than Delta, Virgin America, and Southwest airlines. Except, Delta and Virgin America airways, the rest four has no good customer handling and United and US airways also mostly late on flight time. Comaratively, Virgin America is good than other and then Delta is next choise.

# Does flight time has relation to negative reason?

We will focus on top three airlines with negative sentiment


```python
#time based analysis
data['tweet_created']=data['tweet_created'].astype('datetime64[ns]') ## conversion of data type to datetime
data.head()
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
      <th>tweet_id</th>
      <th>text</th>
      <th>airline_sentiment</th>
      <th>airline_sentiment_confidence</th>
      <th>negativereason</th>
      <th>airline</th>
      <th>retweet_count</th>
      <th>tweet_created</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>570306133677760513</td>
      <td>@VirginAmerica What @dhepburn said.</td>
      <td>neutral</td>
      <td>1.0000</td>
      <td>NaN</td>
      <td>Virgin America</td>
      <td>0</td>
      <td>2015-02-24 19:35:52</td>
    </tr>
    <tr>
      <th>1</th>
      <td>570301130888122368</td>
      <td>@VirginAmerica plus you've added commercials t...</td>
      <td>positive</td>
      <td>0.3486</td>
      <td>NaN</td>
      <td>Virgin America</td>
      <td>0</td>
      <td>2015-02-24 19:15:59</td>
    </tr>
    <tr>
      <th>2</th>
      <td>570301083672813571</td>
      <td>@VirginAmerica I didn't today... Must mean I n...</td>
      <td>neutral</td>
      <td>0.6837</td>
      <td>NaN</td>
      <td>Virgin America</td>
      <td>0</td>
      <td>2015-02-24 19:15:48</td>
    </tr>
    <tr>
      <th>3</th>
      <td>570301031407624196</td>
      <td>@VirginAmerica it's really aggressive to blast...</td>
      <td>negative</td>
      <td>1.0000</td>
      <td>Bad Flight</td>
      <td>Virgin America</td>
      <td>0</td>
      <td>2015-02-24 19:15:36</td>
    </tr>
    <tr>
      <th>4</th>
      <td>570300817074462722</td>
      <td>@VirginAmerica and it's a really big bad thing...</td>
      <td>negative</td>
      <td>1.0000</td>
      <td>Can't Tell</td>
      <td>Virgin America</td>
      <td>0</td>
      <td>2015-02-24 19:14:45</td>
    </tr>
  </tbody>
</table>
</div>




```python
data['tweet_created_date']=data.tweet_created.dt.date
data['tweet_created_weekday_name']=data.tweet_created.dt.weekday_name
data['tweet_created_hour']=data.tweet_created.dt.hour
data.head()
```


```python
negative_tweet=data[(data['airline_sentiment']=='negative')]
neg_by_wkday = negative_tweet.groupby(['tweet_created_weekday_name']).negativereason.value_counts()
neg_by_wkday.unstack()
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
      <th>negativereason</th>
      <th>Bad Flight</th>
      <th>Can't Tell</th>
      <th>Cancelled Flight</th>
      <th>Customer Service Issue</th>
      <th>Damaged Luggage</th>
      <th>Flight Attendant Complaints</th>
      <th>Flight Booking Problems</th>
      <th>Late Flight</th>
      <th>Lost Luggage</th>
      <th>longlines</th>
    </tr>
    <tr>
      <th>tweet_created_weekday_name</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Friday</th>
      <td>71</td>
      <td>119</td>
      <td>52</td>
      <td>207</td>
      <td>4</td>
      <td>41</td>
      <td>47</td>
      <td>237</td>
      <td>57</td>
      <td>14</td>
    </tr>
    <tr>
      <th>Monday</th>
      <td>121</td>
      <td>316</td>
      <td>227</td>
      <td>791</td>
      <td>13</td>
      <td>105</td>
      <td>122</td>
      <td>399</td>
      <td>207</td>
      <td>45</td>
    </tr>
    <tr>
      <th>Saturday</th>
      <td>55</td>
      <td>117</td>
      <td>100</td>
      <td>313</td>
      <td>9</td>
      <td>42</td>
      <td>56</td>
      <td>135</td>
      <td>53</td>
      <td>21</td>
    </tr>
    <tr>
      <th>Sunday</th>
      <td>103</td>
      <td>175</td>
      <td>210</td>
      <td>547</td>
      <td>11</td>
      <td>105</td>
      <td>73</td>
      <td>360</td>
      <td>121</td>
      <td>39</td>
    </tr>
    <tr>
      <th>Thursday</th>
      <td>57</td>
      <td>108</td>
      <td>31</td>
      <td>197</td>
      <td>5</td>
      <td>37</td>
      <td>54</td>
      <td>124</td>
      <td>38</td>
      <td>15</td>
    </tr>
    <tr>
      <th>Tuesday</th>
      <td>112</td>
      <td>241</td>
      <td>173</td>
      <td>615</td>
      <td>19</td>
      <td>102</td>
      <td>128</td>
      <td>268</td>
      <td>160</td>
      <td>27</td>
    </tr>
    <tr>
      <th>Wednesday</th>
      <td>61</td>
      <td>114</td>
      <td>54</td>
      <td>240</td>
      <td>13</td>
      <td>49</td>
      <td>49</td>
      <td>142</td>
      <td>88</td>
      <td>17</td>
    </tr>
  </tbody>
</table>
</div>




```python
neg_by_wkday = neg_by_wkday.unstack().plot(kind='line',figsize=(10,5),rot=0,title="Negetive Reasons by Day of Week")
neg_by_wkday.set_xlabel("Day of Week")
neg_by_wkday.set_ylabel("Negative Reason")
```




    Text(0, 0.5, 'Negative Reason')




![png](output_24_1.png)


The plot clearly depicts expect Friday, Saturday, Thursady and Wednesday flights are comaratively good. Monday, Sunday and Tuesday flights has customer service problem and are mostly late (the green lines also shows that probability of cancelation of flights by Monday, Sunday and Tuesday is high).


```python
neg_by_time = negative_tweet.groupby(['tweet_created_hour']).negativereason.value_counts()

neg_by_time = neg_by_time.unstack().plot(kind='line',figsize=(10, 5),title="Negetive Reasons by Hour")
neg_by_time.set_xlabel("Time")
neg_by_time.set_ylabel("Negative Reason")
```




    Text(0, 0.5, 'Negative Reason')




![png](output_26_1.png)


Time based analysis is showing something good look to optimize airline service.
<ul>
    <li>Relatively good customer sutisfaction period (6 A.M to 10 A.M) </li>
    <li>7:AM to 9:AM less customer service issue </li>
    <li>1:pm to 7:pm almost no late flight</li>
    <li>9:AM no cancelled flight</li>
  </ul>

<b>Flights at time range 0:00 A.M -03:00 A.M and 04:00 PM - 06:00 PM are with high customer dististfaction.</b>
