
## Machine Learning Hypothesis Testing Project

### Data Analysis

Planning for data analysis
<ul type='square'>
    <li>Preparing/collecting the data</li>
    <li>Undertanding data</li>
    <li>Exploring data insights</li>
    <li>Data Cleansing</li>
    <li>Feature selection</li>
    <li>Creating model</li>
    <li>Fit data to the model</li>
    <li>Evaluate the model</li>
    <li>Fine tune the model</li></ul>


```python
## required packages
import pandas as pd
import os
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
DATA_DIR='data/'
```


```python
# Load Excel dataset for analysis. I will use pandas library to work with it.
def load_data(file_name,sheet):
    return pd.read_excel(os.path.join(DATA_DIR,file_name), sheet, index_col=None)
```


```python
control_data=load_data('UdacityABtesting.xlsx','Control')
print(control_data.shape)
experment_data=load_data('UdacityABtesting.xlsx','Experiment')
print(experment_data.shape)
```

    (37, 5)
    (37, 5)
    


```python
corr = control_data.corr()
f, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr,annot=True,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x21f84888668>




![png](ml_5_1.png)



```python
corr = experment_data.corr()
f, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr,annot=True,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x21f84dfbb38>




![png](ml_6_1.png)


For both Control and Experment data, Payment feature is highly correlated (63% Control, 60% Experiment) to the target feature Enrollment. This shows that payment is very critical for enrollment prediction.
Explain what the difference is between using A/B testing to test a hypothesis (in this case showing a message window) vs using - Machine learning to learn the viability of the same effect? 
# Data analysis tasks


```python
##investigating the data
experment_data.head()
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
      <th>Date</th>
      <th>Pageviews</th>
      <th>Clicks</th>
      <th>Enrollments</th>
      <th>Payments</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Sat, Oct 11</td>
      <td>7716</td>
      <td>686</td>
      <td>105.0</td>
      <td>34.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Sun, Oct 12</td>
      <td>9288</td>
      <td>785</td>
      <td>116.0</td>
      <td>91.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Mon, Oct 13</td>
      <td>10480</td>
      <td>884</td>
      <td>145.0</td>
      <td>79.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Tue, Oct 14</td>
      <td>9867</td>
      <td>827</td>
      <td>138.0</td>
      <td>92.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Wed, Oct 15</td>
      <td>9793</td>
      <td>832</td>
      <td>140.0</td>
      <td>94.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
control_data.head()
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
      <th>Date</th>
      <th>Pageviews</th>
      <th>Clicks</th>
      <th>Enrollments</th>
      <th>Payments</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Sat, Oct 11</td>
      <td>7723</td>
      <td>687</td>
      <td>134.0</td>
      <td>70.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Sun, Oct 12</td>
      <td>9102</td>
      <td>779</td>
      <td>147.0</td>
      <td>70.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Mon, Oct 13</td>
      <td>10511</td>
      <td>909</td>
      <td>167.0</td>
      <td>95.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Tue, Oct 14</td>
      <td>9871</td>
      <td>836</td>
      <td>156.0</td>
      <td>105.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Wed, Oct 15</td>
      <td>10014</td>
      <td>837</td>
      <td>163.0</td>
      <td>64.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
control_data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 37 entries, 0 to 36
    Data columns (total 5 columns):
    Date           37 non-null object
    Pageviews      37 non-null int64
    Clicks         37 non-null int64
    Enrollments    23 non-null float64
    Payments       23 non-null float64
    dtypes: float64(2), int64(2), object(1)
    memory usage: 1.5+ KB
    


```python
experment_data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 37 entries, 0 to 36
    Data columns (total 5 columns):
    Date           37 non-null object
    Pageviews      37 non-null int64
    Clicks         37 non-null int64
    Enrollments    23 non-null float64
    Payments       23 non-null float64
    dtypes: float64(2), int64(2), object(1)
    memory usage: 1.5+ KB
    

Both experment and control has total of 5 columns and 37 enteries. Interms of feature distribution both experment and control data has 4 continous and 1 categorical(i.e date column) feature. Moreover, Enrollments and Payments column has only 23 non-null features values out of 37. So, we need to invetigate on this later.

Lets inspect which rows data is missed.


```python
experment_data.loc[experment_data['Enrollments'].isnull()]
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
      <th>Date</th>
      <th>Pageviews</th>
      <th>Clicks</th>
      <th>Enrollments</th>
      <th>Payments</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>23</th>
      <td>Mon, Nov 3</td>
      <td>9359</td>
      <td>789</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Tue, Nov 4</td>
      <td>9427</td>
      <td>743</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Wed, Nov 5</td>
      <td>9633</td>
      <td>808</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Thu, Nov 6</td>
      <td>9842</td>
      <td>831</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Fri, Nov 7</td>
      <td>9272</td>
      <td>767</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Sat, Nov 8</td>
      <td>8969</td>
      <td>760</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Sun, Nov 9</td>
      <td>9697</td>
      <td>850</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>30</th>
      <td>Mon, Nov 10</td>
      <td>10445</td>
      <td>851</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>31</th>
      <td>Tue, Nov 11</td>
      <td>9931</td>
      <td>831</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>32</th>
      <td>Wed, Nov 12</td>
      <td>10042</td>
      <td>802</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>33</th>
      <td>Thu, Nov 13</td>
      <td>9721</td>
      <td>829</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>34</th>
      <td>Fri, Nov 14</td>
      <td>9304</td>
      <td>770</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>35</th>
      <td>Sat, Nov 15</td>
      <td>8668</td>
      <td>724</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>36</th>
      <td>Sun, Nov 16</td>
      <td>8988</td>
      <td>710</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
control_data.loc[control_data['Enrollments'].isnull()]
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
      <th>Date</th>
      <th>Pageviews</th>
      <th>Clicks</th>
      <th>Enrollments</th>
      <th>Payments</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>23</th>
      <td>Mon, Nov 3</td>
      <td>9437</td>
      <td>788</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Tue, Nov 4</td>
      <td>9420</td>
      <td>781</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Wed, Nov 5</td>
      <td>9570</td>
      <td>805</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Thu, Nov 6</td>
      <td>9921</td>
      <td>830</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Fri, Nov 7</td>
      <td>9424</td>
      <td>781</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Sat, Nov 8</td>
      <td>9010</td>
      <td>756</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Sun, Nov 9</td>
      <td>9656</td>
      <td>825</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>30</th>
      <td>Mon, Nov 10</td>
      <td>10419</td>
      <td>874</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>31</th>
      <td>Tue, Nov 11</td>
      <td>9880</td>
      <td>830</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>32</th>
      <td>Wed, Nov 12</td>
      <td>10134</td>
      <td>801</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>33</th>
      <td>Thu, Nov 13</td>
      <td>9717</td>
      <td>814</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>34</th>
      <td>Fri, Nov 14</td>
      <td>9192</td>
      <td>735</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>35</th>
      <td>Sat, Nov 15</td>
      <td>8630</td>
      <td>743</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>36</th>
      <td>Sun, Nov 16</td>
      <td>8970</td>
      <td>722</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



For both control and experment data, target column Enrollments is missed for entries after November 3. So, thee only option we have is droping all with null values.


```python
##Merging two DataFrames
data = control_data.append(experment_data, ignore_index=True)
data.shape
```




    (74, 5)




```python
dummy=[0] * 74
data.insert(1, 'id',dummy)
data.loc[data.Enrollments.isin(experment_data.Enrollments), 'id'] = 1
data.tail(10)##because appended rows are at last position
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
      <th>Date</th>
      <th>id</th>
      <th>Pageviews</th>
      <th>Clicks</th>
      <th>Enrollments</th>
      <th>Payments</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>64</th>
      <td>Fri, Nov 7</td>
      <td>1</td>
      <td>9272</td>
      <td>767</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>65</th>
      <td>Sat, Nov 8</td>
      <td>1</td>
      <td>8969</td>
      <td>760</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>66</th>
      <td>Sun, Nov 9</td>
      <td>1</td>
      <td>9697</td>
      <td>850</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>67</th>
      <td>Mon, Nov 10</td>
      <td>1</td>
      <td>10445</td>
      <td>851</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>68</th>
      <td>Tue, Nov 11</td>
      <td>1</td>
      <td>9931</td>
      <td>831</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>69</th>
      <td>Wed, Nov 12</td>
      <td>1</td>
      <td>10042</td>
      <td>802</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>70</th>
      <td>Thu, Nov 13</td>
      <td>1</td>
      <td>9721</td>
      <td>829</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>71</th>
      <td>Fri, Nov 14</td>
      <td>1</td>
      <td>9304</td>
      <td>770</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>72</th>
      <td>Sat, Nov 15</td>
      <td>1</td>
      <td>8668</td>
      <td>724</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>73</th>
      <td>Sun, Nov 16</td>
      <td>1</td>
      <td>8988</td>
      <td>710</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
# # convert the 'Date' column to datetime format and append new column that holds weekday
data['Date'] = pd.to_datetime(data['Date'],format='%a, %b %d', errors='ignore') 
data.insert(2,'day_of_week',data['Date'].dt.weekday)
##shuffle rows using sklearn utils package to control data leakage
import sklearn
data = sklearn.utils.shuffle(data)

##add column named row_id to hold index of entries
data.insert(0,'row_id',range(1, len(data) + 1))
data.set_index('row_id')
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
      <th>Date</th>
      <th>id</th>
      <th>day_of_week</th>
      <th>Pageviews</th>
      <th>Clicks</th>
      <th>Enrollments</th>
      <th>Payments</th>
    </tr>
    <tr>
      <th>row_id</th>
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
      <th>1</th>
      <td>1900-11-08</td>
      <td>1</td>
      <td>3</td>
      <td>9010</td>
      <td>756</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1900-10-21</td>
      <td>0</td>
      <td>6</td>
      <td>10660</td>
      <td>867</td>
      <td>196.0</td>
      <td>105.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1900-11-15</td>
      <td>1</td>
      <td>3</td>
      <td>8668</td>
      <td>724</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1900-10-14</td>
      <td>1</td>
      <td>6</td>
      <td>9867</td>
      <td>827</td>
      <td>138.0</td>
      <td>92.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1900-10-17</td>
      <td>0</td>
      <td>2</td>
      <td>9008</td>
      <td>748</td>
      <td>146.0</td>
      <td>76.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1900-10-22</td>
      <td>1</td>
      <td>0</td>
      <td>9947</td>
      <td>838</td>
      <td>162.0</td>
      <td>92.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1900-11-13</td>
      <td>1</td>
      <td>1</td>
      <td>9717</td>
      <td>814</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1900-11-06</td>
      <td>1</td>
      <td>1</td>
      <td>9921</td>
      <td>830</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1900-10-12</td>
      <td>0</td>
      <td>4</td>
      <td>9102</td>
      <td>779</td>
      <td>147.0</td>
      <td>70.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1900-11-07</td>
      <td>1</td>
      <td>2</td>
      <td>9272</td>
      <td>767</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1900-10-23</td>
      <td>1</td>
      <td>1</td>
      <td>8176</td>
      <td>642</td>
      <td>122.0</td>
      <td>68.0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1900-11-08</td>
      <td>1</td>
      <td>3</td>
      <td>8969</td>
      <td>760</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1900-10-16</td>
      <td>1</td>
      <td>1</td>
      <td>9670</td>
      <td>823</td>
      <td>138.0</td>
      <td>82.0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1900-11-09</td>
      <td>1</td>
      <td>4</td>
      <td>9656</td>
      <td>825</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1900-11-03</td>
      <td>1</td>
      <td>5</td>
      <td>9359</td>
      <td>789</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1900-10-29</td>
      <td>1</td>
      <td>0</td>
      <td>9262</td>
      <td>727</td>
      <td>201.0</td>
      <td>96.0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1900-11-13</td>
      <td>1</td>
      <td>1</td>
      <td>9721</td>
      <td>829</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>18</th>
      <td>1900-10-30</td>
      <td>1</td>
      <td>1</td>
      <td>9308</td>
      <td>728</td>
      <td>207.0</td>
      <td>67.0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1900-11-07</td>
      <td>1</td>
      <td>2</td>
      <td>9424</td>
      <td>781</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20</th>
      <td>1900-11-16</td>
      <td>1</td>
      <td>4</td>
      <td>8970</td>
      <td>722</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>21</th>
      <td>1900-11-12</td>
      <td>1</td>
      <td>0</td>
      <td>10134</td>
      <td>801</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>22</th>
      <td>1900-11-03</td>
      <td>1</td>
      <td>5</td>
      <td>9437</td>
      <td>788</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>23</th>
      <td>1900-10-30</td>
      <td>0</td>
      <td>1</td>
      <td>9345</td>
      <td>734</td>
      <td>167.0</td>
      <td>75.0</td>
    </tr>
    <tr>
      <th>24</th>
      <td>1900-11-04</td>
      <td>1</td>
      <td>6</td>
      <td>9420</td>
      <td>781</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>25</th>
      <td>1900-11-01</td>
      <td>1</td>
      <td>3</td>
      <td>8448</td>
      <td>695</td>
      <td>142.0</td>
      <td>100.0</td>
    </tr>
    <tr>
      <th>26</th>
      <td>1900-10-11</td>
      <td>0</td>
      <td>3</td>
      <td>7723</td>
      <td>687</td>
      <td>134.0</td>
      <td>70.0</td>
    </tr>
    <tr>
      <th>27</th>
      <td>1900-10-19</td>
      <td>1</td>
      <td>4</td>
      <td>8434</td>
      <td>697</td>
      <td>120.0</td>
      <td>77.0</td>
    </tr>
    <tr>
      <th>28</th>
      <td>1900-11-01</td>
      <td>0</td>
      <td>3</td>
      <td>8460</td>
      <td>681</td>
      <td>156.0</td>
      <td>93.0</td>
    </tr>
    <tr>
      <th>29</th>
      <td>1900-10-28</td>
      <td>0</td>
      <td>6</td>
      <td>9363</td>
      <td>736</td>
      <td>154.0</td>
      <td>91.0</td>
    </tr>
    <tr>
      <th>30</th>
      <td>1900-10-26</td>
      <td>1</td>
      <td>4</td>
      <td>8881</td>
      <td>693</td>
      <td>153.0</td>
      <td>101.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>45</th>
      <td>1900-10-13</td>
      <td>1</td>
      <td>5</td>
      <td>10480</td>
      <td>884</td>
      <td>145.0</td>
      <td>79.0</td>
    </tr>
    <tr>
      <th>46</th>
      <td>1900-11-10</td>
      <td>1</td>
      <td>5</td>
      <td>10419</td>
      <td>874</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>47</th>
      <td>1900-11-15</td>
      <td>1</td>
      <td>3</td>
      <td>8630</td>
      <td>743</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>48</th>
      <td>1900-10-24</td>
      <td>0</td>
      <td>2</td>
      <td>9434</td>
      <td>673</td>
      <td>220.0</td>
      <td>122.0</td>
    </tr>
    <tr>
      <th>49</th>
      <td>1900-10-21</td>
      <td>1</td>
      <td>6</td>
      <td>10551</td>
      <td>864</td>
      <td>143.0</td>
      <td>71.0</td>
    </tr>
    <tr>
      <th>50</th>
      <td>1900-10-27</td>
      <td>1</td>
      <td>5</td>
      <td>9655</td>
      <td>771</td>
      <td>213.0</td>
      <td>119.0</td>
    </tr>
    <tr>
      <th>51</th>
      <td>1900-11-10</td>
      <td>1</td>
      <td>5</td>
      <td>10445</td>
      <td>851</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>52</th>
      <td>1900-11-02</td>
      <td>0</td>
      <td>4</td>
      <td>8836</td>
      <td>693</td>
      <td>206.0</td>
      <td>67.0</td>
    </tr>
    <tr>
      <th>53</th>
      <td>1900-10-20</td>
      <td>1</td>
      <td>5</td>
      <td>10496</td>
      <td>860</td>
      <td>153.0</td>
      <td>98.0</td>
    </tr>
    <tr>
      <th>54</th>
      <td>1900-10-17</td>
      <td>1</td>
      <td>2</td>
      <td>9088</td>
      <td>780</td>
      <td>127.0</td>
      <td>44.0</td>
    </tr>
    <tr>
      <th>55</th>
      <td>1900-11-05</td>
      <td>1</td>
      <td>0</td>
      <td>9633</td>
      <td>808</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>56</th>
      <td>1900-11-09</td>
      <td>1</td>
      <td>4</td>
      <td>9697</td>
      <td>850</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>57</th>
      <td>1900-10-15</td>
      <td>0</td>
      <td>0</td>
      <td>10014</td>
      <td>837</td>
      <td>163.0</td>
      <td>64.0</td>
    </tr>
    <tr>
      <th>58</th>
      <td>1900-10-20</td>
      <td>0</td>
      <td>5</td>
      <td>10667</td>
      <td>861</td>
      <td>165.0</td>
      <td>97.0</td>
    </tr>
    <tr>
      <th>59</th>
      <td>1900-11-11</td>
      <td>1</td>
      <td>6</td>
      <td>9931</td>
      <td>831</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>60</th>
      <td>1900-10-19</td>
      <td>0</td>
      <td>4</td>
      <td>8459</td>
      <td>691</td>
      <td>131.0</td>
      <td>60.0</td>
    </tr>
    <tr>
      <th>61</th>
      <td>1900-10-22</td>
      <td>1</td>
      <td>0</td>
      <td>9737</td>
      <td>801</td>
      <td>128.0</td>
      <td>70.0</td>
    </tr>
    <tr>
      <th>62</th>
      <td>1900-10-11</td>
      <td>1</td>
      <td>3</td>
      <td>7716</td>
      <td>686</td>
      <td>105.0</td>
      <td>34.0</td>
    </tr>
    <tr>
      <th>63</th>
      <td>1900-10-13</td>
      <td>0</td>
      <td>5</td>
      <td>10511</td>
      <td>909</td>
      <td>167.0</td>
      <td>95.0</td>
    </tr>
    <tr>
      <th>64</th>
      <td>1900-11-14</td>
      <td>1</td>
      <td>2</td>
      <td>9304</td>
      <td>770</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>65</th>
      <td>1900-11-16</td>
      <td>1</td>
      <td>4</td>
      <td>8988</td>
      <td>710</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>66</th>
      <td>1900-11-04</td>
      <td>1</td>
      <td>6</td>
      <td>9427</td>
      <td>743</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>67</th>
      <td>1900-10-31</td>
      <td>0</td>
      <td>2</td>
      <td>8890</td>
      <td>706</td>
      <td>174.0</td>
      <td>101.0</td>
    </tr>
    <tr>
      <th>68</th>
      <td>1900-10-23</td>
      <td>1</td>
      <td>1</td>
      <td>8324</td>
      <td>665</td>
      <td>127.0</td>
      <td>56.0</td>
    </tr>
    <tr>
      <th>69</th>
      <td>1900-11-11</td>
      <td>1</td>
      <td>6</td>
      <td>9880</td>
      <td>830</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>70</th>
      <td>1900-10-24</td>
      <td>1</td>
      <td>2</td>
      <td>9402</td>
      <td>697</td>
      <td>194.0</td>
      <td>94.0</td>
    </tr>
    <tr>
      <th>71</th>
      <td>1900-10-18</td>
      <td>0</td>
      <td>3</td>
      <td>7434</td>
      <td>632</td>
      <td>110.0</td>
      <td>70.0</td>
    </tr>
    <tr>
      <th>72</th>
      <td>1900-10-26</td>
      <td>0</td>
      <td>4</td>
      <td>8896</td>
      <td>708</td>
      <td>161.0</td>
      <td>104.0</td>
    </tr>
    <tr>
      <th>73</th>
      <td>1900-11-06</td>
      <td>1</td>
      <td>1</td>
      <td>9842</td>
      <td>831</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>74</th>
      <td>1900-10-14</td>
      <td>0</td>
      <td>6</td>
      <td>9871</td>
      <td>836</td>
      <td>156.0</td>
      <td>105.0</td>
    </tr>
  </tbody>
</table>
<p>74 rows × 7 columns</p>
</div>




```python
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
      <th>row_id</th>
      <th>Date</th>
      <th>id</th>
      <th>day_of_week</th>
      <th>Pageviews</th>
      <th>Clicks</th>
      <th>Enrollments</th>
      <th>Payments</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>28</th>
      <td>1</td>
      <td>1900-11-08</td>
      <td>1</td>
      <td>3</td>
      <td>9010</td>
      <td>756</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2</td>
      <td>1900-10-21</td>
      <td>0</td>
      <td>6</td>
      <td>10660</td>
      <td>867</td>
      <td>196.0</td>
      <td>105.0</td>
    </tr>
    <tr>
      <th>72</th>
      <td>3</td>
      <td>1900-11-15</td>
      <td>1</td>
      <td>3</td>
      <td>8668</td>
      <td>724</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>40</th>
      <td>4</td>
      <td>1900-10-14</td>
      <td>1</td>
      <td>6</td>
      <td>9867</td>
      <td>827</td>
      <td>138.0</td>
      <td>92.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>5</td>
      <td>1900-10-17</td>
      <td>0</td>
      <td>2</td>
      <td>9008</td>
      <td>748</td>
      <td>146.0</td>
      <td>76.0</td>
    </tr>
  </tbody>
</table>
</div>



As we can see from the result, interestingly all operations are successfull. day_of_week column indicateds 0 to 6 fro Monday to Sunday and id is for experment checking and row_id is used reference column.


```python
#drop Date and Payments Coloumns
drop_coloumn_list = ['Date','Payments']
data=data.drop(drop_coloumn_list, axis=1)
data.shape
```




    (74, 6)




```python
##Handle the missing data (NA) by removing these rows
data = data.dropna(how='any',axis=0) #It will delete every row (axis=0) that has "any" Null value in it.
data.shape
```




    (46, 6)



As we can see 28 rows are deleted because of missed values.


```python
data.head(10)
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
      <th>row_id</th>
      <th>id</th>
      <th>day_of_week</th>
      <th>Pageviews</th>
      <th>Clicks</th>
      <th>Enrollments</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10</th>
      <td>2</td>
      <td>0</td>
      <td>6</td>
      <td>10660</td>
      <td>867</td>
      <td>196.0</td>
    </tr>
    <tr>
      <th>40</th>
      <td>4</td>
      <td>1</td>
      <td>6</td>
      <td>9867</td>
      <td>827</td>
      <td>138.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>5</td>
      <td>0</td>
      <td>2</td>
      <td>9008</td>
      <td>748</td>
      <td>146.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>6</td>
      <td>1</td>
      <td>0</td>
      <td>9947</td>
      <td>838</td>
      <td>162.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>9</td>
      <td>0</td>
      <td>4</td>
      <td>9102</td>
      <td>779</td>
      <td>147.0</td>
    </tr>
    <tr>
      <th>49</th>
      <td>11</td>
      <td>1</td>
      <td>1</td>
      <td>8176</td>
      <td>642</td>
      <td>122.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>13</td>
      <td>1</td>
      <td>1</td>
      <td>9670</td>
      <td>823</td>
      <td>138.0</td>
    </tr>
    <tr>
      <th>55</th>
      <td>16</td>
      <td>1</td>
      <td>0</td>
      <td>9262</td>
      <td>727</td>
      <td>201.0</td>
    </tr>
    <tr>
      <th>56</th>
      <td>18</td>
      <td>1</td>
      <td>1</td>
      <td>9308</td>
      <td>728</td>
      <td>207.0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>23</td>
      <td>0</td>
      <td>1</td>
      <td>9345</td>
      <td>734</td>
      <td>167.0</td>
    </tr>
  </tbody>
</table>
</div>



# Training Model

Three algorithms are compared. <ul><li>Random Forest</li><li>Decision Tree</li><li>XGBoost</li></ul>


```python
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error
from sklearn.model_selection import cross_val_score, GridSearchCV,StratifiedKFold, KFold, RandomizedSearchCV, train_test_split
def display_scores(scores):
    print("Scores: {0}\nMean: {1:.3f}\nStd: {2:.3f}".format(scores, np.mean(scores), np.std(scores)))
    
def train_RandomForest(X_train, y_train):
    scores = []
    # Use the random grid to search for best hyperparameters
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}
    # First create the base model to tune
    rf = RandomForestRegressor()
    # Random search of parameters, using 5 fold cross validation, 
    # search across 100 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 5,
                                   verbose=1, random_state=42, n_jobs = -1)
    # Fit the random search model
    rf_random.fit(X_train, y_train)
    predictions=rf_random.predict(y_train)
    
    print("MSE:{0:.3f}\RMSE: {1:.3f}".format(mean_squared_error(y_test, predictions),
                                                            np.sqrt(mean_squared_error(y_train, predictions))))

def train_DT(X_train, y_train,x_test,y_test):
    dtr= DecisionTreeRegressor()
    dtr.fit(x_train,y_train)
    y_pred = dtr.predict(x_test)
    print(mean_squared_error(y_test, y_pred))
    


def train_XGB(X_train,X_test,y_train, y_test):
    data_dmatrix = xgb.DMatrix(data=X_train,label=y_train)
    params = {"objective":"reg:linear",'colsample_bytree': 0.3,'learning_rate': 0.1,
                'max_depth': 5, 'alpha': 10}
    cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold=5,
                    num_boost_round=100,early_stopping_rounds=10,metrics="rmse", as_pandas=True, seed=123)
    print(cv_results.head())
    print((cv_results["test-rmse-mean"]).tail(1))
    xg_reg = xgb.train(params=params, dtrain=data_dmatrix, num_boost_round=10)
    xgb.plot_importance(xg_reg)
    plt.rcParams['figure.figsize'] = [5, 5]
    plt.show()

```


```python
y=data.Enrollments.values
X=data.drop(['row_id','Enrollments'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.1, random_state=42)
```


```python
train_XGB(X_train,X_test,y_train, y_test)
```

       train-rmse-mean  train-rmse-std  test-rmse-mean  test-rmse-std
    0       143.495904        1.871520      143.279126       8.566915
    1       130.381508        1.717758      130.058894       8.708339
    2       118.602792        1.583638      118.318460       8.869647
    3       108.052387        1.467057      107.744661       9.012557
    4        98.600679        1.365147       98.665820       9.160198
    72    28.488258
    Name: test-rmse-mean, dtype: float64
    


![png](ml_30_1.png)


The information gain is 50% from Pageviews and Clicks combined. Experiment has no significan contribution to information gain, indicating it’s still predictive (just not nearly as much as Pageviews). This tells a story that if Enrollments are critical, Udacity should focus on getting clicks and Pageviews.

To generalize the result even if further investigation is required for other models also, If Udacity wants to maximimize enrollments, it should focus on getting clicks. Click is the most important feature in our model.


```python
#train_DT(X_train, y_train,x_test,y_test) 
#train_RandomForest(X,y)
```
Further investigation can be continued, but for now I have to stop because of deadline. Hope I will come up with further investigation.