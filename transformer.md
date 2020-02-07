
### Transformer (BERT, ROBERTa, Transformer-Xl, DistilBERT, XLNet, XLM) for Text Classification
The Transformer was proposed in the paper [Attention is All You Need](https://arxiv.org/abs/1706.03762). A TensorFlow implementation of it is available as a part of the [Tensor2Tensor](https://github.com/tensorflow/tensor2tensor) package. The motivation behind transformer is to deal with practical problem of the popular sequence-to-sequence(Seq2Seq) models RNN and its flavors such as GRU and LSTM. Despite boosted contribution of seq2seq models, there are certain limitations:
- #### Dealing with long-range dependencies: 
Given input sequence, RNNs need to encode the information from the entire sequence in one single context vector and pass the last hidden step to decoder module. The decoder is supposed to generate a translation solely based on the last hidden state from the encoder. It can generalize representation of text meaning, but it fails to detect all sequence when input gets longer as RNN seems to “forget” the previous words. Forexample, 
- #### Unable to parallelize:
While CNN can be parallelized, sequential nature of RNN made it difficult to use today's GPU power. Transformer combined the idea of CNN with attention techniques and included positional encoding to keep sequence order. Encoding the position of each input in the sequence is relevant, since the position of inputs matters for some of NLP tasks such as translation, sequence prediction, etc.  

In addition to the [original paper](https://arxiv.org/abs/1706.03762), I recommend to read https://www.analyticsvidhya.com/blog/2019/06/understanding-transformers-nlp-state-of-the-art-models/

### Objective of this notebook
- classfying text as hate-speech, offensive language, or neutral
- Exploring transformer architecture for text classification

I will use the dataset used by: Thomas Davidson, Dana Warmsley, Michael Macy, and Ingmar Weber. 2017. "Automated Hate Speech Detection and the Problem of Offensive Language.",*Proceedings of the 11th International AAAI Conference on Web and Social Media* (https://aaai.org/ocs/index.php/ICWSM/ICWSM17/paper/view/15665)

### STEPS
- Reading data
- Defining transformer
    - Embedding the inputs: feeding network vector for each words
    - Positional Encodings: tells the network about the word’s position
    - Creating Masks: zeroing padded input attention. (For tasks like translation, its also helps at  decoder to control where to stop predicting next word)
    - The Multi-Head Attention layer:
    - The Feed-Forward prediction layer:
- Fitting data to the model


```python
import pandas as pd
import numpy as np
import pickle
```


```python
tweet = pd.read_csv("../data/labeled_data.csv")
```


```python
tweet.head()
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
      <th>Unnamed: 0</th>
      <th>count</th>
      <th>hate_speech</th>
      <th>offensive_language</th>
      <th>neither</th>
      <th>class</th>
      <th>tweet</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>2</td>
      <td>!!! RT @mayasolovely: As a woman you shouldn't...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>!!!!! RT @mleew17: boy dats cold...tyga dwn ba...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>3</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>!!!!!!! RT @UrKindOfBrand Dawg!!!! RT @80sbaby...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>!!!!!!!!! RT @C_G_Anderson: @viva_based she lo...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>6</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>!!!!!!!!!!!!! RT @ShenikaRoberts: The shit you...</td>
    </tr>
  </tbody>
</table>
</div>



The data are stored as a CSV and as a pickled pandas dataframe. Each data file contains 6 columns:
- `count` = number of CrowdFlower users who coded each tweet.

- `hate_speech` = number of CF users who judged the tweet to be hate speech.

- `offensive_language` = number of CF users who judged the tweet to be offensive.

- `neither` = number of CF users who judged the tweet to be neither offensive nor non-offensive.

- `class` = class label for majority of CF users.
  0 - hate speech
  1 - offensive  language
  2 - neither
- `tweet` = tweets

Since, the aim of the project is classifying hate speech I will use only the last two columns (`class` and `tweet`)


```python
text=tweet['tweet']
label=tweet['class']
print(text.shape,label.shape)
```

    (24783,) (24783,)
    

Defining transformer basically requires defining four basic steps mentioned above: embedding, positional encoding, masking, prediction layer (usually feed forward NN). However, instead of creating the architecture from the scratch we can use it from:
* [huggingface- pytorch implementation](https://github.com/huggingface/transformers)
* [simplerepresentations- wrapper library based on huggingface](https://pypi.org/project/simpletransformers/)
* [tensorflow implementation](https://www.tensorflow.org/tutorials/text/transformer#setup_input_pipeline)

I will use create a class that use simplerepresentations wrapper library to use transformers.


```python
import logging
logging.basicConfig(level=logging.INFO)
from tensorflow.python.keras.utils.data_utils import Sequence
from simplerepresentations import RepresentationModel

class TransformerModel(Sequence):       
    def __init__(self, representation_model, tweet, labels, batch_size, token_level=True):
        self.representation_model = representation_model
        self.tweet = tweet
        self.labels = labels
        self.batch_size = batch_size
        self.token_level = token_level

    def __len__(self):
        return int(np.ceil(len(self.tweet) / float(self.batch_size)))
    
    def tweet_generator(self,idx, tweet):
        tweet_batch = np.array(answer[idx * self.batch_size:(idx + 1) * self.batch_size])
        
        tweet_sen_batch, tweet_tok_batch = self.representation_model(tweet_batch)

        if self.token_level:
            tweet_batch = tweet_tok_batch
        else:
            tweet_batch = tweet_sen_batch
            
        return tweet_batch
    
    def __getitem__(self, idx):
        tweet_generator=self.tweet_generator(idx,self.tweet[0])
      
        labels_batch = np.array(self.labels[idx * self.batch_size:(idx + 1) * self.batch_size])
        
        return tweet_generator, np.array(labels_batch)

representation_model = RepresentationModel(
    model_type=['bert', 'xlnet', 'xlm', 'roberta', 'distilbert'],
    model_name=model_name,
    batch_size=64,
    max_seq_length=60, # truncate sentences to be less than or equal to 128 tokens
    combination_method='cat', # sum the last `last_hidden_to_use` hidden states
    last_hidden_to_use=1, # use the last 1 hidden states to build tokens representations
    verbose=0
)
train_generator= DataGenerator(representation_model, text, label, 64)

```

    INFO:transformers.file_utils:PyTorch version 1.1.0 available.
    INFO:transformers.file_utils:TensorFlow version 2.0.0-alpha0 available.
    
