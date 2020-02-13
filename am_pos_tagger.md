

```python
import os
import numpy as np
```


```python
# from __future__ import absolute_import, division, print_function, unicode_literals
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
import tensorflow as tf
import os
import numpy as np
def find_between( s, first, last ):
    try:
        start = s.index( first ) + len( first )
        end = s.index( last, start )
        return s[start:end]
    except ValueError:
        return ""

sentences=[]
tags=[]

def prepare_tags_and_sentence(sentence):
    seq_tag=[]
    seq_word=[]
    with_tags=sentence.split()
    for wt in with_tags:
        if wt.startswith('<'):
            seq_tag.append(wt)
        else:
            seq_word.append(wt)
    sentences.append(np.array(seq_word))
    tags.append(np.array(seq_tag))

for line in open('WIC-Tagged-UTF8-20080530.xml',encoding='utf-8'):
    content=line.rstrip("\n\r")
    if content.startswith('<')==False:
        if content:
            sen_tag=content.strip().split("። <PUNC>")
            for st in sen_tag:
                if st:
                    prepare_tags_and_sentence(st)
    elif content.startswith('<title>'):
        with_tags=find_between(content,'<title>','</title>')
        prepare_tags_and_sentence(with_tags)

print(len(sentences),len(tags))

from sklearn.model_selection import train_test_split

(train_sentences, 
 test_sentences, 
 train_tags, 
 test_tags) = train_test_split(sentences, tags, test_size=0.2)

words, tags = set([]), set([])
 
for s in train_sentences:
    for w in s:
        words.add(w)
        
for ts in train_tags:
    for t in ts:
        tags.add(t)

word2index = {w: i + 2 for i, w in enumerate(list(words))}
word2index['-PAD-'] = 0  # The special value used for padding
word2index['-OOV-'] = 1  # The special value used for OOVs
 
tag2index = {t: i + 1 for i, t in enumerate(list(tags))}
tag2index['-PAD-'] = 0  # The special value used to padding

train_sentences_X, test_sentences_X, train_tags_y, test_tags_y = [], [], [], []

for s in train_sentences:
    s_int = []
    for w in s:
        try:
            s_int.append(word2index[w])
        except KeyError:
            s_int.append(word2index['-OOV-'])

    train_sentences_X.append(s_int)

for s in test_sentences:
    s_int = []
    for w in s:
        try:
            s_int.append(word2index[w])
        except KeyError:
            s_int.append(word2index['-OOV-'])

    test_sentences_X.append(s_int)

for s in train_tags:
    train_tags_y.append([tag2index[t] for t in s])
for s in test_tags:
   
    test_tags_y.append([tag2index[t] for t in s])

MAX_LENGTH = len(max(train_sentences_X, key=len))

from tensorflow.keras.preprocessing.sequence import pad_sequences
 
train_sentences_X = pad_sequences(train_sentences_X, maxlen=MAX_LENGTH, padding='post')
test_sentences_X = pad_sequences(test_sentences_X, maxlen=MAX_LENGTH, padding='post')
train_tags_y = pad_sequences(train_tags_y, maxlen=MAX_LENGTH, padding='post')
test_tags_y = pad_sequences(test_tags_y, maxlen=MAX_LENGTH, padding='post')

from keras import backend as K
 
def ignore_class_accuracy(to_ignore=0):
    def ignore_accuracy(y_true, y_pred):
        y_true_class = K.argmax(y_true, axis=-1)
        y_pred_class = K.argmax(y_pred, axis=-1)
 
        ignore_mask = K.cast(K.not_equal(y_pred_class, to_ignore), 'int32')
        matches = K.cast(K.equal(y_true_class, y_pred_class), 'int32') * ignore_mask
        accuracy = K.sum(matches) / K.maximum(K.sum(ignore_mask), 1)
        return accuracy
    return ignore_accuracy

from gensim.models import KeyedVectors
am_model= KeyedVectors.load('Model/am_w2v_cbow_300D')
WORD_EMBED_SIZE=300
E = np.zeros((len(word2index), WORD_EMBED_SIZE))
for word,index in word2index.items():
    try:
        E[index]=am_model[word]             
    except KeyError:
        pass
E[1] = np.random.random(WORD_EMBED_SIZE)
print('Embedding matrix shape', E.shape)

from tensorflow.keras.layers import Dense, GRU, InputLayer, Bidirectional, TimeDistributed, Embedding, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential 
from tensorflow.keras.callbacks import ModelCheckpoint
model = Sequential()
model.add(InputLayer(input_shape=(MAX_LENGTH, )))
model.add(Embedding(len(word2index), WORD_EMBED_SIZE, weights=[E]))
model.add(Bidirectional(GRU(256, return_sequences=True)))
model.add(TimeDistributed(Dense(len(tag2index))))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(0.001),
              metrics=['accuracy', ignore_class_accuracy(0)])
 
model.summary()

def to_categorical(sequences, categories):
    cat_sequences = []
    for s in sequences:
        cats = []
        for item in s:
            cats.append(np.zeros(categories))
            cats[-1][item] = 1.0
        cat_sequences.append(cats)
    return np.array(cat_sequences)
# cat_train_tags_y = to_categorical(train_tags_y, len(tag2index))
checkpoint_path = "Model/Am-POS/cbow_w2v_300D/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
model.fit(train_sentences_X, to_categorical(train_tags_y,
 len(tag2index)), batch_size=64, epochs=20, validation_split=0.1,callbacks=[cp_callback])

def logits_to_tokens(sequences, index):
    token_sequences = []
    for categorical_sequence in sequences:
        token_sequence = []
        for categorical in categorical_sequence:
            token_sequence.append(index[np.argmax(categorical)])
 
        token_sequences.append(token_sequence)
 
    return token_sequences

scores = model.evaluate(test_sentences_X, to_categorical(test_tags_y, len(tag2index)))
print(f"{model.metrics_names[1]}: {scores[1] * 100}")   # 

sample=['በቻይና በመቶዎች የሚቆጠሩ የሕክምና ዶክተሮች በኖቭል ኮሮና ቫይረስ መጠቃታቸው ተነገረ ።','ሁለት ታዳጊዎችን አግቶ ገንዘብ የጠየቀው ግለሰብ የ25 ዓመት እስር ተፈረደበት ።']
sample=[s.split() for s in sample]
test_samples_X = []
for s in sample:
    s_int = []
    for w in s:
        try:
            s_int.append(word2index[w])
        except KeyError:
            s_int.append(word2index['-OOV-'])
    test_samples_X.append(s_int)

test_samples_X = pad_sequences(test_samples_X, maxlen=MAX_LENGTH, padding='post')

predictions = model.predict(test_samples_X)
print(logits_to_tokens(predictions, {i: t for t, i in tag2index.items()}))

```

    8076 8076
    

    Using TensorFlow backend.
    

    Couldn't import dot_parser, loading of dot files will not be possible.
    

    C:\Anaconda3\lib\site-packages\gensim\utils.py:1209: UserWarning: detected Windows; aliasing chunkize to chunkize_serial
      warnings.warn("detected Windows; aliasing chunkize to chunkize_serial")
    C:\Anaconda3\lib\site-packages\ipykernel_launcher.py:124: DeprecationWarning: Call to deprecated `__getitem__` (Method will be removed in 4.0.0, use self.wv.__getitem__() instead).
    

    Embedding matrix shape (28553, 300)
    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding (Embedding)        (None, 214, 300)          8565900   
    _________________________________________________________________
    bidirectional (Bidirectional (None, 214, 512)          857088    
    _________________________________________________________________
    time_distributed (TimeDistri (None, 214, 36)           18468     
    _________________________________________________________________
    activation (Activation)      (None, 214, 36)           0         
    =================================================================
    Total params: 9,441,456
    Trainable params: 9,441,456
    Non-trainable params: 0
    _________________________________________________________________
    

    WARNING: Logging before flag parsing goes to stderr.
    W0213 15:09:28.579929 27780 deprecation.py:323] From C:\Anaconda3\lib\site-packages\tensorflow\python\ops\math_grad.py:1250: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use tf.where in 2.0, which has the same broadcast rule as np.where
    

    Train on 5814 samples, validate on 646 samples
    Epoch 1/20
    5760/5814 [============================>.] - ETA: 3s - loss: 0.5866 - accuracy: 0.9170 - ignore_accuracy: 0.3591
    Epoch 00001: saving model to Model/Am-POS/cbow_w2v_300D/cp.ckpt
    5814/5814 [==============================] - 454s 78ms/sample - loss: 0.5833 - accuracy: 0.9171 - ignore_accuracy: 0.3593 - val_loss: 0.2606 - val_accuracy: 0.9255 - val_ignore_accuracy: 0.3762
    Epoch 2/20
    5760/5814 [============================>.] - ETA: 4s - loss: 0.2232 - accuracy: 0.9335 - ignore_accuracy: 0.4224 
    Epoch 00002: saving model to Model/Am-POS/cbow_w2v_300D/cp.ckpt
    5814/5814 [==============================] - 532s 91ms/sample - loss: 0.2230 - accuracy: 0.9335 - ignore_accuracy: 0.4233 - val_loss: 0.2112 - val_accuracy: 0.9376 - val_ignore_accuracy: 0.4790
    Epoch 3/20
    5760/5814 [============================>.] - ETA: 5s - loss: 0.1719 - accuracy: 0.9512 - ignore_accuracy: 0.5774 
    Epoch 00003: saving model to Model/Am-POS/cbow_w2v_300D/cp.ckpt
    5814/5814 [==============================] - 574s 99ms/sample - loss: 0.1720 - accuracy: 0.9512 - ignore_accuracy: 0.5771 - val_loss: 0.1810 - val_accuracy: 0.9495 - val_ignore_accuracy: 0.5807
    Epoch 4/20
    5760/5814 [============================>.] - ETA: 5s - loss: 0.1390 - accuracy: 0.9610 - ignore_accuracy: 0.6631 
    Epoch 00004: saving model to Model/Am-POS/cbow_w2v_300D/cp.ckpt
    5814/5814 [==============================] - 623s 107ms/sample - loss: 0.1390 - accuracy: 0.9610 - ignore_accuracy: 0.6630 - val_loss: 0.1695 - val_accuracy: 0.9516 - val_ignore_accuracy: 0.5938
    Epoch 5/20
    5760/5814 [============================>.] - ETA: 6s - loss: 0.1183 - accuracy: 0.9657 - ignore_accuracy: 0.7037 
    Epoch 00005: saving model to Model/Am-POS/cbow_w2v_300D/cp.ckpt
    5814/5814 [==============================] - 697s 120ms/sample - loss: 0.1184 - accuracy: 0.9657 - ignore_accuracy: 0.7032 - val_loss: 0.1626 - val_accuracy: 0.9526 - val_ignore_accuracy: 0.6057
    Epoch 6/20
    5760/5814 [============================>.] - ETA: 6s - loss: 0.1028 - accuracy: 0.9684 - ignore_accuracy: 0.7269 
    Epoch 00006: saving model to Model/Am-POS/cbow_w2v_300D/cp.ckpt
    5814/5814 [==============================] - 777s 134ms/sample - loss: 0.1028 - accuracy: 0.9684 - ignore_accuracy: 0.7270 - val_loss: 0.1585 - val_accuracy: 0.9537 - val_ignore_accuracy: 0.6203
    Epoch 7/20
    5760/5814 [============================>.] - ETA: 7s - loss: 0.0896 - accuracy: 0.9711 - ignore_accuracy: 0.7491 
    Epoch 00007: saving model to Model/Am-POS/cbow_w2v_300D/cp.ckpt
    5814/5814 [==============================] - 787s 135ms/sample - loss: 0.0896 - accuracy: 0.9711 - ignore_accuracy: 0.7492 - val_loss: 0.1567 - val_accuracy: 0.9544 - val_ignore_accuracy: 0.6179
    Epoch 8/20
    5760/5814 [============================>.] - ETA: 7s - loss: 0.0782 - accuracy: 0.9742 - ignore_accuracy: 0.7756 
    Epoch 00008: saving model to Model/Am-POS/cbow_w2v_300D/cp.ckpt
    5814/5814 [==============================] - 799s 137ms/sample - loss: 0.0783 - accuracy: 0.9742 - ignore_accuracy: 0.7757 - val_loss: 0.1603 - val_accuracy: 0.9544 - val_ignore_accuracy: 0.6299
    Epoch 9/20
    5760/5814 [============================>.] - ETA: 7s - loss: 0.0689 - accuracy: 0.9773 - ignore_accuracy: 0.8029 
    Epoch 00009: saving model to Model/Am-POS/cbow_w2v_300D/cp.ckpt
    5814/5814 [==============================] - 831s 143ms/sample - loss: 0.0687 - accuracy: 0.9774 - ignore_accuracy: 0.8036 - val_loss: 0.1638 - val_accuracy: 0.9546 - val_ignore_accuracy: 0.6181
    Epoch 10/20
    5760/5814 [============================>.] - ETA: 7s - loss: 0.0601 - accuracy: 0.9807 - ignore_accuracy: 0.8317 
    Epoch 00010: saving model to Model/Am-POS/cbow_w2v_300D/cp.ckpt
    5814/5814 [==============================] - 821s 141ms/sample - loss: 0.0601 - accuracy: 0.9807 - ignore_accuracy: 0.8318 - val_loss: 0.1698 - val_accuracy: 0.9543 - val_ignore_accuracy: 0.6014
    Epoch 11/20
    5760/5814 [============================>.] - ETA: 7s - loss: 0.0521 - accuracy: 0.9835 - ignore_accuracy: 0.8562 
    Epoch 00011: saving model to Model/Am-POS/cbow_w2v_300D/cp.ckpt
    5814/5814 [==============================] - 799s 137ms/sample - loss: 0.0522 - accuracy: 0.9835 - ignore_accuracy: 0.8560 - val_loss: 0.1761 - val_accuracy: 0.9543 - val_ignore_accuracy: 0.6220
    Epoch 12/20
     768/5814 [==>...........................] - ETA: 12:47 - loss: 0.0397 - accuracy: 0.9884 - ignore_accuracy: 0.8960


```python
def find_between( s, first, last ):
    try:
        start = s.index( first ) + len( first )
        end = s.index( last, start )
        return s[start:end]
    except ValueError:
        return ""

sentences=[]
tags=[]

def prepare_tags_and_sentence(sentence):
    seq_tag=[]
    seq_word=[]
    with_tags=sentence.split()
    for wt in with_tags:
        if wt.startswith('<'):
            seq_tag.append(wt)
        else:
            seq_word.append(wt)
    sentences.append(np.array(seq_word))
    tags.append(np.array(seq_tag))

for line in open('WIC-Tagged-UTF8-20080530.xml',encoding='utf-8'):
    content=line.rstrip("\n\r")
    if content.startswith('<')==False:
        if content:
            sen_tag=content.strip().split("። <PUNC>")
            for st in sen_tag:
                if st:
                    prepare_tags_and_sentence(st)
    elif content.startswith('<title>'):
        with_tags=find_between(content,'<title>','</title>')
        prepare_tags_and_sentence(with_tags)

print(len(sentences),len(tags))
```

    8076 8076
    


```python
print(sentences[254])
print(tags[254])
```

    ['የዋልታ' 'እንፎርሜሽን' 'ማእከል' 'ሪፖርተር' 'ጋዜጠኛ' 'አበባው' 'ዘውዴ' 'በበኩሉ' 'የመገናኛ' 'ብዙሃን'
     'የሚያጋልጧቸውን' 'አስተዳደራዊ' 'በደሎችና' 'ሙስናዊ' 'ኤ' 'አሰራሮች' 'የሚመለከታቸው' 'የመንግስት'
     'አካላት' 'ተከታትለው' 'እርምጃ' 'ካለመውሰዳቸው' 'የተነሳ' 'ሙሰኞች' 'ጋዜጠኞችን' '"' 'አውርታችሁ'
     'ምን' 'ታመጣላችሁ' '!' 'እንደፈለጋችሁ' 'አውሪ' '"' 'እስከ' 'ማለት' 'መድረሳቸውን' 'አመልክቷል']
    ['<NP>' '<N>' '<N>' '<N>' '<N>' '<N>' '<N>' '<NP>' '<NP>' '<N>' '<VREL>'
     '<N>' '<N>' '<ADJ>' '<N>' '<VREL>' '<NP>' '<N>' '<V>' '<N>' '<NP>'
     '<VREL>' '<N>' '<N>' '<PUNC>' '<N>' '<PRON>' '<V>' '<PUNC>' '<VP>' '<V>'
     '<PUNC>' '<PREP>' '<VN>' '<VN>' '<V>']
    


```python
from sklearn.model_selection import train_test_split

(train_sentences, 
 test_sentences, 
 train_tags, 
 test_tags) = train_test_split(sentences, tags, test_size=0.2)
```


```python
words, tags = set([]), set([])
 
for s in train_sentences:
    for w in s:
        words.add(w)
        
for ts in train_tags:
    for t in ts:
        tags.add(t)

word2index = {w: i + 2 for i, w in enumerate(list(words))}
word2index['-PAD-'] = 0  # The special value used for padding
word2index['-OOV-'] = 1  # The special value used for OOVs

tag2index = {t: i + 1 for i, t in enumerate(list(tags))}
tag2index['-PAD-'] = 0  # The special value used to padding
```


```python
train_sentences_X, test_sentences_X, train_tags_y, test_tags_y = [], [], [], []

for s in train_sentences:
    s_int = []
    for w in s:
        try:
            s_int.append(word2index[w])
        except KeyError:
            s_int.append(word2index['-OOV-'])

    train_sentences_X.append(s_int)

for s in test_sentences:
    s_int = []
    for w in s:
        try:
            s_int.append(word2index[w])
        except KeyError:
            s_int.append(word2index['-OOV-'])

    test_sentences_X.append(s_int)

for s in train_tags:
    train_tags_y.append([tag2index[t] for t in s])
for s in test_tags:
   
    test_tags_y.append([tag2index[t] for t in s])
```


```python
MAX_LENGTH = len(max(train_sentences_X, key=len))
```

    214
    


```python
from tensorflow.keras.preprocessing.sequence import pad_sequences
 
train_sentences_X = pad_sequences(train_sentences_X, maxlen=MAX_LENGTH, padding='post')
test_sentences_X = pad_sequences(test_sentences_X, maxlen=MAX_LENGTH, padding='post')
train_tags_y = pad_sequences(train_tags_y, maxlen=MAX_LENGTH, padding='post')
test_tags_y = pad_sequences(test_tags_y, maxlen=MAX_LENGTH, padding='post')
```


```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, InputLayer, Bidirectional, TimeDistributed, Embedding, Activation
from tensorflow.keras.optimizers import Adam
 

model = Sequential()
model.add(InputLayer(input_shape=(MAX_LENGTH, )))
model.add(Embedding(len(word2index), 128))
model.add(Bidirectional(LSTM(256, return_sequences=True)))
model.add(TimeDistributed(Dense(len(tag2index))))
model.add(Activation('softmax'))
 
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(0.001),
              metrics=['accuracy'])
 
model.summary()
```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding (Embedding)        (None, 214, 128)          3671424   
    _________________________________________________________________
    bidirectional (Bidirectional (None, 214, 512)          788480    
    _________________________________________________________________
    time_distributed (TimeDistri (None, 214, 36)           18468     
    _________________________________________________________________
    activation (Activation)      (None, 214, 36)           0         
    =================================================================
    Total params: 4,478,372
    Trainable params: 4,478,372
    Non-trainable params: 0
    _________________________________________________________________
    


```python
from gensim.models import KeyedVectors
am_model= KeyedVectors.load('Model/am_fasttext_cbow_200D')
WORD_EMBED_SIZE=200
E = np.zeros((len(word2index), WORD_EMBED_SIZE))
for word,index in word2index.items():
    try:
        E[index]=am_model[word]             
    except KeyError:
        pass
E[1] = np.random.random(WORD_EMBED_SIZE)
print('Embedding matrix shape', E.shape)
```

    C:\Anaconda3\lib\site-packages\gensim\utils.py:1209: UserWarning: detected Windows; aliasing chunkize to chunkize_serial
      warnings.warn("detected Windows; aliasing chunkize to chunkize_serial")
    C:\Anaconda3\lib\site-packages\ipykernel_launcher.py:7: DeprecationWarning: Call to deprecated `__getitem__` (Method will be removed in 4.0.0, use self.wv.__getitem__() instead).
      import sys
    

    Embedding matrix shape (28683, 200)
    


```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, InputLayer, Bidirectional, TimeDistributed, Embedding, Activation
from tensorflow.keras.optimizers import Adam
 

model = Sequential()
model.add(InputLayer(input_shape=(MAX_LENGTH, )))
model.add(Embedding(len(word2index), WORD_EMBED_SIZE, weights=[E]))
model.add(Bidirectional(LSTM(256, return_sequences=True)))
model.add(TimeDistributed(Dense(len(tag2index))))
model.add(Activation('softmax'))
 
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(0.001),
              metrics=['accuracy'])
 
model.summary()
```


```python
def to_categorical(sequences, categories):
    cat_sequences = []
    for s in sequences:
        cats = []
        for item in s:
            cats.append(np.zeros(categories))
            cats[-1][item] = 1.0
        cat_sequences.append(cats)
    return np.array(cat_sequences)
```


```python
cat_train_tags_y = to_categorical(train_tags_y, len(tag2index))
print(cat_train_tags_y[0])
```


```python
model.fit(train_sentences_X, to_categorical(train_tags_y, len(tag2index)), batch_size=128, epochs=40, validation_split=0.2)
```

    WARNING: Logging before flag parsing goes to stderr.
    W0213 00:08:54.303955 15160 deprecation.py:323] From C:\Anaconda3\lib\site-packages\tensorflow\python\ops\math_grad.py:1250: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use tf.where in 2.0, which has the same broadcast rule as np.where
    

    Train on 5168 samples, validate on 1292 samples
    Epoch 1/40
    5168/5168 [==============================] - 198s 38ms/sample - loss: 0.7227 - accuracy: 0.8829 - val_loss: 0.2955 - val_accuracy: 0.9232
    Epoch 2/40
    5168/5168 [==============================] - 284s 55ms/sample - loss: 0.2820 - accuracy: 0.9246 - val_loss: 0.2756 - val_accuracy: 0.9250
    Epoch 3/40
    5168/5168 [==============================] - 316s 61ms/sample - loss: 0.2639 - accuracy: 0.9263 - val_loss: 0.2639 - val_accuracy: 0.9255
    Epoch 4/40
    5168/5168 [==============================] - 400s 77ms/sample - loss: 0.2532 - accuracy: 0.9281 - val_loss: 0.2618 - val_accuracy: 0.9259
    Epoch 5/40
    5168/5168 [==============================] - 557s 108ms/sample - loss: 0.2487 - accuracy: 0.9291 - val_loss: 0.2570 - val_accuracy: 0.9263
    Epoch 6/40
    5168/5168 [==============================] - 352s 68ms/sample - loss: 0.2442 - accuracy: 0.9302 - val_loss: 0.2580 - val_accuracy: 0.9263
    Epoch 7/40
    5168/5168 [==============================] - 724s 140ms/sample - loss: 0.2395 - accuracy: 0.9312 - val_loss: 0.2507 - val_accuracy: 0.9268
    Epoch 8/40
    5168/5168 [==============================] - 375s 73ms/sample - loss: 0.2315 - accuracy: 0.9317 - val_loss: 0.2410 - val_accuracy: 0.9284
    Epoch 9/40
    5168/5168 [==============================] - 371s 72ms/sample - loss: 0.2186 - accuracy: 0.9371 - val_loss: 0.2333 - val_accuracy: 0.9345
    Epoch 10/40
    5168/5168 [==============================] - 376s 73ms/sample - loss: 0.2047 - accuracy: 0.9438 - val_loss: 0.2249 - val_accuracy: 0.9369
    Epoch 11/40
    5168/5168 [==============================] - 365s 71ms/sample - loss: 0.1928 - accuracy: 0.9465 - val_loss: 0.2184 - val_accuracy: 0.9384
    Epoch 12/40
    5168/5168 [==============================] - 377s 73ms/sample - loss: 0.1801 - accuracy: 0.9496 - val_loss: 0.2127 - val_accuracy: 0.9406
    Epoch 13/40
    5168/5168 [==============================] - 371s 72ms/sample - loss: 0.1667 - accuracy: 0.9534 - val_loss: 0.2047 - val_accuracy: 0.9429
    Epoch 14/40
    5168/5168 [==============================] - 605s 117ms/sample - loss: 0.1546 - accuracy: 0.9566 - val_loss: 0.1999 - val_accuracy: 0.9442
    Epoch 15/40
    5168/5168 [==============================] - 351s 68ms/sample - loss: 0.1443 - accuracy: 0.9589 - val_loss: 0.2018 - val_accuracy: 0.9451
    Epoch 16/40
    5168/5168 [==============================] - 354s 68ms/sample - loss: 0.1355 - accuracy: 0.9612 - val_loss: 0.1940 - val_accuracy: 0.9465
    Epoch 17/40
    5168/5168 [==============================] - 355s 69ms/sample - loss: 0.1272 - accuracy: 0.9636 - val_loss: 0.1929 - val_accuracy: 0.9474
    Epoch 18/40
    5168/5168 [==============================] - 355s 69ms/sample - loss: 0.1199 - accuracy: 0.9657 - val_loss: 0.1906 - val_accuracy: 0.9484
    Epoch 19/40
    5168/5168 [==============================] - 370s 72ms/sample - loss: 0.1130 - accuracy: 0.9675 - val_loss: 0.1908 - val_accuracy: 0.9488
    Epoch 20/40
    5168/5168 [==============================] - 341s 66ms/sample - loss: 0.1070 - accuracy: 0.9690 - val_loss: 0.1879 - val_accuracy: 0.9495
    Epoch 21/40
    5168/5168 [==============================] - 349s 67ms/sample - loss: 0.1007 - accuracy: 0.9704 - val_loss: 0.1877 - val_accuracy: 0.9497
    Epoch 22/40
    5168/5168 [==============================] - 345s 67ms/sample - loss: 0.0943 - accuracy: 0.9721 - val_loss: 0.1871 - val_accuracy: 0.9503
    Epoch 23/40
    5168/5168 [==============================] - 345s 67ms/sample - loss: 0.0876 - accuracy: 0.9737 - val_loss: 0.1863 - val_accuracy: 0.9506
    Epoch 24/40
    5168/5168 [==============================] - 347s 67ms/sample - loss: 0.0813 - accuracy: 0.9756 - val_loss: 0.1872 - val_accuracy: 0.9506
    Epoch 25/40
    5168/5168 [==============================] - 343s 66ms/sample - loss: 0.0750 - accuracy: 0.9776 - val_loss: 0.1866 - val_accuracy: 0.9506
    Epoch 26/40
    5168/5168 [==============================] - 351s 68ms/sample - loss: 0.0688 - accuracy: 0.9795 - val_loss: 0.1880 - val_accuracy: 0.9510
    Epoch 27/40
    5168/5168 [==============================] - 340s 66ms/sample - loss: 0.0633 - accuracy: 0.9813 - val_loss: 0.1920 - val_accuracy: 0.9505
    Epoch 28/40
    5168/5168 [==============================] - 352s 68ms/sample - loss: 0.0581 - accuracy: 0.9830 - val_loss: 0.1950 - val_accuracy: 0.9500
    Epoch 29/40
    5168/5168 [==============================] - 344s 67ms/sample - loss: 0.0536 - accuracy: 0.9845 - val_loss: 0.2002 - val_accuracy: 0.9506
    Epoch 30/40
    5168/5168 [==============================] - 344s 67ms/sample - loss: 0.0496 - accuracy: 0.9857 - val_loss: 0.2048 - val_accuracy: 0.9505
    Epoch 31/40
    5168/5168 [==============================] - 385s 74ms/sample - loss: 0.0460 - accuracy: 0.9869 - val_loss: 0.2100 - val_accuracy: 0.9507
    Epoch 32/40
    5168/5168 [==============================] - 332s 64ms/sample - loss: 0.0427 - accuracy: 0.9879 - val_loss: 0.2132 - val_accuracy: 0.9499
    Epoch 33/40
    5168/5168 [==============================] - 324s 63ms/sample - loss: 0.0393 - accuracy: 0.9891 - val_loss: 0.2191 - val_accuracy: 0.9503
    Epoch 34/40
    5168/5168 [==============================] - 335s 65ms/sample - loss: 0.0361 - accuracy: 0.9901 - val_loss: 0.2233 - val_accuracy: 0.9490
    Epoch 35/40
    5168/5168 [==============================] - 17161s 3s/sample - loss: 0.0333 - accuracy: 0.9910 - val_loss: 0.2293 - val_accuracy: 0.9498
    Epoch 36/40
    5168/5168 [==============================] - 359s 70ms/sample - loss: 0.0309 - accuracy: 0.9918 - val_loss: 0.2351 - val_accuracy: 0.9497
    Epoch 37/40
    5168/5168 [==============================] - 394s 76ms/sample - loss: 0.0286 - accuracy: 0.9924 - val_loss: 0.2430 - val_accuracy: 0.9503
    Epoch 38/40
    5168/5168 [==============================] - 398s 77ms/sample - loss: 0.0258 - accuracy: 0.9933 - val_loss: 0.2479 - val_accuracy: 0.9493
    Epoch 39/40
    5168/5168 [==============================] - 370s 72ms/sample - loss: 0.0239 - accuracy: 0.9940 - val_loss: 0.2552 - val_accuracy: 0.9470
    Epoch 40/40
    5168/5168 [==============================] - 386s 75ms/sample - loss: 0.0387 - accuracy: 0.9891 - val_loss: 0.2321 - val_accuracy: 0.9475
    




    <tensorflow.python.keras.callbacks.History at 0x26d02729048>




```python
scores = model.evaluate(test_sentences_X, to_categorical(test_tags_y, len(tag2index)))
print(f"{model.metrics_names[1]}: {scores[1] * 100}")   # 
```

    1616/1616 [==============================] - 29s 18ms/sample - loss: 0.2548 - accuracy: 0.9427
    accuracy: 94.27223205566406
    


```python
from keras import backend as K
 
def ignore_class_accuracy(to_ignore=0):
    def ignore_accuracy(y_true, y_pred):
        y_true_class = K.argmax(y_true, axis=-1)
        y_pred_class = K.argmax(y_pred, axis=-1)
 
        ignore_mask = K.cast(K.not_equal(y_pred_class, to_ignore), 'int32')
        matches = K.cast(K.equal(y_true_class, y_pred_class), 'int32') * ignore_mask
        accuracy = K.sum(matches) / K.maximum(K.sum(ignore_mask), 1)
        return accuracy
    return ignore_accuracy
```


```python
from tensorflow.keras.layers import Dense, LSTM, InputLayer, Bidirectional, TimeDistributed, Embedding, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential 

model = Sequential()
model.add(InputLayer(input_shape=(MAX_LENGTH, )))
model.add(Embedding(len(word2index), WORD_EMBED_SIZE, weights=[E]))
model.add(Bidirectional(LSTM(256, return_sequences=True)))
model.add(TimeDistributed(Dense(len(tag2index))))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(0.001),
              metrics=['accuracy', ignore_class_accuracy(0)])
 
model.summary()
 
```

    Model: "sequential_1"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_1 (Embedding)      (None, 214, 200)          5736600   
    _________________________________________________________________
    bidirectional_1 (Bidirection (None, 214, 512)          935936    
    _________________________________________________________________
    time_distributed_1 (TimeDist (None, 214, 36)           18468     
    _________________________________________________________________
    activation_1 (Activation)    (None, 214, 36)           0         
    =================================================================
    Total params: 6,691,004
    Trainable params: 6,691,004
    Non-trainable params: 0
    _________________________________________________________________
    


```python
model.fit(train_sentences_X, to_categorical(train_tags_y, len(tag2index)), batch_size=128, epochs=40, validation_split=0.2)
```

    Train on 5168 samples, validate on 1292 samples
    Epoch 1/40
    


```python
def logits_to_tokens(sequences, index):
    token_sequences = []
    for categorical_sequence in sequences:
        token_sequence = []
        for categorical in categorical_sequence:
            token_sequence.append(index[np.argmax(categorical)])
 
        token_sequences.append(token_sequence)
 
    return token_sequences

# predictions = model.predict(test_samples_X)
# print(logits_to_tokens(predictions, {i: t for t, i in tag2index.items()}))
sample=['በቻይና በመቶዎች የሚቆጠሩ የሕክምና ዶክተሮች በኖቭል ኮሮና ቫይረስ መጠቃታቸው ተነገረ ።','ሁለት ታዳጊዎችን አግቶ ገንዘብ የጠየቀው ግለሰብ የ25 ዓመት እስር ተፈረደበት ።']
sample=[s.split() for s in sample]
test_samples_X = []
for s in sample:
    s_int = []
    for w in s:
        try:
            s_int.append(word2index[w])
        except KeyError:
            s_int.append(word2index['-OOV-'])
    test_samples_X.append(s_int)
 
test_samples_X = pad_sequences(test_samples_X, maxlen=MAX_LENGTH, padding='post')

predictions = model.predict(test_samples_X)
print(logits_to_tokens(predictions, {i: t for t, i in tag2index.items()}))
```

    [['<NUMP>', '<NUMP>', '<ADV>', '<ADV>', '<UNC>', '<UNC>', '<N>', '<V>', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-'], ['<NUMCR>', '<NUMCR>', '<N>', '<N>', '<N>', '<V>', '<NUMP>', '<N>', '<N>', '<V>', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-', '-PAD-']]
    
