

```python
# import required libraries
import warnings
warnings.filterwarnings("ignore") #UndefinedMetricWarning
import re
import os 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
import tensorflow as tf 
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import cv2
```


```python
# import the necessary packages
from imutils import paths
import shutil
import os

# grab all image paths in the current split
print("[INFO] processing '{} dataset'...".format(TRAIN))
for classes in os.listdir(data_dir_train):
    print("[INFO] loading images {}...".format(classes))
    classPath=os.path.join(data_dir_train,classes)
    imagePaths = list(paths.list_images(classPath))
    for imagePath in imagePaths:
        # extract class label from the filename
        filename = imagePath.split(os.path.sep)[-1]
        # construct the path to the output directory
        dirPath = os.path.sep.join([BASE_PATH, TRAIN, classes])
        # if the output directory does not exist, create it
        if not os.path.exists(dirPath):
            os.makedirs(dirPath)
        # construct the path to the output image file and copy it
        p = os.path.sep.join([dirPath, filename])
        shutil.copy2(imagePath, p)
```


```python
data_dir_train = "train/train/"
data_dir_test = "test/test/"
```


```python
# import the necessary packages
from imutils import paths
import shutil
import os
# grab all image paths in the current split
print("[INFO] processing '{} dataset'...".format(VAL))
for image in os.listdir(data_dir_test):
    imagePath=os.path.join(data_dir_test,image)
    # extract class label from the filename
#     filename = imagePath.split(os.path.sep)[-1]
    # construct the path to the output directory
    dirPath = os.path.sep.join([BASE_PATH, VAL])
    # if the output directory does not exist, create it
    if not os.path.exists(dirPath):
        os.makedirs(dirPath)
    # construct the path to the output image file and copy it
    p = os.path.sep.join([dirPath, image])
    shutil.copy2(imagePath, p)
```

    [INFO] processing 'testing dataset'...
    


```python
# import the necessary packages
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from imutils import paths
import pickle
import random
# load the VGG16 network and initialize the label encoder
print("[INFO] loading network...")
model = VGG16(weights="imagenet", include_top=False)
le = None
```

    [INFO] loading network...
    


```python
# define the names of the training, testing, and validation
# directories
TRAIN = "training"
VAL = "testing"
BASE_PATH="Dataset"
```


```python
# initialize the list of class label names
CLASSES = ["healthy_wheat","leaf_rust","stem_rust"]

# set the batch size
BATCH_SIZE = 32

# initialize the label encoder file path and the output directory to
# where the extracted features (in CSV file format) will be stored
LE_PATH = os.path.sep.join(["output", "le.cpickle"])
BASE_CSV_PATH = "output"

# set the path to the serialized model after training
MODEL_PATH = os.path.sep.join(["output", "model.cpickle"])
```


```python
# import the necessary packages
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from imutils import paths
import pickle
import random
# load the VGG16 network and initialize the label encoder
print("[INFO] loading network...")
model = VGG16(weights="imagenet", include_top=False)
le = None

# grab all image paths in the current split
print("[INFO] processing '{} dataset'...".format(TRAIN))
p = os.path.sep.join([BASE_PATH, TRAIN])
imagePaths = list(paths.list_images(p))

# randomly shuffle the image paths and then extract the class
# labels from the file paths
random.shuffle(imagePaths)
labels = [p.split(os.path.sep)[-2] for p in imagePaths]

# if the label encoder is None, create it
if le is None:
    le = LabelEncoder()
    le.fit(labels)

# open the output CSV file for writing
csvPath = os.path.sep.join([BASE_CSV_PATH,
    "{}.csv".format(TRAIN)])
csv = open(csvPath, "w")

# loop over the images in batches
for (b, i) in enumerate(range(0, len(imagePaths), BATCH_SIZE)):
    # extract the batch of images and labels, then initialize the
    # list of actual images that will be passed through the network
    # for feature extraction
    print("[INFO] processing batch {}/{}".format(b + 1,
        int(np.ceil(len(imagePaths) / float(BATCH_SIZE)))))
    batchPaths = imagePaths[i:i + BATCH_SIZE]
    batchLabels = le.transform(labels[i:i + BATCH_SIZE])
    batchImages = []

    # loop over the images and labels in the current batch
    for imagePath in batchPaths:
        # load the input image using the Keras helper utility
        # while ensuring the image is resized to 224x224 pixels
        image = load_img(imagePath, target_size=(224, 224))
        image = img_to_array(image)

        # preprocess the image by (1) expanding the dimensions and
        # (2) subtracting the mean RGB pixel intensity from the
        # ImageNet dataset
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)

        # add the image to the batch
        batchImages.append(image)

    # pass the images through the network and use the outputs as
    # our actual features, then reshape the features into a
    # flattened volume
    batchImages = np.vstack(batchImages)
    features = model.predict(batchImages, batch_size=BATCH_SIZE)
    features = features.reshape((features.shape[0], 7 * 7 * 512))

    # loop over the class labels and extracted features
    for (label, vec) in zip(batchLabels, features):
        # construct a row that exists of the class label and
        # extracted features
        vec = ",".join([str(v) for v in vec])
        csv.write("{},{}\n".format(label, vec))

# close the CSV file
csv.close()

# serialize the label encoder to disk
f = open(LE_PATH, "wb")
f.write(pickle.dumps(le))
f.close()
```

    [INFO] loading network...
    [INFO] processing 'training dataset'...
    [INFO] processing batch 1/28
    [INFO] processing batch 2/28
    [INFO] processing batch 3/28
    [INFO] processing batch 4/28
    [INFO] processing batch 5/28
    [INFO] processing batch 6/28
    [INFO] processing batch 7/28
    [INFO] processing batch 8/28
    [INFO] processing batch 9/28
    [INFO] processing batch 10/28
    [INFO] processing batch 11/28
    [INFO] processing batch 12/28
    [INFO] processing batch 13/28
    [INFO] processing batch 14/28
    [INFO] processing batch 15/28
    [INFO] processing batch 16/28
    [INFO] processing batch 17/28
    [INFO] processing batch 18/28
    [INFO] processing batch 19/28
    [INFO] processing batch 20/28
    [INFO] processing batch 21/28
    [INFO] processing batch 22/28
    [INFO] processing batch 23/28
    [INFO] processing batch 24/28
    [INFO] processing batch 25/28
    [INFO] processing batch 26/28
    [INFO] processing batch 27/28
    [INFO] processing batch 28/28
    


```python
# import the necessary packages
import warnings
warnings.filterwarnings("ignore") 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
import os

def load_data_split(splitPath):
	# initialize the data and labels
	data = []
	labels = []

	# loop over the rows in the data split file
	for row in open(splitPath):
		# extract the class label and features from the row
		row = row.strip().split(",")
		label = row[0]
		features = np.array(row[1:], dtype="float")

		# update the data and label lists
		data.append(features)
		labels.append(label)

	# convert the data and labels to NumPy arrays
	data = np.array(data)
	labels = np.array(labels)

	# return a tuple of the data and labels
	return (data, labels)

# derive the paths to the training and testing CSV files
trainingPath = os.path.sep.join([BASE_CSV_PATH, "{}.csv".format(TRAIN)])
# testingPath = os.path.sep.join([config.BASE_CSV_PATH, "{}.csv".format(TEST)])

# load the data from disk
print("[INFO] loading data...")
(trainX, trainY) = load_data_split(trainingPath)
print(trainX.shape, trainY.shape)
# load the label encoder from disk
le = pickle.loads(open(LE_PATH, "rb").read())
```

    [INFO] loading data...
    (876, 25088) (876,)
    


```python

# train the model
print("[INFO] training model...")
model_lg = LogisticRegression(solver="lbfgs", multi_class="auto")
model_lg.fit(trainX, trainY)

# evaluate the model
print("[INFO] evaluating...")
prediction_lg = model_lg.predict(trainX)
print(classification_report(trainY, prediction_lg, target_names=le.classes_))


# serialize the model to disk
print("[INFO] saving model...")
f = open(MODEL_PATH, "wb")
f.write(pickle.dumps(model_lg))
f.close()
```

    [INFO] training model...
    [INFO] evaluating...
                   precision    recall  f1-score   support
    
    healthy_wheat       1.00      1.00      1.00       142
        leaf_rust       0.99      0.99      0.99       358
        stem_rust       0.99      0.99      0.99       376
    
         accuracy                           1.00       876
        macro avg       1.00      1.00      1.00       876
     weighted avg       1.00      1.00      1.00       876
    
    [INFO] saving model...
    


```python
from xgboost import XGBClassifier
# fit model no training data
xgb_clf = XGBClassifier()
print('INFO: Training XGB ...')
xgb_clf.fit(trainX, trainY)

# evaluate the model
print("[INFO] evaluating in training set...")
predsTrain = xgb_clf.predict(trainX)
print(classification_report(trainY, predsTrain, target_names=le.classes_))

# serialize the model to disk
print("[INFO] saving model...")
MODEL_PATH = os.path.sep.join(["output", "model_xgb.cpickle"])
f = open(MODEL_PATH, "wb")
f.write(pickle.dumps(xgb_clf))
f.close()
```

    INFO: Training XGB ...
    [INFO] evaluating in training set...
                   precision    recall  f1-score   support
    
    healthy_wheat       1.00      1.00      1.00       142
        leaf_rust       1.00      0.99      0.99       358
        stem_rust       0.99      1.00      0.99       376
    
         accuracy                           1.00       876
        macro avg       1.00      1.00      1.00       876
     weighted avg       1.00      1.00      1.00       876
    
    [INFO] saving model...
    


```python
# import the necessary packages

# load the VGG16 network and initialize the label encoder
# print("[INFO] loading network...")
# model = VGG16(weights="imagenet", include_top=False)
# le = None

# grab all image paths in the current split
print("[INFO] processing '{} dataset'...".format(VAL))
p = os.path.sep.join([BASE_PATH, VAL])
imagePaths = list(paths.list_images(p))


# open the output CSV file for writing
csvPathTest = os.path.sep.join([BASE_CSV_PATH,
    "{}.csv".format(VAL)])
csvTest = open(csvPathTest, "w")

# loop over the images in batches
for (b, i) in enumerate(range(0, len(imagePaths), BATCH_SIZE)):
    # extract the batch of images and labels, then initialize the
    # list of actual images that will be passed through the network
    # for feature extraction
    print("[INFO] processing batch {}/{}".format(b + 1,
        int(np.ceil(len(imagePaths) / float(BATCH_SIZE)))))
    batchPaths = imagePaths[i:i + BATCH_SIZE]
    batchImages = []

    # loop over the images and labels in the current batch
    for imagePath in batchPaths:
        # load the input image using the Keras helper utility
        # while ensuring the image is resized to 224x224 pixels
        image = load_img(imagePath, target_size=(224, 224))
        image = img_to_array(image)

        # preprocess the image by (1) expanding the dimensions and
        # (2) subtracting the mean RGB pixel intensity from the
        # ImageNet dataset
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)

        # add the image to the batch
        batchImages.append(image)

    # pass the images through the network and use the outputs as
    # our actual features, then reshape the features into a
    # flattened volume
    batchImages = np.vstack(batchImages)
    features = model.predict(batchImages, batch_size=BATCH_SIZE)
    features = features.reshape((features.shape[0], 7 * 7 * 512))

    # loop over the class labels and extracted features
    for vec in features:
        # construct a row that exists of the class label and
        # extracted features
        vec = ",".join([str(v) for v in vec])
        csvTest.write("{}\n".format(vec))

# close the CSV file
csvTest.close()

```

    [INFO] processing 'testing dataset'...
    [INFO] processing batch 1/20
    [INFO] processing batch 2/20
    [INFO] processing batch 3/20
    [INFO] processing batch 4/20
    [INFO] processing batch 5/20
    [INFO] processing batch 6/20
    [INFO] processing batch 7/20
    [INFO] processing batch 8/20
    [INFO] processing batch 9/20
    [INFO] processing batch 10/20
    [INFO] processing batch 11/20
    [INFO] processing batch 12/20
    [INFO] processing batch 13/20
    [INFO] processing batch 14/20
    [INFO] processing batch 15/20
    [INFO] processing batch 16/20
    [INFO] processing batch 17/20
    [INFO] processing batch 18/20
    [INFO] processing batch 19/20
    [INFO] processing batch 20/20
    [INFO] loading data...
    (610, 25088)
    


```python

# import the necessary packages
import warnings
warnings.filterwarnings("ignore") 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import numpy as np
import pickle
import os

def load_data_split(splitPath):
    # initialize the data and labels
    data = []
    # loop over the rows in the data split file
    for row in open(splitPath):
        # extract the class label and features from the row
        row = row.strip().split(",")
        features = np.array(row, dtype="float")
        # update the data and label lists
        data.append(features)
    
    # convert the data and labels to NumPy arrays
    data = np.array(data)
    # return a tuple of the data and labels
    return data

# derive the paths to the training and testing CSV files
testingPath = os.path.sep.join([BASE_CSV_PATH, "{}.csv".format(VAL)])

# load the data from disk
print("[INFO] loading data...")
testX = load_data_split(testingPath)
print(testX.shape)
```

    [INFO] loading data...
    (610, 25088)
    


```python
import os
import pandas as pd
def create_submision(prediction,submission_name):
    test_images=[s.split('.')[0].strip() for s in os.listdir(data_dir_test)]
    print(len(test_images),test_images[:5])
    submission = pd.DataFrame({'ID':test_images})
    # create a dummy dataset
    leaf_rust = pd.Series(range(610), name="leaf_rust", dtype=np.float32)
    stem_rust = pd.Series(range(610), name="stem_rust", dtype=np.float32)
    healthy_wheat = pd.Series(range(610), name="healthy_wheat", dtype=np.float32)
    sub = pd.concat([leaf_rust, stem_rust, healthy_wheat], axis=1)
    for i in tqdm(range(0 ,len(prediction))):
        sub.loc[i] = prediction[i]
    submission = pd.concat([submission,sub], axis=1)
    submission.to_csv("submission/{}.csv".format(submission_name), index=False)
```


```python
# evaluate the model
print("[INFO] evaluating...")
prediction_lg = model_lg.predict_proba(testX)
```

    [INFO] evaluating...
    


```python
create_submision(prediction_lg,'vggnet_feature_extraction_lg')
```

    610 ['008FWT', '00AQXY', '01OJZX', '07OXKK', '085IEC']
    

    100%|██████████████████████████████████████████████████████████████████████████████| 610/610 [00:00<00:00, 1091.22it/s]
    


```python
# evaluate the model
print("[INFO] evaluating...")
prediction_xgb = xgb_clf.predict_proba(testX)
```

    [INFO] evaluating...
    


```python
create_submision(prediction_xgb,'vggnet_feature_extraction_xgb')
```

    610 ['008FWT', '00AQXY', '01OJZX', '07OXKK', '085IEC']
    

    100%|██████████████████████████████████████████████████████████████████████████████| 610/610 [00:00<00:00, 1153.11it/s]
    


```python
from sklearn.svm import SVC
svm_clf = SVC(gamma='auto')
print("[INFO] Training SVM...")
svm_clf.fit(trainX, trainY)

# evaluate the model
print("[INFO] evaluating...")
preds_svm = svm_clf.predict_proba(trainX)
print(classification_report(trainY, preds_svm, target_names=le.classes_))

# serialize the model to disk
print("[INFO] saving model...")
MODEL_PATH = os.path.sep.join(["output", "model_svm.cpickle"])
f = open(MODEL_PATH, "wb")
f.write(pickle.dumps(svm_clf))
f.close()
```

    [INFO] Training SVM...
    [INFO] evaluating...
                   precision    recall  f1-score   support
    
    healthy_wheat       1.00      1.00      1.00       142
        leaf_rust       0.99      1.00      0.99       358
        stem_rust       1.00      0.99      0.99       376
    
         accuracy                           1.00       876
        macro avg       1.00      1.00      1.00       876
     weighted avg       1.00      1.00      1.00       876
    
    [INFO] saving model...
    
