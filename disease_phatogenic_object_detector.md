

```python
import skimage
from lxml import etree
import os ## for OS level file processing such as file directory reading
import glob ## to match file name pattern for files on defined directory
from sklearn.model_selection import train_test_split ## to split train, test, and validation sets of dataset
import numpy as np ## for array processing
from skimage import io
from scipy import misc
import get_patches
from sklearn import metrics
import detectobjects as det
import cv2

## directories to load images
MALARIA_DIR='malaria/images/' #'plasmodium/'#
TUBERCLOSIS_DIR='tuberculosis/'
INTESTINAL_PARSITE_DIR='intestinalparasites/'
```


```python
def read_and_split_data(img_dir, train_split=0.6, test_split=0.2, val_split=0.2):
    '''Split a list of image files into training, testing and validation sets.'''

    imgfilenames = glob.glob(img_dir + '*.jpg')
    baseimgfilenames = [os.path.basename(f) for f in imgfilenames]
    all_images=np.arange(len(baseimgfilenames))
    if val_split>0:
        train,val = train_test_split(all_images,
                                           train_size=train_split+test_split,
                                           test_size=val_split,
                                           random_state=1)
        train,test=train_test_split(train,
                                  train_size=1-test_split,
                                  test_size=test_split,
                                  random_state=1)
    else:
        val=[]
        train,test = train_test_split(all_images,
                                  train_size=train_split + val_split,
                                  test_size=test_split,
                                  random_state=1)

    trainfiles = [baseimgfilenames[i] for i in train]
    testfiles = [baseimgfilenames[i] for i in test]
    valfiles = [baseimgfilenames[i] for i in val]

    return trainfiles, valfiles,testfiles

train,test,val=read_and_split_data(MALARIA_DIR)
print(len(train), len(test), len(val))
```


```python
##Prepare bound boxes with posetive and negative classess for training and testing
if os.path.isfile('malaria/bounds/train_X.npy'):
    test_y=np.load('malaria/bounds/test_y.npy')
    test_X=np.load('malaria/bounds/test_X.npy')
    train_X=np.load('malaria/bounds/train_X.npy')
    train_y=np.load('malaria/bounds/train_y.npy')
else:
    opts = {'img_dir': 'malaria/images/',
            'annotation_dir': 'malaria/annotation/',
            'detection_probability_threshold': 0.5,
            'detection_overlap_threshold': 0.3, 
            'gauss': 1,
            'patch_size': (40,40),
            'image_downsample' : 2,
            'detection_step': 5,
            'patch_creation_step': 40,
            'object_class': None,
            'negative_training_discard_rate': .9
           }
    opts['patch_stride_training'] = int(opts['patch_size'][0]*0.25)
    trainfiles, valfiles, testfiles = get_patches.create_sets(opts['img_dir'], train_set_proportion=.5, 
                                                      test_set_proportion=.5,
                                                      val_set_proportion=0)
    print(len(trainfiles), len(valfiles), len(testfiles))
    print('Creating patches ....')
    train_y, train_X = get_patches.create_patches(trainfiles, opts['annotation_dir'], opts['img_dir'], opts['patch_size'][0],
                                                  opts['patch_stride_training'], grayscale=False, progressbar=True, 
                                                  downsample=opts['image_downsample'], objectclass=opts['object_class'],
                                                  negative_discard_rate=opts['negative_training_discard_rate'])
    test_y, test_X = get_patches.create_patches(testfiles,  opts['annotation_dir'], opts['img_dir'], opts['patch_size'][0],
                                                opts['patch_stride_training'], grayscale=False, progressbar=True, 
                                                downsample=opts['image_downsample'], objectclass=opts['object_class'],
                                                negative_discard_rate=opts['negative_training_discard_rate'])

    # Cut down on disproportionately large numbers of negative patches
    train_X, train_y = get_patches.balance(train_X, train_y, mult_neg=200)

    # Create rotated and flipped versions of the positive patches
    train_X, train_y = get_patches.augment_positives(train_X, train_y)
    test_X, test_y = get_patches.augment_positives(test_X, test_y)
    
    np.save('malaria/bounds/train_X.npy', np.array(train_X), allow_pickle=True)
    np.save('malaria/bounds/train_y.npy', np.array(train_y), allow_pickle=True)
    np.save('malaria/bounds/test_X.npy', np.array(test_X), allow_pickle=True)
    np.save('malaria/bounds/test_y.npy', np.array(test_y), allow_pickle=True)
    
print('\n')
print('%d positive training examples, %d negative training examples' % (sum(train_y), len(train_y)-sum(train_y)))
print('%d positive testing examples, %d negative testing examples' % (sum(test_y), len(test_y)-sum(test_y)))
print('%d patches (%.1f%% positive)' % (len(train_y)+len(test_y), 100.*((sum(train_y)+sum(test_y))/(len(train_y)+len(test_y)))))
```

    
    
    187728 positive training examples, 782967 negative training examples
    198312 positive testing examples, 775234 negative testing examples
    1944241 patches (19.9% positive)
    


```python
def CNN(n_epochs):
    net1 = NeuralNet(
        layers=[
        ('input', layers.InputLayer),
        ('conv1', layers.Conv2DLayer),      #Convolutional layer.  Params defined below
        ('pool1', layers.MaxPool2DLayer),   # Like downsampling, for execution speed
        ('conv2', layers.Conv2DLayer),
        ('hidden3', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],
        
    input_shape=(None, 3, opts['patch_size'][0]/opts['image_downsample'], 
                 opts['patch_size'][0]/opts['image_downsample']),
    conv1_num_filters=7, 
    conv1_filter_size=(3, 3), 
    conv1_nonlinearity=lasagne.nonlinearities.rectify,
        
    pool1_pool_size=(2, 2),
        
    conv2_num_filters=12, 
    conv2_filter_size=(2, 2),    
    conv2_nonlinearity=lasagne.nonlinearities.rectify,
        
    hidden3_num_units=500,
    output_num_units=2, 
    output_nonlinearity=lasagne.nonlinearities.softmax,

    update_learning_rate=0.0001,
    update_momentum=0.9,

    max_epochs=n_epochs,
    verbose=1,
    )
    return net1

cnn = CNN(50).fit(train_X, train_y)
```


```python
"""This file is for patches classification (step 1).

Sample predictive model.
You must supply at least 4 methods:
- fit: trains the model.
- predict: uses the model to perform predictions.
- save: saves the model.
- load: reloads the model.
"""
import numpy as np   # We recommend to use numpy arrays
from sklearn.base import BaseEstimator
from resnet import ResNet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.callbacks import LearningRateScheduler
import tensorflow as tf

class model(BaseEstimator):
    """Main class for Classification problem."""

    def __init__(self):
        """Init method.

        """
        self.num_train_samples = 0
        self.num_feat = 1
        self.num_labels = 1
        self.is_trained = False

        self.model=ResNet.build(40, 40, 3, 2, (1, 2),(64, 32, 64, 128), reg=0.0005)
        self.model.compile(loss='binary_crossentropy', optimizer='adam',
                           metrics=['accuracy'])
        self.model.summary()

    def poly_decay(self,epoch):
        NUM_EPOCHS = 5
        INIT_LR = 1e-1
        BS = 16
        maxEpochs = NUM_EPOCHS
        baseLR = INIT_LR
        power = 1.0
        # compute the new learning rate based on polynomial decay
        alpha = baseLR * (1 - (epoch / float(maxEpochs))) ** power
        # return the new learning rate
        return alpha

    def fit(self, X, y):
        """Fit method.

        This function should train the model parameters.
        Args:
            X: Training data matrix of dim num_train_samples * num_feat.
               An image has the following shape (40, 40, 3) then 4800 features.
            y: Training label matrix of dim num_train_samples.
        Both inputs are numpy arrays.
        """
        self.num_train_samples = X.shape[0]
        batch_size=32
        X = X.reshape((self.num_train_samples, 40, 40, 3))
        # initialize the training data augmentation object
        trainAug = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1 / 255.0,
            rotation_range=20,
            zoom_range=0.05,
            width_shift_range=0.05,
            height_shift_range=0.05,
            shear_range=0.05,
            horizontal_flip=True,
            fill_mode="nearest")
        # initialize the training generator
        train_generator = trainAug.flow(
            X,
            y,
            shuffle=True,
            batch_size=batch_size)
        #checkpoint
        callbacks = [LearningRateScheduler(self.poly_decay)]#
        
        self.model.fit_generator(train_generator,
            steps_per_epoch=X.shape[0] // batch_size,
            epochs=5, callbacks=callbacks)
        self.is_trained = True

    def predict(self, X):

        num_test_samples = X.shape[0]
        X = X.reshape((num_test_samples, 40, 40, 3))
        # initialize the validation (and testing) data augmentation object
        testAug = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255.0)
        # initialize the testing generator
        testGen = testAug.flow(
            X,
            shuffle=False,
            batch_size=32)
        # testGen.reset()
        return self.model.predict_generator(testGen)

```


```python
# reload(det)

fname = testfiles[1]
imfile = opts['img_dir'] + fname
opts['detection_probability_threshold'] = 0.95

found = det.detect(imfile, cnn, opts)

im = misc.imread(imfile)

plt.box(False)
plt.xticks([])
plt.yticks([])

annofile = opts['annotation_dir'] + fname[:-3] + 'xml'
bboxes = readdata.get_bounding_boxes_for_single_image(annofile)
for bb in bboxes:
    bb = bb.astype(int)
    cv2.rectangle(im, (bb[0],bb[2]), (bb[1],bb[3]), (255,255,255), 2)  

for f in found:
    f = f.astype(int)
    cv2.rectangle(im, (f[0],f[1]), (f[2],f[3]), (255,0,0), 2)

plt.gcf().set_size_inches(10,10)
plt.title('Detected objects in %s' % (imfile))
plt.imshow(im)

#cv2.imwrite('detectionimages/detected-' + os.path.basename(imfile),im)
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-14-4b406515f3b6> in <module>
          5 opts['detection_probability_threshold'] = 0.95
          6 
    ----> 7 found = det.detect(imfile, cnn, opts)
          8 
          9 im = misc.imread(imfile)
    

    NameError: name 'cnn' is not defined



```python
import matplotlib.pyplot as plt ## for plotting data
import seaborn as sns ## another library to visualize data features

N_samples_to_display = 10
pos_indices = np.where(train_y)[0]
pos_indices = pos_indices[np.random.permutation(len(pos_indices))]
for i in range(N_samples_to_display):
    plt.subplot(2,N_samples_to_display,i+1)
    example_pos = train_X[pos_indices[i],:,:,:]
    example_pos = np.swapaxes(example_pos,0,2)
    plt.axis('off')
    plt.imshow(example_pos)
#     cv2.rectangle(img_bbox,(xmin,ymin),(xmax,ymax),(0,255,0),2)
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     cv2.putText(img_bbox,class_name,(xmin,ymin-10), font, 1,(0,255,0),2)
# plt.subplot(1,2,2)
# plt.title('Image with Bounding Box')
# plt.imshow(img_bbox)
# plt.show()

neg_indices = np.where(train_y==0)[0]
neg_indices = neg_indices[np.random.permutation(len(neg_indices))]
for i in range(N_samples_to_display,2*N_samples_to_display):
    plt.subplot(2,N_samples_to_display,i+1)
    example_neg = train_X[neg_indices[i],:,:,:]
    example_neg = np.swapaxes(example_neg,0,2)
    plt.axis('off')
    plt.imshow(example_neg)
    

plt.gcf().set_size_inches(1.5*N_samples_to_display,3)
```


![png](output_6_0.png)


In the above plot, we the first 10 shows posetively labeled i.e., cell infected and the second 10 samples shows negatively labeled means non-infected box that is randomly selected bound from blod-smear image which does not overlap expert annotated bounding box.
