
### Amharic Character Recognition using Deep Neural Network

#### Objective
I will use a [dataset](amharic.zip) consisting of Amharic digits (1-9). I will develop develop a deep, fully-connected ("feed-forward") neural network model that can classify these images. In the process we will discuss:<br>

* Data collection and preparation
* Implementing Deep feed forward Neural networks
  - Libraries
  - Reading Data with data generators
      - Data augmentation
* Transfer Learning




```python
import keras
from keras_preprocessing.image import ImageDataGenerator,array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense, BatchNormalization, Activation
from keras.optimizers  import RMSprop, SGD, Adam
from keras.losses import sparse_categorical_crossentropy
import matplotlib.pyplot as plt
import numpy as np
import os
from keras import models
import cv2
```

## Data Generators

**Motivation**

Data preparation is required when working with neural network and deep learning models. Increasingly data augmentation is also required on more complex object recognition tasks.

Like the rest of Keras, the image augmentation API is simple and powerful.

Keras provides the *ImageDataGenerator* class that defines the configuration for image data preparation and augmentation. This includes capabilities such as:



*   Sample-wise standardization.
*   Feature-wise standardization.
*  ZCA whitening.
*   Random rotation, shifts, shear and flips.
*  Dimension reordering.
*  Save augmented images to disk.






```python
## Using the ImageDataGenerator to: Provide a validation split of 20% of the data rescale the images, perform augmentation 
datagen = ImageDataGenerator(validation_split=0.2,
                             rescale=1./255 ,
                             shear_range=0.2,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             zoom_range=0.2,
                             fill_mode='nearest',
                             horizontal_flip=True)

### Specify directory containing amharic dataset
TRAIN_DIR = 'amharic'


## Use train generator to create train data from entire dataset
train_generator = datagen.flow_from_directory(
    TRAIN_DIR, 
    subset='training',
        target_size=(200, 200),
        batch_size=32,
        class_mode='categorical'
)

# use validation_generator to create dataset from 20% of the data
validation_generator = datagen.flow_from_directory(
    TRAIN_DIR,
    subset='validation',
        target_size=(200, 200),
        batch_size=32,
        class_mode='categorical'
)
```

    Found 95 images belonging to 9 classes.
    Found 18 images belonging to 9 classes.
    

# Data Description

Dataset consists of images of Amharic numbers mostly extracted from the internet. The Idea here is to train a machine learning model, specifically a deep convolutional neural network to learn to recognize the amharic numbers.

## Data preparation

The images were rescaled to 200x200 px with the help of the data generators. 

### Note:

In this tutorial, the dataset was not standardized as in the MNIST dataset, an additional task is to use the lessons learnt from the computer vision class to standardize the dataset to have consistent backgrounds like that from the MNIST dataset.

![alt text](https://conx.readthedocs.io/en/latest/_images/MNIST_6_0.png)






# Amharic Number system

Ethiopic numerals have a familiar quality about them that seems to catch the eye and pique the imagination of the first-time viewer. In particular, the bars above and below the letter-like symbols appear reminiscent of their Roman counterparts. The symbols in between the bars, however, are clearly not of Roman origin. The shapes appear Ethiopic but only half seem to correspond to Ethiopic syllables and in an incomprehensible order.

![alt text](https://www.geez.org/Numerals/images/NumberTable2-cropped.gif)


```python


def plot_images(img_gen, img_title):
    fig, ax = plt.subplots(6,6, figsize=(10,10))
    plt.suptitle(img_title, size=32)
    plt.setp(ax, xticks=[], yticks=[])
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    for (img, label) in img_gen:
        for i in range(6):
            for j in range(6):
                if i*6 + j < 256:
                    ax[i][j].imshow(img[i*3 + j])
        break

plot_images(train_generator, "Subset of training Data")
```


![png](am_0.png)



```python
!mkdir preview

img = load_img('amharic/1/Annotation 2019-05-23 124618.png')  # this is a PIL image
x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

# the .flow() command below generates batches of randomly transformed images
# and saves the results to the `preview/` directory
i = 0
for batch in datagen.flow(x, batch_size=1,
                          save_to_dir='preview', save_prefix='cat', save_format='jpeg'):
    i += 1
    if i > 20:
        break  # otherwise the generator would loop indefinitely
```

# Computer Vision 

First we will cover non deep learning approaches used to pre-process and analyse images 



**1.   Otsu Thresholding**

Segment out the actual image


A technique used to create a binary image that can be used in a fully connected network model
Create image resized to fit into a CNN




```python
from skimage.filters import threshold_otsu

amharic = './amharic/3/Annotation 2019-05-23 125407.png'
```


```python
img = cv2.imread(amharic)
imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(img)
```




    <matplotlib.image.AxesImage at 0x7eff53d8fc18>




![png](am_1.png)



```python
# Get threshold from image
threshold_value = threshold_otsu(imggray)
img_background = imggray > threshold_value


bimage = img_background.astype(np.int)
bimage2 = img_background.astype(np.uint8)
plt.imshow(bimage2, cmap='gray')

```




    <matplotlib.image.AxesImage at 0x7eff55fbd6d8>




![png](am_13_1.png)



```python
img = cv2.imread('amharic/4/Annotation 2019-05-23 125039.png',0)
ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
ret,thresh2 = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
ret,thresh3 = cv2.threshold(img,127,255,cv2.THRESH_TRUNC)
ret,thresh4 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO)
ret,thresh5 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO_INV)

titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]

for i in range(6):
    plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])

plt.show()
```


![png](am_14_0.png)


### 2. Edge Detection




```python
img = cv2.imread('amharic/6/Annotation 2019-05-23 125918.png',0)
edges = cv2.Canny(img,100,200)

plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()
```


![png](am_16_0.png)


## 3. Template Matching

Template Matching is a method for searching and finding the location of a template image in a larger image. OpenCV comes with a function cv2.matchTemplate() for this purpose. It simply slides the template image over the input image (as in 2D convolution) and compares the template and patch of input image under the template image. Several comparison methods are implemented in OpenCV. (You can check docs for more details). It returns a grayscale image, where each pixel denotes how much does the neighbourhood of that pixel match with template.


### 3.1 Single Object template Matching


```python
image = cv2.imread('geez_number.jpg')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.show()
```


![png](am_18_0.png)



```python
template = cv2.imread('geez_template.jpg')
plt.imshow(cv2.cvtColor(template, cv2.COLOR_BGR2RGB))
plt.title("Template")
plt.show()
```


![png](am_19_0.png)



```python
img = cv2.imread('geez_number.jpg',0)
img2 = img.copy()
template = cv2.imread('geez_template.jpg',0)
w, h = template.shape[::-1]

# All the 6 methods for comparison in a list
plt.imshow(img)
methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

for meth in methods:
    img = img2.copy()
    method = eval(meth)

    # Apply template Matching
    res = cv2.matchTemplate(img,template,method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    cv2.rectangle(img,top_left, bottom_right, 255, 2)

    plt.subplot(121),plt.imshow(res)
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(img)
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.suptitle(meth)

    plt.show()
```


![png](am_20_0.png)



![png](am_20_1.png)



![png](am_20_2.png)



![png](am_20_3.png)



![png](am_20_4.png)



![png](am_20_5.png)


###  3.2 Multiple Object Template Matchin


```python
image = cv2.imread('taxis.jpg')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.show()
```


![png](am_22_0.png)



```python
template = cv2.imread('taxis_template_2.jpg')
plt.imshow(cv2.cvtColor(template, cv2.COLOR_BGR2RGB))
plt.title("Template")
plt.show()
```


![png](am_23_0.png)



```python
img_rgb = cv2.imread('taxis.jpg')
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
template = cv2.imread('taxis_template_2.jpg',0)
w, h = template.shape[::-1]

res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
threshold = 0.8
loc = np.where( res >= threshold)
for pt in zip(*loc[::-1]):
    cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)

cv2.imwrite('res.png',img_rgb)

processed_image = plt.imread('res.png')
plt.title("Processed Image with matched templates")
plt.imshow(processed_image)
```




    <matplotlib.image.AxesImage at 0x7eff55528668>




![png](am_24_1.png)



```python
image = cv2.imread('taxis_2.jpg')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.show()
```


![png](am_25_0.png)



```python
template = cv2.imread('taxi_template.jpg')
plt.imshow(cv2.cvtColor(template, cv2.COLOR_BGR2RGB))
plt.title("Template")
plt.show()
```


![png](am_26_0.png)



```python
img_rgb = cv2.imread('taxis_2.jpg')
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
template = cv2.imread('taxi_template.jpg',0)
w, h = template.shape[::-1]

res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
threshold = 0.8
loc = np.where( res >= threshold)
for pt in zip(*loc[::-1]):
    cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)

cv2.imwrite('res2.png',img_rgb)

processed_image = plt.imread('res2.png')
plt.title("Processed Image with matched templates")
plt.imshow(processed_image)
```




    <matplotlib.image.AxesImage at 0x7eff55490860>




![png](am_27_1.png)



# Convolutional Neural Nets with Keras Sequential API

There are two ways to build Keras models: **sequential** and **functional**.

The sequential API allows you to create models layer-by-layer for most problems. It is limited in that it does not allow you to create models that share layers or have multiple inputs or outputs.

Alternatively, the **functional API**  allows you to create models that have a lot more flexibility as you can easily define models where layers connect to more than just the previous and next layers. In fact, you can connect layers to (literally) any other layer. As a result, creating complex networks such as siamese networks and residual networks become possible.

[Documentation](https://keras.io/getting-started/sequential-model-guide/)

[Sequential Vs Functional](https://jovianlin.io/keras-models-sequential-vs-functional/)

# Steps Involved


1.   Initialize Sequential Model
2.   Add layers (conv, Pooling, Dense, flatten etc)
3. Compile Model (Loss function, optimizer, metrics)
4. Fit Model (Means to train the model)




```python
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(200, 200, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(9))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
```

    WARNING:tensorflow:From c:\users\user\appdata\local\programs\python\python36\lib\site-packages\tensorflow\python\framework\op_def_library.py:263: colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Colocations handled automatically by placer.
    WARNING:tensorflow:From c:\users\user\appdata\local\programs\python\python36\lib\site-packages\keras\backend\tensorflow_backend.py:3445: calling dropout (from tensorflow.python.ops.nn_ops) with keep_prob is deprecated and will be removed in a future version.
    Instructions for updating:
    Please use `rate` instead of `keep_prob`. Rate should be set to `rate = 1 - keep_prob`.
    


```python
model.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_1 (Conv2D)            (None, 198, 198, 32)      896       
    _________________________________________________________________
    activation_1 (Activation)    (None, 198, 198, 32)      0         
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 99, 99, 32)        0         
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 97, 97, 32)        9248      
    _________________________________________________________________
    activation_2 (Activation)    (None, 97, 97, 32)        0         
    _________________________________________________________________
    max_pooling2d_2 (MaxPooling2 (None, 48, 48, 32)        0         
    _________________________________________________________________
    conv2d_3 (Conv2D)            (None, 46, 46, 64)        18496     
    _________________________________________________________________
    activation_3 (Activation)    (None, 46, 46, 64)        0         
    _________________________________________________________________
    max_pooling2d_3 (MaxPooling2 (None, 23, 23, 64)        0         
    _________________________________________________________________
    flatten_1 (Flatten)          (None, 33856)             0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 64)                2166848   
    _________________________________________________________________
    activation_4 (Activation)    (None, 64)                0         
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 64)                0         
    _________________________________________________________________
    dense_2 (Dense)              (None, 9)                 585       
    _________________________________________________________________
    activation_5 (Activation)    (None, 9)                 0         
    =================================================================
    Total params: 2,196,073
    Trainable params: 2,196,073
    Non-trainable params: 0
    _________________________________________________________________
    


```python
batch_size = 16
history = model.fit_generator(
        train_generator,
        steps_per_epoch=2000 // batch_size,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=800 // batch_size)
```

    WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow/python/ops/math_ops.py:3066: to_int32 (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use tf.cast instead.
    Epoch 1/50
    125/125 [==============================] - 46s 364ms/step - loss: 2.2660 - acc: 0.1270 - val_loss: 2.1976 - val_acc: 0.1056
    Epoch 2/50
    125/125 [==============================] - 38s 307ms/step - loss: 2.1230 - acc: 0.1800 - val_loss: 2.1956 - val_acc: 0.1233
    Epoch 3/50
    125/125 [==============================] - 40s 320ms/step - loss: 2.0463 - acc: 0.2182 - val_loss: 2.1697 - val_acc: 0.1422
    Epoch 4/50
    125/125 [==============================] - 40s 323ms/step - loss: 1.9005 - acc: 0.2925 - val_loss: 2.1469 - val_acc: 0.1911
    Epoch 5/50
    125/125 [==============================] - 40s 319ms/step - loss: 1.7748 - acc: 0.3444 - val_loss: 2.0834 - val_acc: 0.2144
    Epoch 6/50
    125/125 [==============================] - 39s 308ms/step - loss: 1.6230 - acc: 0.4071 - val_loss: 2.0261 - val_acc: 0.2900
    Epoch 7/50
    125/125 [==============================] - 40s 316ms/step - loss: 1.5100 - acc: 0.4432 - val_loss: 1.9231 - val_acc: 0.3133
    Epoch 8/50
    125/125 [==============================] - 39s 314ms/step - loss: 1.4584 - acc: 0.4799 - val_loss: 1.8885 - val_acc: 0.3733
    Epoch 9/50
    125/125 [==============================] - 40s 317ms/step - loss: 1.3437 - acc: 0.5039 - val_loss: 1.8675 - val_acc: 0.3733
    Epoch 10/50
    125/125 [==============================] - 39s 309ms/step - loss: 1.2557 - acc: 0.5459 - val_loss: 1.7921 - val_acc: 0.4344
    Epoch 11/50
    125/125 [==============================] - 40s 317ms/step - loss: 1.1758 - acc: 0.5763 - val_loss: 1.6952 - val_acc: 0.4633
    Epoch 12/50
    125/125 [==============================] - 40s 319ms/step - loss: 1.1045 - acc: 0.6046 - val_loss: 1.7389 - val_acc: 0.4133
    Epoch 13/50
    125/125 [==============================] - 40s 318ms/step - loss: 1.0729 - acc: 0.6208 - val_loss: 1.6569 - val_acc: 0.4433
    Epoch 14/50
    125/125 [==============================] - 38s 307ms/step - loss: 1.0311 - acc: 0.6303 - val_loss: 1.7715 - val_acc: 0.4367
    Epoch 15/50
    125/125 [==============================] - 39s 316ms/step - loss: 1.0009 - acc: 0.6586 - val_loss: 1.7016 - val_acc: 0.4400
    Epoch 16/50
    125/125 [==============================] - 39s 312ms/step - loss: 0.9397 - acc: 0.6572 - val_loss: 1.8776 - val_acc: 0.3967
    Epoch 17/50
    125/125 [==============================] - 39s 315ms/step - loss: 0.8857 - acc: 0.6751 - val_loss: 1.3812 - val_acc: 0.5711
    Epoch 18/50
    125/125 [==============================] - 38s 306ms/step - loss: 0.8698 - acc: 0.6816 - val_loss: 1.6375 - val_acc: 0.4356
    Epoch 19/50
    125/125 [==============================] - 40s 317ms/step - loss: 0.8408 - acc: 0.6936 - val_loss: 1.3586 - val_acc: 0.6022
    Epoch 20/50
    125/125 [==============================] - 40s 320ms/step - loss: 0.8060 - acc: 0.7142 - val_loss: 1.4839 - val_acc: 0.5189
    Epoch 21/50
    125/125 [==============================] - 40s 317ms/step - loss: 0.8351 - acc: 0.7056 - val_loss: 1.4856 - val_acc: 0.5456
    Epoch 22/50
    125/125 [==============================] - 39s 310ms/step - loss: 0.7839 - acc: 0.7221 - val_loss: 1.4683 - val_acc: 0.5267
    Epoch 23/50
    125/125 [==============================] - 40s 324ms/step - loss: 0.8305 - acc: 0.7068 - val_loss: 1.6215 - val_acc: 0.5400
    Epoch 24/50
    125/125 [==============================] - 39s 311ms/step - loss: 0.8091 - acc: 0.7103 - val_loss: 1.2579 - val_acc: 0.6056
    Epoch 25/50
    125/125 [==============================] - 40s 319ms/step - loss: 0.7652 - acc: 0.7254 - val_loss: 1.3563 - val_acc: 0.5900
    Epoch 26/50
    125/125 [==============================] - 38s 307ms/step - loss: 0.7794 - acc: 0.7317 - val_loss: 1.3431 - val_acc: 0.5678
    Epoch 27/50
    125/125 [==============================] - 40s 318ms/step - loss: 0.7555 - acc: 0.7339 - val_loss: 1.3445 - val_acc: 0.5411
    Epoch 28/50
    125/125 [==============================] - 40s 318ms/step - loss: 0.7213 - acc: 0.7377 - val_loss: 1.1485 - val_acc: 0.6233
    Epoch 29/50
    125/125 [==============================] - 40s 317ms/step - loss: 0.8154 - acc: 0.7360 - val_loss: 1.2860 - val_acc: 0.5756
    Epoch 30/50
    125/125 [==============================] - 39s 310ms/step - loss: 0.7235 - acc: 0.7475 - val_loss: 1.3299 - val_acc: 0.5511
    Epoch 31/50
    125/125 [==============================] - 40s 319ms/step - loss: 0.7306 - acc: 0.7512 - val_loss: 1.3363 - val_acc: 0.5322
    Epoch 32/50
    125/125 [==============================] - 39s 310ms/step - loss: 0.7316 - acc: 0.7366 - val_loss: 1.1590 - val_acc: 0.5967
    Epoch 33/50
    125/125 [==============================] - 40s 318ms/step - loss: 0.6847 - acc: 0.7602 - val_loss: 1.2039 - val_acc: 0.6100
    Epoch 34/50
    125/125 [==============================] - 39s 309ms/step - loss: 0.7102 - acc: 0.7595 - val_loss: 1.3023 - val_acc: 0.5844
    Epoch 35/50
    125/125 [==============================] - 40s 322ms/step - loss: 0.7061 - acc: 0.7554 - val_loss: 1.2237 - val_acc: 0.6267
    Epoch 36/50
    125/125 [==============================] - 39s 313ms/step - loss: 0.7030 - acc: 0.7627 - val_loss: 1.1402 - val_acc: 0.6189
    Epoch 37/50
    125/125 [==============================] - 40s 317ms/step - loss: 0.7481 - acc: 0.7491 - val_loss: 1.3752 - val_acc: 0.5700
    Epoch 38/50
    125/125 [==============================] - 38s 308ms/step - loss: 0.6787 - acc: 0.7682 - val_loss: 1.2422 - val_acc: 0.5700
    Epoch 39/50
    125/125 [==============================] - 41s 324ms/step - loss: 0.6983 - acc: 0.7658 - val_loss: 2.5488 - val_acc: 0.4489
    Epoch 40/50
    125/125 [==============================] - 39s 310ms/step - loss: 0.7723 - acc: 0.7416 - val_loss: 1.2494 - val_acc: 0.5878
    Epoch 41/50
    125/125 [==============================] - 40s 317ms/step - loss: 0.7114 - acc: 0.7622 - val_loss: 1.3310 - val_acc: 0.6000
    Epoch 42/50
    125/125 [==============================] - 39s 311ms/step - loss: 0.7846 - acc: 0.7534 - val_loss: 1.1058 - val_acc: 0.6311
    Epoch 43/50
    125/125 [==============================] - 40s 319ms/step - loss: 0.6706 - acc: 0.7705 - val_loss: 1.3311 - val_acc: 0.5244
    Epoch 44/50
    125/125 [==============================] - 39s 311ms/step - loss: 0.6820 - acc: 0.7644 - val_loss: 1.2269 - val_acc: 0.5822
    Epoch 45/50
    125/125 [==============================] - 39s 311ms/step - loss: 0.6934 - acc: 0.7656 - val_loss: 1.4280 - val_acc: 0.5600
    Epoch 46/50
    125/125 [==============================] - 39s 313ms/step - loss: 0.6762 - acc: 0.7698 - val_loss: 1.2354 - val_acc: 0.6667
    Epoch 47/50
    125/125 [==============================] - 39s 315ms/step - loss: 0.6923 - acc: 0.7718 - val_loss: 1.2367 - val_acc: 0.6311
    Epoch 48/50
    125/125 [==============================] - 39s 314ms/step - loss: 0.7104 - acc: 0.7611 - val_loss: 1.1435 - val_acc: 0.6156
    Epoch 49/50
    125/125 [==============================] - 38s 305ms/step - loss: 0.6568 - acc: 0.7828 - val_loss: 1.1379 - val_acc: 0.6322
    Epoch 50/50
    125/125 [==============================] - 39s 316ms/step - loss: 0.7349 - acc: 0.7598 - val_loss: 0.9543 - val_acc: 0.7189
    


```python
### Save weights
model.save_weights('first.h5')
```

# Transfer Learning

Transfer learning is a machine learning technique where a model trained on one task is re-purposed on a second related task.

## Transfer Learning with Image Data
It is common to perform transfer learning with predictive modeling problems that use image data as input.

This may be a prediction task that takes photographs or video data as input.

For these types of problems, it is common to use a deep learning model pre-trained for a large and challenging image classification task such as the ImageNet 1000-class photograph classification competition.

The research organizations that develop models for this competition and do well often release their final model under a permissive license for reuse. These models can take days or weeks to train on modern hardware.

These models can be downloaded and incorporated directly into new models that expect image data as input.



 [More info](https://machinelearningmastery.com/transfer-learning-for-deep-learning/)
 
 
 ![alt text](https://cdn-images-1.medium.com/max/1600/1*9GTEzcO8KxxrfutmtsPs3Q.png)


```python
from keras.applications import VGG16
```

# VGG 16

![alt text](https://neurohive.io/wp-content/uploads/2018/11/vgg16-1-e1542731207177.png)


```python
# Use imagenet weights
vgg_conv = VGG16(weights='imagenet',
                include_top = False, 
                input_shape = (200,200,3))
```

    WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow/python/framework/op_def_library.py:263: colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Colocations handled automatically by placer.
    Downloading data from https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5
    58892288/58889256 [==============================] - 4s 0us/step
    


```python
vgg_conv.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_2 (InputLayer)         (None, 200, 200, 3)       0         
    _________________________________________________________________
    block1_conv1 (Conv2D)        (None, 200, 200, 64)      1792      
    _________________________________________________________________
    block1_conv2 (Conv2D)        (None, 200, 200, 64)      36928     
    _________________________________________________________________
    block1_pool (MaxPooling2D)   (None, 100, 100, 64)      0         
    _________________________________________________________________
    block2_conv1 (Conv2D)        (None, 100, 100, 128)     73856     
    _________________________________________________________________
    block2_conv2 (Conv2D)        (None, 100, 100, 128)     147584    
    _________________________________________________________________
    block2_pool (MaxPooling2D)   (None, 50, 50, 128)       0         
    _________________________________________________________________
    block3_conv1 (Conv2D)        (None, 50, 50, 256)       295168    
    _________________________________________________________________
    block3_conv2 (Conv2D)        (None, 50, 50, 256)       590080    
    _________________________________________________________________
    block3_conv3 (Conv2D)        (None, 50, 50, 256)       590080    
    _________________________________________________________________
    block3_pool (MaxPooling2D)   (None, 25, 25, 256)       0         
    _________________________________________________________________
    block4_conv1 (Conv2D)        (None, 25, 25, 512)       1180160   
    _________________________________________________________________
    block4_conv2 (Conv2D)        (None, 25, 25, 512)       2359808   
    _________________________________________________________________
    block4_conv3 (Conv2D)        (None, 25, 25, 512)       2359808   
    _________________________________________________________________
    block4_pool (MaxPooling2D)   (None, 12, 12, 512)       0         
    _________________________________________________________________
    block5_conv1 (Conv2D)        (None, 12, 12, 512)       2359808   
    _________________________________________________________________
    block5_conv2 (Conv2D)        (None, 12, 12, 512)       2359808   
    _________________________________________________________________
    block5_conv3 (Conv2D)        (None, 12, 12, 512)       2359808   
    _________________________________________________________________
    block5_pool (MaxPooling2D)   (None, 6, 6, 512)         0         
    =================================================================
    Total params: 14,714,688
    Trainable params: 14,714,688
    Non-trainable params: 0
    _________________________________________________________________
    


```python
### Add our custom laters

tr_model = vgg_conv.output
tr_model = Flatten()(tr_model)
tr_model = Dense(64, activation='relu')(tr_model)
tr_model = Dropout(0.5)(tr_model)
tr_model = Dense(9, activation='softmax')(tr_model)
```

    WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:3445: calling dropout (from tensorflow.python.ops.nn_ops) with keep_prob is deprecated and will be removed in a future version.
    Instructions for updating:
    Please use `rate` instead of `keep_prob`. Rate should be set to `rate = 1 - keep_prob`.
    


```python
new_model = models.Model(inputs=vgg_conv.input, outputs=tr_model)
```

# Freezing and Fine-tuning vgg16

![alt text](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAfQAAAOeCAYAAAD1JuKfAAAACXBIWXMAAAsTAAALEwEAmpwYAABDu2lUWHRYTUw6Y29tLmFkb2JlLnhtcAAAAAAAPD94cGFja2V0IGJlZ2luPSLvu78iIGlkPSJXNU0wTXBDZWhpSHpyZVN6TlRjemtjOWQiPz4KPHg6eG1wbWV0YSB4bWxuczp4PSJhZG9iZTpuczptZXRhLyIgeDp4bXB0az0iQWRvYmUgWE1QIENvcmUgNS42LWMwNjcgNzkuMTU3NzQ3LCAyMDE1LzAzLzMwLTIzOjQwOjQyICAgICAgICAiPgogICA8cmRmOlJERiB4bWxuczpyZGY9Imh0dHA6Ly93d3cudzMub3JnLzE5OTkvMDIvMjItcmRmLXN5bnRheC1ucyMiPgogICAgICA8cmRmOkRlc2NyaXB0aW9uIHJkZjphYm91dD0iIgogICAgICAgICAgICB4bWxuczp4bXA9Imh0dHA6Ly9ucy5hZG9iZS5jb20veGFwLzEuMC8iCiAgICAgICAgICAgIHhtbG5zOmRjPSJodHRwOi8vcHVybC5vcmcvZGMvZWxlbWVudHMvMS4xLyIKICAgICAgICAgICAgeG1sbnM6cGhvdG9zaG9wPSJodHRwOi8vbnMuYWRvYmUuY29tL3Bob3Rvc2hvcC8xLjAvIgogICAgICAgICAgICB4bWxuczp4bXBNTT0iaHR0cDovL25zLmFkb2JlLmNvbS94YXAvMS4wL21tLyIKICAgICAgICAgICAgeG1sbnM6c3RFdnQ9Imh0dHA6Ly9ucy5hZG9iZS5jb20veGFwLzEuMC9zVHlwZS9SZXNvdXJjZUV2ZW50IyIKICAgICAgICAgICAgeG1sbnM6c3RSZWY9Imh0dHA6Ly9ucy5hZG9iZS5jb20veGFwLzEuMC9zVHlwZS9SZXNvdXJjZVJlZiMiCiAgICAgICAgICAgIHhtbG5zOnRpZmY9Imh0dHA6Ly9ucy5hZG9iZS5jb20vdGlmZi8xLjAvIgogICAgICAgICAgICB4bWxuczpleGlmPSJodHRwOi8vbnMuYWRvYmUuY29tL2V4aWYvMS4wLyI+CiAgICAgICAgIDx4bXA6Q3JlYXRvclRvb2w+QWRvYmUgUGhvdG9zaG9wIENDIDIwMTUgKE1hY2ludG9zaCk8L3htcDpDcmVhdG9yVG9vbD4KICAgICAgICAgPHhtcDpDcmVhdGVEYXRlPjIwMTYtMDYtMDVUMTk6MDI6MDktMDc6MDA8L3htcDpDcmVhdGVEYXRlPgogICAgICAgICA8eG1wOk1vZGlmeURhdGU+MjAxNi0wNi0wNlQyMTo0MjowOC0wNzowMDwveG1wOk1vZGlmeURhdGU+CiAgICAgICAgIDx4bXA6TWV0YWRhdGFEYXRlPjIwMTYtMDYtMDZUMjE6NDI6MDgtMDc6MDA8L3htcDpNZXRhZGF0YURhdGU+CiAgICAgICAgIDxkYzpmb3JtYXQ+aW1hZ2UvcG5nPC9kYzpmb3JtYXQ+CiAgICAgICAgIDxwaG90b3Nob3A6Q29sb3JNb2RlPjM8L3Bob3Rvc2hvcDpDb2xvck1vZGU+CiAgICAgICAgIDx4bXBNTTpJbnN0YW5jZUlEPnhtcC5paWQ6ZTllNTAwZmMtOGVhMi00NDRiLTk5YmQtNTE2NDc5NzllNzBmPC94bXBNTTpJbnN0YW5jZUlEPgogICAgICAgICA8eG1wTU06RG9jdW1lbnRJRD5hZG9iZTpkb2NpZDpwaG90b3Nob3A6Mzg3ODZkNTEtNmMzZi0xMTc5LTliYTQtZGM1OTI3YTU1ZTVkPC94bXBNTTpEb2N1bWVudElEPgogICAgICAgICA8eG1wTU06T3JpZ2luYWxEb2N1bWVudElEPnhtcC5kaWQ6MWJhZTViNmEtMTk5MS00ZTRjLTgzOWItMzYwNWQzNjMyNmFlPC94bXBNTTpPcmlnaW5hbERvY3VtZW50SUQ+CiAgICAgICAgIDx4bXBNTTpIaXN0b3J5PgogICAgICAgICAgICA8cmRmOlNlcT4KICAgICAgICAgICAgICAgPHJkZjpsaSByZGY6cGFyc2VUeXBlPSJSZXNvdXJjZSI+CiAgICAgICAgICAgICAgICAgIDxzdEV2dDphY3Rpb24+Y3JlYXRlZDwvc3RFdnQ6YWN0aW9uPgogICAgICAgICAgICAgICAgICA8c3RFdnQ6aW5zdGFuY2VJRD54bXAuaWlkOjFiYWU1YjZhLTE5OTEtNGU0Yy04MzliLTM2MDVkMzYzMjZhZTwvc3RFdnQ6aW5zdGFuY2VJRD4KICAgICAgICAgICAgICAgICAgPHN0RXZ0OndoZW4+MjAxNi0wNi0wNVQxOTowMjowOS0wNzowMDwvc3RFdnQ6d2hlbj4KICAgICAgICAgICAgICAgICAgPHN0RXZ0OnNvZnR3YXJlQWdlbnQ+QWRvYmUgUGhvdG9zaG9wIENDIDIwMTUgKE1hY2ludG9zaCk8L3N0RXZ0OnNvZnR3YXJlQWdlbnQ+CiAgICAgICAgICAgICAgIDwvcmRmOmxpPgogICAgICAgICAgICAgICA8cmRmOmxpIHJkZjpwYXJzZVR5cGU9IlJlc291cmNlIj4KICAgICAgICAgICAgICAgICAgPHN0RXZ0OmFjdGlvbj5jb252ZXJ0ZWQ8L3N0RXZ0OmFjdGlvbj4KICAgICAgICAgICAgICAgICAgPHN0RXZ0OnBhcmFtZXRlcnM+ZnJvbSBpbWFnZS9wbmcgdG8gYXBwbGljYXRpb24vdm5kLmFkb2JlLnBob3Rvc2hvcDwvc3RFdnQ6cGFyYW1ldGVycz4KICAgICAgICAgICAgICAgPC9yZGY6bGk+CiAgICAgICAgICAgICAgIDxyZGY6bGkgcmRmOnBhcnNlVHlwZT0iUmVzb3VyY2UiPgogICAgICAgICAgICAgICAgICA8c3RFdnQ6YWN0aW9uPnNhdmVkPC9zdEV2dDphY3Rpb24+CiAgICAgICAgICAgICAgICAgIDxzdEV2dDppbnN0YW5jZUlEPnhtcC5paWQ6MjU3MDgxODQtMGZjNi00YWJhLThhYzYtMTM4OWQ4YzQ1Mzg0PC9zdEV2dDppbnN0YW5jZUlEPgogICAgICAgICAgICAgICAgICA8c3RFdnQ6d2hlbj4yMDE2LTA2LTA2VDIwOjU5OjE2LTA3OjAwPC9zdEV2dDp3aGVuPgogICAgICAgICAgICAgICAgICA8c3RFdnQ6c29mdHdhcmVBZ2VudD5BZG9iZSBQaG90b3Nob3AgQ0MgMjAxNSAoTWFjaW50b3NoKTwvc3RFdnQ6c29mdHdhcmVBZ2VudD4KICAgICAgICAgICAgICAgICAgPHN0RXZ0OmNoYW5nZWQ+Lzwvc3RFdnQ6Y2hhbmdlZD4KICAgICAgICAgICAgICAgPC9yZGY6bGk+CiAgICAgICAgICAgICAgIDxyZGY6bGkgcmRmOnBhcnNlVHlwZT0iUmVzb3VyY2UiPgogICAgICAgICAgICAgICAgICA8c3RFdnQ6YWN0aW9uPnNhdmVkPC9zdEV2dDphY3Rpb24+CiAgICAgICAgICAgICAgICAgIDxzdEV2dDppbnN0YW5jZUlEPnhtcC5paWQ6MzI2N2VkYjYtOGU3Mi00ODgzLWJlODEtYzE5NTJlMWZhY2YwPC9zdEV2dDppbnN0YW5jZUlEPgogICAgICAgICAgICAgICAgICA8c3RFdnQ6d2hlbj4yMDE2LTA2LTA2VDIxOjA1OjQwLTA3OjAwPC9zdEV2dDp3aGVuPgogICAgICAgICAgICAgICAgICA8c3RFdnQ6c29mdHdhcmVBZ2VudD5BZG9iZSBQaG90b3Nob3AgQ0MgMjAxNSAoTWFjaW50b3NoKTwvc3RFdnQ6c29mdHdhcmVBZ2VudD4KICAgICAgICAgICAgICAgICAgPHN0RXZ0OmNoYW5nZWQ+Lzwvc3RFdnQ6Y2hhbmdlZD4KICAgICAgICAgICAgICAgPC9yZGY6bGk+CiAgICAgICAgICAgICAgIDxyZGY6bGkgcmRmOnBhcnNlVHlwZT0iUmVzb3VyY2UiPgogICAgICAgICAgICAgICAgICA8c3RFdnQ6YWN0aW9uPmNvbnZlcnRlZDwvc3RFdnQ6YWN0aW9uPgogICAgICAgICAgICAgICAgICA8c3RFdnQ6cGFyYW1ldGVycz5mcm9tIGFwcGxpY2F0aW9uL3ZuZC5hZG9iZS5waG90b3Nob3AgdG8gaW1hZ2UvcG5nPC9zdEV2dDpwYXJhbWV0ZXJzPgogICAgICAgICAgICAgICA8L3JkZjpsaT4KICAgICAgICAgICAgICAgPHJkZjpsaSByZGY6cGFyc2VUeXBlPSJSZXNvdXJjZSI+CiAgICAgICAgICAgICAgICAgIDxzdEV2dDphY3Rpb24+ZGVyaXZlZDwvc3RFdnQ6YWN0aW9uPgogICAgICAgICAgICAgICAgICA8c3RFdnQ6cGFyYW1ldGVycz5jb252ZXJ0ZWQgZnJvbSBhcHBsaWNhdGlvbi92bmQuYWRvYmUucGhvdG9zaG9wIHRvIGltYWdlL3BuZzwvc3RFdnQ6cGFyYW1ldGVycz4KICAgICAgICAgICAgICAgPC9yZGY6bGk+CiAgICAgICAgICAgICAgIDxyZGY6bGkgcmRmOnBhcnNlVHlwZT0iUmVzb3VyY2UiPgogICAgICAgICAgICAgICAgICA8c3RFdnQ6YWN0aW9uPnNhdmVkPC9zdEV2dDphY3Rpb24+CiAgICAgICAgICAgICAgICAgIDxzdEV2dDppbnN0YW5jZUlEPnhtcC5paWQ6Y2QzNzdmMTktMDZmYS00MmNiLWEzNmUtZGMzNDBlODNiYzY2PC9zdEV2dDppbnN0YW5jZUlEPgogICAgICAgICAgICAgICAgICA8c3RFdnQ6d2hlbj4yMDE2LTA2LTA2VDIxOjA1OjQwLTA3OjAwPC9zdEV2dDp3aGVuPgogICAgICAgICAgICAgICAgICA8c3RFdnQ6c29mdHdhcmVBZ2VudD5BZG9iZSBQaG90b3Nob3AgQ0MgMjAxNSAoTWFjaW50b3NoKTwvc3RFdnQ6c29mdHdhcmVBZ2VudD4KICAgICAgICAgICAgICAgICAgPHN0RXZ0OmNoYW5nZWQ+Lzwvc3RFdnQ6Y2hhbmdlZD4KICAgICAgICAgICAgICAgPC9yZGY6bGk+CiAgICAgICAgICAgICAgIDxyZGY6bGkgcmRmOnBhcnNlVHlwZT0iUmVzb3VyY2UiPgogICAgICAgICAgICAgICAgICA8c3RFdnQ6YWN0aW9uPnNhdmVkPC9zdEV2dDphY3Rpb24+CiAgICAgICAgICAgICAgICAgIDxzdEV2dDppbnN0YW5jZUlEPnhtcC5paWQ6ZTllNTAwZmMtOGVhMi00NDRiLTk5YmQtNTE2NDc5NzllNzBmPC9zdEV2dDppbnN0YW5jZUlEPgogICAgICAgICAgICAgICAgICA8c3RFdnQ6d2hlbj4yMDE2LTA2LTA2VDIxOjQyOjA4LTA3OjAwPC9zdEV2dDp3aGVuPgogICAgICAgICAgICAgICAgICA8c3RFdnQ6c29mdHdhcmVBZ2VudD5BZG9iZSBQaG90b3Nob3AgQ0MgMjAxNSAoTWFjaW50b3NoKTwvc3RFdnQ6c29mdHdhcmVBZ2VudD4KICAgICAgICAgICAgICAgICAgPHN0RXZ0OmNoYW5nZWQ+Lzwvc3RFdnQ6Y2hhbmdlZD4KICAgICAgICAgICAgICAgPC9yZGY6bGk+CiAgICAgICAgICAgIDwvcmRmOlNlcT4KICAgICAgICAgPC94bXBNTTpIaXN0b3J5PgogICAgICAgICA8eG1wTU06RGVyaXZlZEZyb20gcmRmOnBhcnNlVHlwZT0iUmVzb3VyY2UiPgogICAgICAgICAgICA8c3RSZWY6aW5zdGFuY2VJRD54bXAuaWlkOjMyNjdlZGI2LThlNzItNDg4My1iZTgxLWMxOTUyZTFmYWNmMDwvc3RSZWY6aW5zdGFuY2VJRD4KICAgICAgICAgICAgPHN0UmVmOmRvY3VtZW50SUQ+eG1wLmRpZDoxYmFlNWI2YS0xOTkxLTRlNGMtODM5Yi0zNjA1ZDM2MzI2YWU8L3N0UmVmOmRvY3VtZW50SUQ+CiAgICAgICAgICAgIDxzdFJlZjpvcmlnaW5hbERvY3VtZW50SUQ+eG1wLmRpZDoxYmFlNWI2YS0xOTkxLTRlNGMtODM5Yi0zNjA1ZDM2MzI2YWU8L3N0UmVmOm9yaWdpbmFsRG9jdW1lbnRJRD4KICAgICAgICAgPC94bXBNTTpEZXJpdmVkRnJvbT4KICAgICAgICAgPHRpZmY6T3JpZW50YXRpb24+MTwvdGlmZjpPcmllbnRhdGlvbj4KICAgICAgICAgPHRpZmY6WFJlc29sdXRpb24+NzIwMDAwLzEwMDAwPC90aWZmOlhSZXNvbHV0aW9uPgogICAgICAgICA8dGlmZjpZUmVzb2x1dGlvbj43MjAwMDAvMTAwMDA8L3RpZmY6WVJlc29sdXRpb24+CiAgICAgICAgIDx0aWZmOlJlc29sdXRpb25Vbml0PjI8L3RpZmY6UmVzb2x1dGlvblVuaXQ+CiAgICAgICAgIDxleGlmOkNvbG9yU3BhY2U+NjU1MzU8L2V4aWY6Q29sb3JTcGFjZT4KICAgICAgICAgPGV4aWY6UGl4ZWxYRGltZW5zaW9uPjUwMDwvZXhpZjpQaXhlbFhEaW1lbnNpb24+CiAgICAgICAgIDxleGlmOlBpeGVsWURpbWVuc2lvbj45MjY8L2V4aWY6UGl4ZWxZRGltZW5zaW9uPgogICAgICA8L3JkZjpEZXNjcmlwdGlvbj4KICAgPC9yZGY6UkRGPgo8L3g6eG1wbWV0YT4KICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAKPD94cGFja2V0IGVuZD0idyI/PrKyrkgAAAAgY0hSTQAAeiUAAICDAAD5/wAAgOkAAHUwAADqYAAAOpgAABdvkl/FRgABYs1JREFUeNrs3XWYVOXfx/H35M5sN5vE0t1Id0kqqYIgKAIGKqmACIJYSCoGICkGIl0iCEhLCUgusMB21+zs9PPHuudhCeNnwML3dV1espPn3HNmPueOc98ql8t1HQgG8hFCCCFEcWMAkrWAN6D/7T8hhBBCFD/eaikDIYQQoviTQBdCCCEk0IUQQgghgS6EEEIICXQhhBBCSKALIYQQEuhCCCGEkEAXQgghhAS6EEIIISTQhRBCCAl0IYQQQkigCyGEEEICXQghhBAS6EIIIYQEuhBCCCEk0IUQf9GxY8e4ePHi/byLdsAkn7QQEuhC3NfGjx/PsmXL7uddzAVi5JMWQgJdiPuaXq/H5XLd779BWvmkhZBAF0IIIYQEuhBCCCGBLoQQQggJdCGEEEJIoAshhBBCAl2I4sBgMNzvo9xdgEM+aSH+PpXL5coCvKUoxH9p3759PNGvH+5uWqwWM2q1Rgrlxi+mSoXL5SI2IQkPd3eCA/yw2e334646ABtg+F9fINtkYfjw4UyePFkOHPEgy5brP8VdkZWVybWrV4koVwuvoFLYbVYplJuqrTqdjnC3APItVpxewagc92Wga377738+6Um+spdL0RfloBEPPAl0cXeqZfaCVtYJMxbxUKs6pKdImRRNKwgKhs/fnUOWKYfRb08kMVaK5UZqTcGZT7uq/pjzZPZYISTQxV2Vk51LegpkpmVJYdyU6BqVN+a8HPLN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HDhA9+7dWb9+PT169MBmszFp0iSgYADa999/DxT0azscDg4fPgwU9MFHR0cr7x0ZGamMrn/mmWf+cFsLL5tbvHgxa9as4a233mLYsGF06tSJAQMGAPDTTz9x6dIlABYtWoTVaqV///506dKFdevW8eSTT7Jp0yZlBHvhdhe+/9GjRylTpgwajYbPPvuMcePGATB37lzS0tJYs2YNq1evVgawvfzyyzz77LPExsbecbtPnz6t1OavX7/OokWLlO4JgF27dlGiRAm6d+9eLI8hzeTJk18D3OTrJP5LF86f58uvvqLdo4MIjSxFvtnyt17Pw8vA/p0/UMILWrdp96efl5ycTGxsLGq1GqPRqDRR3o1mOZvNRk5ODiqVCq32r00RYbVayczMxM3N7R+ZxMNisZCRkYHBYFBeLzU1VbkGOCMjA5vN9pem3QRwOZ1s27YNN79SRJatit36v3/uKlXBdn3z+UyqVCxDnz6PFfvvRbNmzWjTpg0Wi4WjR4+yfft2Dh48SO3atfn6669p27at8tguXbrg5eXF4cOH2bp1KxqNhs8//5yuXbsCMHPmTE6cOEHdunWJiYkhLi6Offv2UatWLa5evYqHhweNGjVSXq9EiRIYDAb69et32+u4bxQREYHBYCA1NZWNGzdy+vRpnn32WWbPno2bmxtpaWmMHj0anU5H5cqVOXToEDVq1KB06dL06NEDlUrFnj172LBhA35+fqxZs4ZatWoBULlyZerUqUNWVhahoaFUr16d+fPn06JFC44ePcqsWbOIioqibNmyHDhwAK1Wi4+PD6VLl+bkyZOEhIRQp06d2273Z599xtKlS2natCl+fn5s3ryZihUrUrlyZQCysrL4+eefady48R+O8L8HWVQulysL8JaIEf+lDevX0617d95btps6jZuTmfb3ZgwLDvNhxsRx1AhzMXX6H/enrVmzhh07dlCjRg18fHwwmUz8+OOPBAcHM23atD/8QfunORwOPv30U7744gs+/PBDateu/Zee//PPPzNx4kQ+++yzWyYX+au+/fZbNm3axJUrV9BoNLz++uu0bNmS06dP8/bbb5OdnU23bt3w9fXl2rVrJCYm0qFDhyJhc8f9tNsYOXIk3lHNaNS+z9+aKU6t1gAuejaJoGfX1qz6do18sW46ydPpdKhUBZdz2u125VIvp9OJw+FQ/v43jmen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27b069dPeX2Hw4Gvry87duygVq1alCtXjo8//vgvhfknn3zCt99+y9dff11kwJ5Op6NmzZqEhoZKoAshhPjrypUrh4eHBwkJCRw8ePAPH18YyDeHf5s2bcjPzyc7u6Arw2Kx/G4gA0RFRTF48GBOnDjBu+++y+LFi+nXr9+f2u4bB6YFBgYCKO99OxkZGWRkZJCWlqbcFhQUREREBGazmbi4OHJycrBarUorQHBwMCtWrGDgwIG/nby70Ol0mEwm7HY7x44d49q1a3+6rFetWsWkSZP45ptvqFix4g2VguJ/himBLoQQd1nFihXp3r07UHB51e3C5ddff+XFF1/E4XBQrlw5APbt21fkMRcvXiQwMPAPa5g3j+guHIj36quv4nK56N+//5/abjc3N+Xf6enpANSrV+/WoPmt9lynTh0AfvrppyL35+fn4+PjQ1RUFH5+fphMJj7++GPl/suXL/Puu+/icrlwd3cnPT2dHj168NFHH3HmzBnat29PamrqH27vL7/8wsyZM5Xme4BLly4xc+bM+2KUuwS6EELcA959911q1KjBtm3b6NSpE3Fxccp9e/fupWXLlhw5cgSNRkOXLl0oWbIkM2fOZMOGDQDs3LmT3bt307ZtW6UP+MbAvdHNk7E0a9aMrl27AjBo0CC8vb1/d1sLa8+//vqrUuOfPXs2gHJicrv379evHzqdjldffVVpiVi4cCGnT59mzJgx+Pj40KBBA+XkYtCgQYwbN45GjRpx+fJlVCoVZrOZ/Px8zp8/z5AhQ5g5cybnz5+nY8eOXLp06Y7bnJSURO/evUlISODQoUOMGTOG0aNH06dPHy5evAgUXN72zDPPsHDhwmJ5DMmgOCGEuAdERESwY8cOnn76adavX0+jRo0oX7486enpXL9+nccff1wZAOfj48PatWtp2bIl3bp1o0WLFhw9epQ+ffooNdt33nmH999/H4DJkyeTmprKjBkzgIIBYf7+/gwaNEh5/8cff5x9+/bRq1evP9zWGjVq4O/vz5IlS7h48SJxcXGkpKSwfft26tSpQ25uLi+99JJSE+/RowcrVqygZs2arFmzhieffJJOnTpRrVo1fvrpJwYNGsTLL78MwLRp07h27RobN25kyZIlyonAp59+SnR0NAMGDCArK4usrCweeeQRqlSpAsDRo0d56KGH+PTTT4v00QOYTCYe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o0aNGDRoEOPHj1deaoYMGcKOHTuoVKkSixYtYvDgwQBUrVpVCebFwb9bt26UK1dOqXFwOBxKTUSxbt264enpyapVqySTCSEBXYi/n1arVQLTrWRlZTF8+HAaNmxIQEAAAQEBvPzyy/zyyy9s27aN5s2bExUVRcWKFalevTrPPfccqampnDhxAqPRyHvvvcf3339Pbm4uP/74I23atKFKlSpMmTKFRYsW8fjjj6PRaOjatStVqlTh/fffV2oJAKVqPSoqCk9PTwB8fHyIiYlRrj8iIgIvLy8qVaoEQL169fD19cXf358zZ84wefJkevTogVar5YEHHqBnz5588MEH5OXl0bZtW8qUKUODBg1o2rQpXbt25fjx4zekw/Lly/H39+fDDz9UtqlUqn9P3VqyM5zb7cZisZT6ZWuFkIAuRCnx6KOPArBo0aKb7rfZbEq7d3p6urI9JiYGLy8v8vPzlW1eXl4A6PV6fHx8lKVN+/TpQ61atejcuTNpaWk0bNiwxEuE1WpVvqNmzZro9XoAPD098fDwwGAwAJCbm6u0f7vd7hK93wsKCtDpdMpLgEqlQqfToVarlesobosvLnkHBARgsVhQqVRoNBrl+k0mk/LvYrt27cLX15dx48bh4eGhXLNer8dkMmEymUoc//nnn+N0Ounfv79kMiEkoAvx92vevDnTpk3js88+Y8SIEezZs4dTp06xe/duVq5cydq1awkKCuLjjz/mhx9+YMeOHQAsWbKEhg0b0qFDBxISErh06ZLSIe3AgQOkpKSQmpqqnOfNN99k69athIeHExkZCcCAAQNo1KgRY8aMIT09ndTUVI4dO6ZUd+/du5fk5GQOHToEFFX/79mzhy1btrBjxw6Sk5NJSEgAoEKFChw9epSFCxcCcOLECdLT09m/fz8VKlTgiSee4M033+T8+fO43W7Wr1/PY489RnBwMPv27SM5OZkLFy4owfvUqVPk5eWRmZlJ3759GTduHEeOHGHSpEk8++yzLF68mPz8fPbt28fly5dZt24dJ06cYN++fbz33ntMmzaNOXPmKC9MQog7TzrFifvOkCFDaNasGd9//z1r167Fz88Pu91O+fLllY5jo0ePJioqiv3792O1WqlcuTLPPPOMUjoeN24c5cuXx2azERERwYsvvqhUhwPUrl2b9evX07x5c2Wbj48Py5YtY8GCBWzb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)


```python
def print_layer_trainable():
    for layer in new_model.layers:
        print("{0}:\t{1}".format(layer.trainable, layer.name))
```


```python
for i, layer in enumerate(new_model.layers):
    print(i, layer.name)
```

    0 input_2
    1 block1_conv1
    2 block1_conv2
    3 block1_pool
    4 block2_conv1
    5 block2_conv2
    6 block2_pool
    7 block3_conv1
    8 block3_conv2
    9 block3_conv3
    10 block3_pool
    11 block4_conv1
    12 block4_conv2
    13 block4_conv3
    14 block4_pool
    15 block5_conv1
    16 block5_conv2
    17 block5_conv3
    18 block5_pool
    19 flatten_2
    20 dense_3
    21 dropout_2
    22 dense_4
    


```python
for layer in new_model.layers[:18]:
    layer.trainable = False
for layer in new_model.layers[18:]:
    layer.trainable = True
```


```python
print_layer_trainable()
```

    False:	input_2
    False:	block1_conv1
    False:	block1_conv2
    False:	block1_pool
    False:	block2_conv1
    False:	block2_conv2
    False:	block2_pool
    False:	block3_conv1
    False:	block3_conv2
    False:	block3_conv3
    False:	block3_pool
    False:	block4_conv1
    False:	block4_conv2
    False:	block4_conv3
    False:	block4_pool
    False:	block5_conv1
    False:	block5_conv2
    False:	block5_conv3
    True:	block5_pool
    True:	flatten_2
    True:	dense_3
    True:	dropout_2
    True:	dense_4
    


```python
new_model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
```


```python
history = new_model.fit_generator(
        train_generator,
        steps_per_epoch=2000 // batch_size,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=800 // batch_size)
```

    Epoch 1/50
    125/125 [==============================] - 49s 391ms/step - loss: 2.2662 - acc: 0.1765 - val_loss: 1.9555 - val_acc: 0.2978
    Epoch 2/50
    125/125 [==============================] - 43s 348ms/step - loss: 1.8748 - acc: 0.2395 - val_loss: 1.6836 - val_acc: 0.4144
    Epoch 3/50
    125/125 [==============================] - 42s 337ms/step - loss: 1.6959 - acc: 0.3216 - val_loss: 1.4831 - val_acc: 0.4722
    Epoch 4/50
    125/125 [==============================] - 43s 348ms/step - loss: 1.5186 - acc: 0.4015 - val_loss: 1.2469 - val_acc: 0.5578
    Epoch 5/50
    125/125 [==============================] - 44s 353ms/step - loss: 1.2423 - acc: 0.5065 - val_loss: 1.0319 - val_acc: 0.7711
    Epoch 6/50
    125/125 [==============================] - 43s 343ms/step - loss: 1.0721 - acc: 0.5712 - val_loss: 0.8763 - val_acc: 0.7200
    Epoch 7/50
    125/125 [==============================] - 43s 345ms/step - loss: 0.9445 - acc: 0.6178 - val_loss: 0.6730 - val_acc: 0.7922
    Epoch 8/50
    125/125 [==============================] - 43s 342ms/step - loss: 0.8673 - acc: 0.6464 - val_loss: 0.5820 - val_acc: 0.8667
    Epoch 9/50
    125/125 [==============================] - 43s 346ms/step - loss: 0.8263 - acc: 0.6709 - val_loss: 0.5039 - val_acc: 0.8589
    Epoch 10/50
    125/125 [==============================] - 42s 336ms/step - loss: 0.7714 - acc: 0.6927 - val_loss: 0.5687 - val_acc: 0.9133
    Epoch 11/50
    125/125 [==============================] - 43s 344ms/step - loss: 0.7339 - acc: 0.7027 - val_loss: 0.4528 - val_acc: 0.8756
    Epoch 12/50
    125/125 [==============================] - 43s 346ms/step - loss: 0.7018 - acc: 0.7331 - val_loss: 0.4825 - val_acc: 0.8922
    Epoch 13/50
    125/125 [==============================] - 43s 343ms/step - loss: 0.7131 - acc: 0.7278 - val_loss: 0.4079 - val_acc: 0.8878
    Epoch 14/50
    125/125 [==============================] - 42s 335ms/step - loss: 0.6805 - acc: 0.7386 - val_loss: 0.3441 - val_acc: 0.9278
    Epoch 15/50
    125/125 [==============================] - 43s 345ms/step - loss: 0.6588 - acc: 0.7418 - val_loss: 0.3927 - val_acc: 0.9033
    Epoch 16/50
    125/125 [==============================] - 42s 334ms/step - loss: 0.6528 - acc: 0.7466 - val_loss: 0.3433 - val_acc: 0.9100
    Epoch 17/50
    125/125 [==============================] - 43s 344ms/step - loss: 0.6405 - acc: 0.7541 - val_loss: 0.3574 - val_acc: 0.8756
    Epoch 18/50
    125/125 [==============================] - 42s 332ms/step - loss: 0.6267 - acc: 0.7567 - val_loss: 0.5040 - val_acc: 0.8456
    Epoch 19/50
    125/125 [==============================] - 44s 354ms/step - loss: 0.6169 - acc: 0.7621 - val_loss: 0.4060 - val_acc: 0.8556
    Epoch 20/50
    125/125 [==============================] - 42s 338ms/step - loss: 0.5923 - acc: 0.7749 - val_loss: 0.4408 - val_acc: 0.8989
    Epoch 21/50
    125/125 [==============================] - 43s 346ms/step - loss: 0.6054 - acc: 0.7632 - val_loss: 0.3231 - val_acc: 0.9033
    Epoch 22/50
    125/125 [==============================] - 44s 350ms/step - loss: 0.5647 - acc: 0.7838 - val_loss: 0.3578 - val_acc: 0.8911
    Epoch 23/50
    125/125 [==============================] - 42s 338ms/step - loss: 0.5438 - acc: 0.7911 - val_loss: 0.1968 - val_acc: 0.9500
    Epoch 24/50
    125/125 [==============================] - 43s 347ms/step - loss: 0.5381 - acc: 0.7887 - val_loss: 0.2241 - val_acc: 0.9411
    Epoch 25/50
    125/125 [==============================] - 42s 337ms/step - loss: 0.5093 - acc: 0.7991 - val_loss: 0.2531 - val_acc: 0.9278
    Epoch 26/50
    125/125 [==============================] - 43s 346ms/step - loss: 0.4970 - acc: 0.8086 - val_loss: 0.2138 - val_acc: 0.9567
    Epoch 27/50
    125/125 [==============================] - 43s 346ms/step - loss: 0.4922 - acc: 0.8084 - val_loss: 0.3187 - val_acc: 0.9200
    Epoch 28/50
    125/125 [==============================] - 43s 345ms/step - loss: 0.4746 - acc: 0.8239 - val_loss: 0.4983 - val_acc: 0.8711
    Epoch 29/50
    125/125 [==============================] - 42s 339ms/step - loss: 0.4408 - acc: 0.8281 - val_loss: 0.4270 - val_acc: 0.8800
    Epoch 30/50
    125/125 [==============================] - 43s 345ms/step - loss: 0.4710 - acc: 0.8202 - val_loss: 0.2943 - val_acc: 0.9089
    Epoch 31/50
    125/125 [==============================] - 42s 337ms/step - loss: 0.4598 - acc: 0.8168 - val_loss: 0.3496 - val_acc: 0.8989
    Epoch 32/50
    125/125 [==============================] - 43s 345ms/step - loss: 0.4529 - acc: 0.8266 - val_loss: 0.2500 - val_acc: 0.9400
    Epoch 33/50
    125/125 [==============================] - 42s 337ms/step - loss: 0.4361 - acc: 0.8348 - val_loss: 0.2828 - val_acc: 0.9356
    Epoch 34/50
    125/125 [==============================] - 44s 355ms/step - loss: 0.4467 - acc: 0.8282 - val_loss: 0.3925 - val_acc: 0.9078
    Epoch 35/50
    125/125 [==============================] - 42s 337ms/step - loss: 0.4598 - acc: 0.8197 - val_loss: 0.5981 - val_acc: 0.8478
    Epoch 36/50
    125/125 [==============================] - 44s 351ms/step - loss: 0.4366 - acc: 0.8323 - val_loss: 0.2859 - val_acc: 0.9333
    Epoch 37/50
    125/125 [==============================] - 43s 344ms/step - loss: 0.4636 - acc: 0.8219 - val_loss: 0.3378 - val_acc: 0.9167
    Epoch 38/50
    125/125 [==============================] - 42s 339ms/step - loss: 0.4873 - acc: 0.8204 - val_loss: 0.3408 - val_acc: 0.9122
    Epoch 39/50
    125/125 [==============================] - 43s 347ms/step - loss: 0.4422 - acc: 0.8336 - val_loss: 0.4412 - val_acc: 0.9133
    Epoch 40/50
    125/125 [==============================] - 42s 337ms/step - loss: 0.4416 - acc: 0.8290 - val_loss: 0.3737 - val_acc: 0.9078
    Epoch 41/50
    125/125 [==============================] - 44s 349ms/step - loss: 0.4314 - acc: 0.8337 - val_loss: 0.4530 - val_acc: 0.8633
    Epoch 42/50
    125/125 [==============================] - 42s 337ms/step - loss: 0.4149 - acc: 0.8423 - val_loss: 0.3574 - val_acc: 0.9011
    Epoch 43/50
    125/125 [==============================] - 44s 353ms/step - loss: 0.4534 - acc: 0.8290 - val_loss: 0.3258 - val_acc: 0.9178
    Epoch 44/50
    125/125 [==============================] - 42s 336ms/step - loss: 0.4423 - acc: 0.8307 - val_loss: 0.3778 - val_acc: 0.9067
    Epoch 45/50
    125/125 [==============================] - 43s 347ms/step - loss: 0.4176 - acc: 0.8374 - val_loss: 0.3783 - val_acc: 0.8933
    Epoch 46/50
    125/125 [==============================] - 42s 338ms/step - loss: 0.4086 - acc: 0.8394 - val_loss: 0.3790 - val_acc: 0.9022
    Epoch 47/50
    125/125 [==============================] - 43s 347ms/step - loss: 0.4120 - acc: 0.8376 - val_loss: 0.5045 - val_acc: 0.8678
    Epoch 48/50
    125/125 [==============================] - 44s 348ms/step - loss: 0.4341 - acc: 0.8314 - val_loss: 0.2613 - val_acc: 0.9144
    Epoch 49/50
    125/125 [==============================] - 43s 346ms/step - loss: 0.4405 - acc: 0.8280 - val_loss: 0.5647 - val_acc: 0.8856
    Epoch 50/50
    125/125 [==============================] - 42s 338ms/step - loss: 0.4096 - acc: 0.8408 - val_loss: 0.4827 - val_acc: 0.9089
    


```python
new_model.save('transfer_model.h5')
```


```python
from keras.models import load_model
new_model = load_model('transfer_model.h5')
```

    WARNING:tensorflow:From c:\users\user\appdata\local\programs\python\python36\lib\site-packages\tensorflow\python\ops\math_ops.py:3066: to_int32 (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use tf.cast instead.
    

# Visualization


```python
test_files = validation_generator.filenames
```


```python
def get_test_batch(root_path,img_paths):
    return np.array([cv2.resize(cv2.imread(os.path.join(root_path,t)),(200,200)) for t in img_paths]), np.array([l.split("\\")[0] for l in img_paths])
```


```python
batch, labels = get_test_batch(TRAIN_DIR, test_files)
```


```python
preds = new_model.predict_on_batch(batch)
```


```python
plt.figure(figsize=(15,100))
count = 1
for img,orig_img in zip(preds,batch):
    plt.subplot(batch.shape[0],2,count)
    plt.imshow(orig_img)
    plt.xlabel('Original class: {},\n Predicted class: {}, \n Confidence: {:.4f}'.format(
        labels[count-1],
        np.argmax(img)+1,
        img[np.argmax(img)]), fontsize=20)
    plt.tight_layout()
    
    count+=1
```


![png](am_54_0.png)

