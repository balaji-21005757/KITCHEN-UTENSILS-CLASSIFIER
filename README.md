# KITCHEN UTENSIL CLASSIFIER
# Algorithm
1. Import the packages.
2. Read the images.
3. Using classes and epoch find the accuracy and array.
4. Using array we find the name of the name of the utensil.

## Program:
```
Program to implement 
Developed by   : G.SAI DARSHAN, R.SOMEASVAR, K.BALAJI
RegisterNumber : 212221240047, 212221230103, 212221230011
```


```
import splitfolders

splitfolders.ratio("/content/drive/MyDrive/Utensils-final/Raw", output="output", seed=1337, ratio=(.9, .1), group_prefix=None)

import matplotlib.pyplot as plt
import matplotlib.image as mping

img = mping.imread("/content/drive/MyDrive/Utensils-final/Raw/BREAD_KNIFE/breadkniferaw2.JPG")

plt.imshow(img)

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255,
    rotation_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2)

test_datagen = ImageDataGenerator(rescale=1./255)

train = train_datagen.flow_from_directory("output/train/",target_size=(224,224),seed=42,batch_size=32,class_mode="categorical")

test = train_datagen.flow_from_directory("output/val/",target_size=(224,224),seed=42,batch_size=32,class_mode="categorical")

from tensorflow.keras.preprocessing import image

test_image = image.load_img('/content/drive/MyDrive/Utensils-final/Raw/BREAD_KNIFE/breadkniferaw2.JPG', target_size=(224,224))

test_image = image.img_to_array(test_image)

test_image = tf.expand_dims(test_image,axis=0)

test_image = test_image/255.

test_image.shape

import tensorflow_hub as hub

m = tf.keras.Sequential([
hub.KerasLayer("https://tfhub.dev/tensorflow/efficientnet/b0/feature-vector/1"),
tf.keras.layers.Dense(20, activation='softmax')
])

m.compile(loss=tf.keras.losses.CategoricalCrossentropy(),optimizer=tf.keras.optimizers.Adam(),metrics=["accuracy"])

history = m.fit(train,epochs=5,steps_per_epoch=len(train),validation_data=test,validation_steps=len(test))

classes=train.class_indices

classes=list(classes.keys())

m.predict(test_image)

classes[tf.argmax(m.predict(test_image),axis=1).numpy()[0]]

import pandas as pd

pd.DataFrame(history.history).plot()


m.summary()
```
# OUTPUT:
![image](https://github.com/SOMEASVAR/KITCHEN-UTENSILS-CLASSIFIER/assets/93434149/4870d9d5-1d1b-4544-b8dc-989dda128cfd)

![OUTPUT](./1.jpg)
![image](https://github.com/SOMEASVAR/KITCHEN-UTENSILS-CLASSIFIER/assets/93434149/76457ff6-212d-4fe7-b3e6-89f1b0b17d28)

![image](https://github.com/SOMEASVAR/KITCHEN-UTENSILS-CLASSIFIER/assets/93434149/3a7e66b0-1ba1-4b1b-b181-9d63816e1e91)




2. DEMO VIDEO YOUTUBE LINK:
```
https://youtu.be/cQkZ7AeK8sE
```
