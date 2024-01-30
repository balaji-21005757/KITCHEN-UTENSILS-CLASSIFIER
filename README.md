# KITCHEN UTENSIL CLASSIFIER
# Algorithm
1. Import the packages.
2. Read the images.
3. Using classes and epoch find the accuracy and array.
4. Using array we find the name of the name of the utensil.
# Dataset Link
https://www.kaggle.com/datasets/jehanbhathena/utensil-image-recognition?resource=download
## Program:
```
Program to implement 
Developed by   : R.SOMEASVAR, K.BALAJI
RegisterNumber : 212221230103, 212221230011
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

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import utils
from tensorflow.keras import models
model = models.Sequential()
model.add(layers.Input(shape=(28,28,1)))
model.add(layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu',))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu',))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu',))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Flatten())

model.add(layers.Dense(128))
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

batch_size = 16
model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),optimizer=tf.keras.optimizers.Adam(),metrics=["accuracy"])


history = model.fit(train,epochs=5,steps_per_epoch=len(train),validation_data=test,validation_steps=len(test))

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

![image](https://github.com/SOMEASVAR/KITCHEN-UTENSILS-CLASSIFIER/assets/93434149/a2a2b904-64cf-458a-ae94-3c27f71c04f4)






