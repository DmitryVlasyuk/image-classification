#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
import pickle
import os
import sys
import cv2


# In[2]:


model = load_model(os.path.join('models','food_products.h5'))


# In[3]:


class_names = pickle.loads(open('models/labels.pickle', "rb").read())


# In[4]:


img_path = sys.argv[1]


# In[5]:


img = tf.keras.utils.load_img(img_path, target_size=(100,100))
img = tf.keras.utils.img_to_array(img)
img = tf.expand_dims(img, 0)
predictions = model.predict(img)
score = tf.nn.softmax(predictions[0])


# In[6]:


print(
    "This image most likely belongs to {}"
    .format(class_names[np.argmax(score)])
)


# In[7]:


file_name = os.path.basename(img_path).split('.')[0]
file_extension = os.path.splitext(img_path)[1][1:]

file_path='kb/{}.scs'.format(file_name)

f = open(file_path, "w")

txt="{}\n<- concept_image;\n<= nrel_includes: concept_{};\n<- rrel_key_sc_element: ...\n(*\n<-sc_illustration;;\n<=nrel_sc_text_translation: ...\n(*\n-> rrel_example: \"{}\" (*\n=> nrel_format: format_{};;\n *);;\n*);;\n*);\n<- sc_node_not_relation;;".format(file_name,class_names[np.argmax(score)],img_path,file_extension)


# In[8]:


f.write(txt)
print('file {}.scs created'.format(file_name))
f.close()

