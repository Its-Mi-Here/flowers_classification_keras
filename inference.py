import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os 
#import shutil
from scipy.io import loadmat


from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Activation, MaxPooling2D, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.layers import Flatten, Dense, BatchNormalization, Activation, Dropout
from keras.models import load_model


datagen = ImageDataGenerator(rescale = 1./255)
eval_gen = datagen.flow_from_directory('test/test1', target_size = (200,200), class_mode='binary')


model = load_model("model.h5")
model.load_weights("weights_flowers.hdf5")

score = model.evaluate(eval_gen)

print("Loss is: "+str(score[0]))
print("Accuracy is: "+str(score[1]*100))