from ethicnet import config
import tensorflow as tf
import tensorflow.keras
from keras.datasets import cifar10
import pandas as pd
import numpy as np
import os

import matplotlib.image

def load_cifar10_dataset():
    ''' Loads cifar10 train and test dataset '''
    # load dataset
    (trainX, trainY), (testX, testY) = cifar10.load_data()
    # one hot encode target values
    trainY = tf.keras.utils.to_categorical(trainY)
    testY = tf.keras.utils.to_categorical(testY)
    return trainX, trainY, testX, testY

img_augmentation = tensorflow.keras.models.Sequential(
    [
        tensorflow.keras.layers.experimental.preprocessing.RandomRotation(factor=0.15),
        tensorflow.keras.layers.experimental.preprocessing.RandomTranslation(height_factor=0.1, width_factor=0.1),
        tensorflow.keras.layers.experimental.preprocessing.RandomFlip(),
        tensorflow.keras.layers.experimental.preprocessing.RandomContrast(factor=0.1),
    ],
    name="img_augmentation",
)

def scale_pixels(train, test):
    '''Takes Train and Test arrays, returns them divided by 255'''
    # convert from integers to floats
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')
	# normalize to range 0-1
    # train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0
    # return normalized images
    return train_norm, test_norm

def extract_infos_from_images_name(data_dir = config.data_dir):
    '''Returns a dataframe with the information contained in the name of the images''' 
    liste = []
    for k in os.walk(data_dir):
        liste.append(k)
    image_name = liste[0][2]
    df = pd.DataFrame(image_name,columns=['image_name'])
    df['age'] = df.apply(lambda row : row.image_name.split('_')[0] ,axis=1)
    df['sex'] = df.apply(lambda row : row.image_name.split('_')[1] ,axis=1)
    df['ethnicity'] = df.apply(lambda row : row.image_name.split('_')[2] ,axis=1)
    df.age = df.apply(lambda row : int(row.age), axis=1)
    df.replace({'ethnicity':{'0':'white','1':'black','2':'asian','3':'indian','4':'others' }},inplace=True)
    df.replace({'sex':{'0':'male','1':'female'}},inplace=True)
    return df

def correct_undefined_ethnicity(df):
    '''correction des ethinicités manquantes à la main'''
    df_wrong_ethnicity = df.loc[(df.ethnicity > str(2) ) & (df.ethnicity < str(3) )]
    df.loc[df.ethnicity == df_wrong_ethnicity.iloc[0].ethnicity , 'ethnicity'] = 'black'
    df.loc[df.ethnicity == df_wrong_ethnicity.iloc[1].ethnicity , 'ethnicity'] = 'indian'
    df.loc[df.ethnicity == df_wrong_ethnicity.iloc[2].ethnicity , 'ethnicity'] = 'black'   

def categorize_age(row):
    ''' map to categorize age '''
    age = row.age 
    if age < 5 :
        return 'baby'
    elif age < 16 :
        return 'kid'
    elif age < 31 :
        return 'young'
    elif age < 61 :
        return 'adult'
    else :
        return 'old'

def dictionnary_from_dataframe(df):
    '''creates dataframes for each combinaison of features, and gathers them in a dictionnary'''
    list_of_values = lambda label_string : list(df[label_string].value_counts().index)
    d = {}
    for sex in list_of_values('sex'):
        for ethnicity in list_of_values('ethnicity'):
            for age in list_of_values('age'):
                d[sex+'_'+ethnicity+'_'+age]=df.where((df.sex == sex) & (df.ethnicity == ethnicity) & (df.age == age)).dropna()
    return d

def read_image(image_name,path=config.data_dir):
    '''from the name of an image, return the array of the image''' 
    im = matplotlib.image.imread(path+image_name)
    return im.reshape((1,)+im.shape)

def images_of_a_category(sex,ethnicity,age,d):
    ''' returns an array of all the images of people with a certain sex, ethnicity and age'''
    images = []
    for image_name in list(d[sex+'_'+ethnicity+'_'+age]['image_name']):
        images.append(read_image(image_name))  
    return images

def output_of_layer_for_one_image(layer_name,model,image):
    ''' returns the output of a model until a specific layer, from an image'''
    layer_model = tf.keras.models.Model(inputs=model.input,outputs=model.get_layer(layer_name).output)
    output = layer_model.predict(image)
    return output

def output_of_layer_for_multiple_images(layer_name,model,images):
    '''passes the array of images in the model until the specific layer, and return the mean of the output for each kernel'''
    output = output_of_layer_for_one_image(layer_name,model,np.concatenate(images))
    return np.mean(output,axis=0).reshape((1,)+output.shape[1:])

def list_kernels_above_value(output,value):
    '''returns the list of the kernel indexes the mean on which are above some value'''
    return list(np.where(np.mean(output,axis=(1,2))>value)[1])