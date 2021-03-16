import datetime
from google.cloud import storage
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from numpy import expand_dims
import gc
from sklearn import preprocessing
import os
import zipfile
import cv2
import sys
print('Importing process completed',flush=True)

def dataset_transformation(path):
    images = []
    counter = 0
    for dirname, _, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith('.png'):
                image = cv2.imread(os.path.join(dirname, filename))
                image = cv2.resize(image, (128, 128))
                images.append(image)
                counter += 1
    for dirname, _, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith('.png'):
                result = data_augmentation(dirname,filename)
                for i in range(len(result)):
                    if i==0:
                        continue
                    else:
                        images.append(result[i])
                        counter += 1
            if counter >= 8000:
                break
        if counter >= 8000:
            break
  
    return images


def data_augmentation(dirname,filename):

    image_data = []
    #reading the image
    image = cv2.imread(os.path.join(dirname, filename))
    image = cv2.resize(image, (128, 128))
    #expanding the image dimension to one sample
    samples = expand_dims(image, 0)
    # creating the image data augmentation generators
    datagen1 = ImageDataGenerator(width_shift_range=[-20,20])
    datagen2 = ImageDataGenerator(zoom_range=[0.8,1.0])
    datagen3 = ImageDataGenerator(brightness_range=[0.5,1.0])
    datagen4 = ImageDataGenerator(rotation_range=20)
    # preparing iterators
    it1 = datagen1.flow(samples, batch_size=1)
    it2 = datagen2.flow(samples, batch_size=1)
    it3 = datagen3.flow(samples, batch_size=1)
    it4 = datagen4.flow(samples, batch_size=1)
    image_data.append(image)
    for i in range(9):
        # generating batch of images
        batch1 = it1.next()
        batch2 = it2.next()
        batch3 = it3.next()
        batch4 = it4.next()
        # convert to unsigned integers
        image1 = batch1[0].astype('uint8')
        image2 = batch2[0].astype('uint8')
        image3 = batch3[0].astype('uint8')
        image4 = batch4[0].astype('uint8')
        #appending to the list of images
        image_data.append(image1)
        image_data.append(image2)
        image_data.append(image3)
        image_data.append(image4)

    return image_data


def load_data(args):
    print("Starting data processing",flush=True)
    
    print('Current directory:', flush=True)
    print(os.getcwdb(), flush=True)
    entries = os.listdir('/root/AutomaticAITrainingWithCICD/COVID_RX/')
    print(entries)
    print('Extracting files', flush=True)

    file_1 = '/root/AutomaticAITrainingWithCICD/COVID_RX/normal_images.zip'
    file_2 = '/root/AutomaticAITrainingWithCICD/COVID_RX/covid_images.zip'
    file_3 = '/root/AutomaticAITrainingWithCICD/COVID_RX/viral_images.zip'
    extract_to = '/root/AutomaticAITrainingWithCICD/COVID_RX/'

    with zipfile.ZipFile(file_1, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    
    with zipfile.ZipFile(file_2, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

    with zipfile.ZipFile(file_3, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

    print('Files extracted', flush=True)
    print('Starting dataset transformation', flush=True)

    normal = dataset_transformation('/root/AutomaticAITrainingWithCICD/COVID_RX/normal_images')[:5000]
    print('First image set created', flush=True)
    covid = dataset_transformation('/root/AutomaticAITrainingWithCICD/COVID_RX/covid_images')[:5000]
    print('Second image set created', flush=True)
    viral = dataset_transformation('/root/AutomaticAITrainingWithCICD/COVID_RX/viral_images')[:5000]
    print('Third image set created', flush=True)

    #Class combination
    print('Combining image sets', flush=True)
    X = normal + viral + covid
    #Transforming from list to numpy array.
    X = np.array(X)

    #Creating labels.
    print('Creating labels', flush=True)
    y = []
    for i in range(5000):
        y.append(0)
    for i in range(5000):
        y.append(1)
    for i in range(5000):
        y.append(2)
    y = np.array(y)

    class_names = ['Normal','Viral Pneumonia','COVID-19']

    #Dataset splitting
    print('Splitting dataset', flush=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=0,shuffle=True)

    #Removing garbage data
    print('Removing gargage data', flush=True)
    del X
    del y
    del normal
    del viral
    del covid
    gc.collect()

    print('Data loading process has ended.', flush=True)
    return X_train, X_test, y_train, y_test

def save_model(bucket_name, best_model):
    try:
        print('Saving model to GCS.',flush=True)
        storage_client = storage.Client() #if running on GCP
        bucket = storage_client.bucket(bucket_name)
        print('Bucket: ',bucket)
        blob1 = bucket.blob('{}/{}'.format('testing',best_model))
        blob1.upload_from_filename(best_model)
        print('Model saved.',flush=True)
        return True,None
    except Exception as e:
        print('Something went wrong when trying to save model on GCS bucket. Exception: ',flush=True)
        return False,e


def previous_model(bucket_name,model_type,model_filename):
    try:
        print('Checking if a previous '+str(model_type)+' model exists at GCS.',flush=True)
        storage_client = storage.Client() #if running on GCP
        bucket = storage_client.bucket(bucket_name)
        status = storage.Blob(bucket=bucket, name='{}/{}'.format(model_type,model_filename)).exists(storage_client)
        print('Model existance: '+str(status))
        return status,None
    except Exception as e:
        print('Something went wrong when trying to check if previous model exists GCS bucket. Exception: ',flush=True)
        return None,e


def load_model(bucket_name,model_type,model_filename):
    print('Loading previous model from GCS bucket. Downloading file.',flush=True)
    try:
        storage_client = storage.Client() #if running on GCP
        bucket = storage_client.bucket(bucket_name)
        blob1 = bucket.blob('{}/{}'.format(model_type,model_filename))
        blob1.download_to_filename('/root/'+str(model_filename))
        print('Previous model downloaded.',flush=True)
        return True,None
    except Exception as e:
        print('Something went wrong when trying to load previous model from GCS bucket. Exception: ',flush=True)
        print(e,flush=True)
        return False,e




    

