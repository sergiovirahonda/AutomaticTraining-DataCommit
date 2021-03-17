import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
import argparse
import data_utils
import data_utils, email_notifications
import sys
import os
from google.cloud import storage
import datetime

# general variables declaration
# ------------------------------------------------------------------------------------------------------------------------------------
model_name = 'best_model.hdf5'
# ------------------------------------------------------------------------------------------------------------------------------------

def initialize_gpu():

    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    if len(tf.config.experimental.list_physical_devices('GPU')) > 0:
        tf.config.set_soft_device_placement(True)
        tf.debugging.set_log_device_placement(True)
    
    return

def start_training(args):

    # Loading splitted data
    X_train, X_test, y_train, y_test = data_utils.load_data(args)

    # Initializing GPU if available (if available)
    initialize_gpu()

    # Checking if there's any model saved at testing or production folders in GCS
    model_gcs_prod = data_utils.previous_model(args.bucket_name,'production',model_name)
    model_gcs_test = data_utils.previous_model(args.bucket_name,'testing',model_name)

    # If any model exists at production, load it, test it on data and if it doesn't reach good metric then retrain it and save it to testing folder
    if model_gcs_prod[0] == True:
        train_prod_model(X_train, X_test, y_train, y_test,args)
    if model_gcs_prod[0] == False:
        if model_gcs_test[0] == True:
            train_test_model(X_train, X_test, y_train, y_test,args)
        if model_gcs_test[0] == False:
            print('No previous models were found at GCS. Exiting automatic training and emailing the owner.',flush=True)
            email_notifications.training_result('not_found',' ')
            sys.exit(1)
        if model_gcs_test[0] == None:
            print('Something went wrong when trying to check if old testing model exists. Exception: '+str(model_gcs_test[1])+'. Aborting automatic training.',flush=True)
            email_notifications.exception('Something went wrong when trying to check if old testing model exists. Exception: '+model_gcs_test[1]+'. Aborting automatic training.')
            sys.exit(1) 
    if model_gcs_prod[0] == None:
        print('Something went wrong when trying to check if old production model exists. Exception: '+str(model_gcs_prod[1])+'. Aborting automatic training.',flush=True)
        email_notifications.exception('Something went wrong when trying to check if old production model exists. Exception: '+model_gcs_prod[1]+'. Aborting automatic training.')
        sys.exit(1) 


def train_prod_model(X_train, X_test, y_train, y_test,args):
    model_gcs_prod = data_utils.load_model(args.bucket_name,'production',model_name) #done
    if model_gcs_prod[0] == True:
        try:
            print('Loading old model stored at GCS/production and performing evaluation',flush=True)
            cnn = load_model(model_name)
            print('Starting model evaluation.',flush=True)
            model_loss, model_acc = cnn.evaluate(X_test, y_test,verbose=2)
            if model_acc > 0.90:
                saved_ok = data_utils.save_model(args.bucket_name,model_name)
                if saved_ok[0] == True:
                    print("Old model from production has reached more than 0.90, it's been saved into to GCS/testing and email the owner.",flush=True)
                    email_notifications.training_result('old_evaluation_prod',model_acc) #done
                    sys.exit(0) 
                else:
                    print("Old model from production has reached more than 0.90 after, but something went wrong when trying to save it onto GCP. Check the logs for more info. Exception: "+str(saved_ok[1]),flush=True)
                    email_notifications.exception(saved_ok[1])
                    sys.exit(1) 
            else:
                print("Old model from production hasn't reached more than 0.90 of accuracy, proceeding to retrain.",flush=True)
                cnn = load_model(model_name)
                cnn.fit(X_train, y_train, epochs=args.epochs,validation_data=(X_test, y_test),callbacks=[checkpoint])
                print('Old production model training has ended. ',flush=True)
                model_loss, model_acc = cnn.evaluate(X_test, y_test,verbose=2)
                if model_acc > 0.90:
                    saved_ok = data_utils.save_model(args.bucket_name,model_name)
                    if saved_ok[0] == True:
                        print("Old model from production has reached more than 0.90 after re-training, it's been saved into to GCS/testing and email the owner.",flush=True)
                        email_notifications.training_result('retrain_prod',model_acc) #done
                        sys.exit(0) 
                    else:
                        print("Old model from production has reached more than 0.90 after re-training, but something went wrong when trying to save it onto GCP. Check the logs for more info. Exception: "+str(saved_ok[1]),flush=True)
                        email_notifications.exception(saved_ok[1])
                        sys.exit(1) 
                else:
                    print("Old model from production hasn't reached more than 0.90 after re-training, proceeding to check if any model exists at /testing.",flush=True)
                    return
        except Exception as e:
            print('Something went wrong when trying to retrain old production model. Exception: '+str(status)+'. Aborting automatic training.',flush=True)
            email_notifications.exception('Something went wrong when trying to retrain old production model. Exception: '+str(status))
            sys.exit(1) 
    else:
        email_notifications.exception('Something went wrong when trying to load old production model. Exception: '+str(model_gcs_prod[1]))
        print('Something went wrong when trying to load old production model. Exception: '+str(model_gcs_prod[1])+'. Aborting automatic training.',flush=True)
        sys.exit(1)


def train_test_model(X_train, X_test, y_train, y_test,args):
    model_gcs_test = data_utils.load_model(args.bucket_name,'testing',model_name)
    if model_gcs_test[0] == True:
        try:
            print('Loading old model stored at GCS/testing and performing evaluation',flush=True)
            cnn = load_model(model_name)
            print('Starting model evaluation.',flush=True)
            model_loss, model_acc = cnn.evaluate(X_test, y_test,verbose=2)
            if model_acc > 0.90: # Nothing to do, keep the model the way it is.
                print("Old model from testing has reached more than 0.90 of accuracy after evaluation, there's no need to be retrained. Emailing the owner and exiting model testing execution.",flush=True)
                email_notifications.training_result('old_evaluation_test',model_acc)
                sys.exit(0) 
            else:
                print("Old model from testing hasn't reached more than 0.90 of accuracy, proceeding to retrain.",flush=True)
                cnn = load_model(model_name)
                cnn.fit(X_train, y_train, epochs=args.epochs,validation_data=(X_test, y_test),callbacks=[checkpoint])
                print('Old testing model training has ended. ',flush=True)
                model_loss, model_acc = cnn.evaluate(X_test, y_test,verbose=2)
                if model_acc > 0.90:
                    saved_ok = data_utils.save_model(args.bucket_name,model_name)
                    if saved_ok[0] == True:
                        print("Old model from testing has reached more than 0.90 after re-training, it's been saved into to GCS/testing and email the owner.",flush=True)
                        email_notifications.training_result('retrain_test',model_acc) 
                        sys.exit(0) 
                    else:
                        print("Old model from testing has reached more than 0.90 after re-training, but something went wrong when trying to save it onto GCP. Check the logs for more info. Exception: "+str(saved_ok[1]),flush=True)
                        email_notifications.exception(saved_ok[1])
                        sys.exit(1)
                else:
                    print("None of the models have reached more than 0.90 of accuracy during retraining. Emailing the owner and exiting automatic training with code status of 1.",flush=True)
                    email_notifications.training_result('poor_metrics',model_acc) 
                    sys.exit(1)
        except Exception as e:
            print('Something went wrong when trying to retrain old testing model. Exception: '+str(status)+'. Aborting automatic training.',flush=True)
            email_notifications.exception('Something went wrong when trying to retrain old testing model. Exception: '+str(status))
            sys.exit(1) 
    else:
        email_notifications.exception('Something went wrong when trying to load old testing model. Exception: '+str(model_gcs_test[1]))
        print('Something went wrong when trying to load old testing model. Exception: '+str(model_gcs_test[1])+'. Aborting automatic training.',flush=True)
        sys.exit(1)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bucket-name',
                        type=str,
                        default='automatictrainingcicd-aiplatform',
                        help='GCP bucket name')
    parser.add_argument('--epochs',
                        type=int,
                        default=2,
                        help='Epochs number')
    args = parser.parse_args()
    return args


def main():

    args = get_args()
    start_training(args)

if __name__ == '__main__':
    main()