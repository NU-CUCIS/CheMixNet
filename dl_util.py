import os
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from os.path import join

def warn(*args, **kwargs):
    pass

warnings.filterwarnings("ignore")
warnings.warn = warn

import sys
import platform
import datetime
import h5py
import time
import numpy as np

if platform.system() == 'Linux':
    sys.path.append("/home/apx748/OPV")
else:#It is Mac/Darwin
    sys.path.append("/Users/arindam/OPV")

if platform.python_version().split(".")[0] == "2":
    from keras.layers import Merge

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, GRU, Bidirectional, TimeDistributed, Flatten
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.wrappers.scikit_learn import KerasRegressor
from keras.models import model_from_json
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer

import pickle
import matplotlib.pyplot as plt
from ml_util import str_round, loadData

def as_keras_metric(method):
    import functools
    from keras import backend as K
    import tensorflow as tf
    @functools.wraps(method)
    def wrapper(self, args, **kwargs):
        """ Wrapper for turning tensorflow metrics into keras metrics """
        value, update_op = method(self, args, **kwargs)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([update_op]):
            value = tf.identity(value)
        return value
    return wrapper
precision = as_keras_metric(tf.metrics.precision)
recall = as_keras_metric(tf.metrics.recall)
auc = as_keras_metric(tf.metrics.auc)

def save_history(history,history_file,path):
    obj = {"history":history.history, "epoch":history.epoch}
    with open(path+"/"+history_file + '.pkl', 'wb') as f:
            pickle.dump(obj, f)
    print("History saved successfully", history_file+".pkl")

def save_model(model_type, model, history, dropout, epochs=10, batch_size=32, lr=0.001, prefix=None):
    if prefix is None:
        file_suffix = "_"+model_type+"_dropout_"+str(dropout)\
                                        +"_epochs_"+str(epochs)+"_batch_"+str(batch_size)+"_lr_"+str(lr)
    else:
        file_suffix = "_"+prefix+"_"+model_type+"_dropout_"+str(dropout)\
                                        +"_epochs_"+str(epochs)+"_batch_"+str(batch_size)+"_lr_"+str(lr)
    model_file = join("model","model"+file_suffix+".json")
    weights_file = join("model","weights"+file_suffix+".h5")
    history_file = "history"+file_suffix

    model.save_weights(weights_file)
    # serialize model to JSON
    model_json = model.to_json()
    with open(model_file, "w") as json_file:
        json_file.write(model_json)
    save_history(history,history_file,"model")

    print("Model saved successfully in", model_file)
    print("Weights saved successfully in", weights_file)

def prepare_message(model_type, stats, dropout, epochs, batch_size, lr, time_elapsed, size=1, ml_task = "regression", prefix=None):

    message = "Type of model: "
    message += model_type+"\n\n"
    if ml_task == "classification":
        message += "AUC score: "+str_round(stats["auc"])+"\n"
        message += "Precision: "+str_round(stats["precision"])+"\n"
        message += "Recall: "+str_round(stats["recall"])+"\n"
        message += "Accuracy: "+str_round(stats["accuracy"])+"\n\n"
    else:
        #stats = {"mape":mape, "mae":mean_absolute_err, "mse":mean_squared_err, "r2":r2}
        message += "Mean absolute percent error: "+str_round(stats["mape"])+" %\n"
        message += "Mean absolute error: "+str_round(stats["mae"])+"\n"
        message += "Mean squared error: "+str_round(stats["mse"])+"\n"
        message += "R^2 : "+str_round(stats["r2"])+"\n\n"

    if size==1:
        message+="Whole dataset was used\n\n"
    else:
        message+=str(size*100)+" % of the dataset was used\n\n"

    message+="Training took "+str(time_elapsed)+" seconds\n\n"

    if prefix is None:
        file_suffix = "_"+model_type+"_dropout_"+str(dropout)\
                                        +"_epochs_"+str(epochs)+"_batch_"+str(batch_size)+"_lr"+str(lr)
    else:
        file_suffix = "_"+prefix+"_"+model_type+"_dropout_"+str(dropout)\
                                        +"_epochs_"+str(epochs)+"_batch_"+str(batch_size)+"_lr"+str(lr)

    model_file = join("model","model"+file_suffix+".json")
    weights_file = join("model","weights"+file_suffix+".h5")
    history_file = join("model","history"+file_suffix)
    stats_file = join("model","stats"+file_suffix)

    message+="Model saved successfully in "+ model_file+"\n"
    message+="Weights saved successfully in "+ weights_file+"\n"
    message+="History saved successfully in "+ history_file+".pkl\n"
    message+="Stats saved successfully in "+ stats_file+".pkl\n"

    return message


def generate_onehot_encoding(docs,type="encoded_docs"):

    t = Tokenizer(char_level=True)
    t.fit_on_texts(docs)

    if type=="vocab_size":
        vocab_size = len(t.word_index) + 1
        return vocab_size

    encoded_docs = t.texts_to_sequences(docs)#, mode='count')
    return encoded_docs

def plot_history(history, metric="mean_absolute_percentage_error"):
    plt.figure()
    plt.xlabel('Epoch')
    metric_label = ""
    for item in metric.split("_"):
        metric_label = metric_label + " "+ item.capitalize()

    if "absolute" in metric:
        plt.ylabel(metric_label+" %")
    else:
        plt.ylabel(metric_label)
    if "error" in metric.lower():
        label_train, label_val = "Train Error", "Validation Error"
    elif "auc" in metric.lower():
        label_train, label_val = "Train AUC", "Validation AUC"
    elif "precision" in metric.lower():
        label_train, label_val = "Train Precision", "Validation Precision"
    elif "recall" in metric.lower():
        label_train, label_val = "Train Recall", "Validation Recall"
    else:
        label_train, label_val = "Train Accuracy", "Validation Accuracy"

    plt.plot(history.epoch, np.array(history.history[metric]),
           label=label_train)
    plt.plot(history.epoch, np.array(history.history['val_'+metric]),
           label =label_val)
    plt.legend()
    if "auc" in metric or "acc" in metric or "precision" in metric or "recall" in metric:
        max_lim = 1
    else:
        max_lim = max(max(np.array(history.history[metric])),\
            max(np.array(history.history['val_'+metric])))*2
        if max_lim>100:
            max_lim = 100
    plt.ylim([0, max_lim])

def plot_history_from_file(history_file, metric="mean_absolute_percentage_error"):
    history = loadData(history_file,"model")
    plt.figure()
    plt.xlabel('Epoch')
    metric_label = ""
    for item in metric.split("_"):
        metric_label = metric_label + " "+ item.capitalize()

    if "absolute" in metric:
        plt.ylabel(metric_label+" %")
    else:
        plt.ylabel(metric_label)
    if "error" in metric.lower():
        label_train, label_val = "Train Error", "Validation Error"
    elif "auc" in metric.lower():
        label_train, label_val = "Train AUC", "Validation AUC"
    elif "precision" in metric.lower():
        label_train, label_val = "Train Precision", "Validation Precision"
    elif "recall" in metric.lower():
        label_train, label_val = "Train Recall", "Validation Recall"
    else:
        label_train, label_val = "Train Accuracy", "Validation Accuracy"

    plt.plot(history.epoch, np.array(history["history"][metric]),
           label='Train Error')
    plt.plot(history.epoch, np.array(history["history"]['val_'+metric]),
           label = 'Val Error')
    plt.legend()

    if "auc" in metric or "r2" in metric.lower() or "acc" in metric or "precision" in metric or "recall" in metric:
        max_lim = 1
    else:
        max_lim = max(max(np.array(history["history"][metric])),\
            max(np.array(history["history"]['val_'+metric])))*2
        if max_lim>100:
            max_lim = 100
    plt.ylim([0, max_lim])
