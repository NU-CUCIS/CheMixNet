from __future__ import print_function
from dl_util import *
from ml_util import *
import pickle
import keras
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Input, GlobalMaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
import json
import time

def Inception0(input):
    tower_1 = Conv2D(16, (1, 1), padding='same', activation='relu')(input)
    tower_1 = Conv2D(16, (3, 3), padding='same', activation='relu')(tower_1)

    tower_2 = Conv2D(16, (1, 1), padding='same', activation='relu')(input)
    tower_2 = Conv2D(16, (5, 5), padding='same', activation='relu')(tower_2)

    tower_3 = Conv2D(16, (1, 1), padding='same', activation='relu')(input)

    output = keras.layers.concatenate([tower_1, tower_2, tower_3], axis=-1)
    return output


def Inception(input):
    tower_1 = Conv2D(16, (1, 1), padding='same', activation='relu')(input)
    tower_1 = Conv2D(16, (3, 3), padding='same', activation='relu')(tower_1)

    tower_2 = Conv2D(16, (1, 1), padding='same', activation='relu')(input)
    tower_2 = Conv2D(16, (5, 5), padding='same', activation='relu')(tower_2)

    tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(input)
    tower_3 = Conv2D(16, (1, 1), padding='same', activation='relu')(tower_3)

    output = keras.layers.concatenate([tower_1, tower_2, tower_3], axis=-1)
    return output


generator = ImageDataGenerator(rotation_range=180,
                               width_shift_range=0.1,height_shift_range=0.1,
                               fill_mode="constant",cval = 0,
                               horizontal_flip=True, vertical_flip=True,data_format='channels_last',)


if __name__ == "__main__":

    print("Code Version 3.9 - fixed classification ")
    print()
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--dataset", help= "dataset to use", required=True)
    parser.add_argument("-p", "--processor", help= "which processor to run- required in server", required=False)
    parser.add_argument("-e", "--epochs", help= "epochs(default 30)", required=False)
    parser.add_argument("-b", "--batch_size", help= "batch_size(default 128)", required=False)
    parser.add_argument("-l", "--learning_rate", help= "initial learning_rate", required=False)
    parser.add_argument("-c", "--concatenation_factor", help= "degree of concatenation", required=False)
    #parser.add_argument("-t", "--task", help= "type of ML task (classification or regression)", required=True)

    args = parser.parse_args()
    if platform.system() == 'Linux' and "prismatic" not in platform.node():
        if not args.processor:
            print("++++++++++++++++++++++++++")
            print("No GPU mentioned on multi-GPU server")
            print("++++++++++++++++++++++++++")
        else:
            processor = args.processor
            print("++++++++++++++++++++++++++")
            print("Setting GPU to ", processor)
            os.environ["CUDA_VISIBLE_DEVICES"]=processor
            print("++++++++++++++++++++++++++")

    dataset = args.dataset
    if "tox" in dataset or "hiv" in dataset:
        task = "classification"
    else:
        task = "regression"
    # task = args.task
    print("++++++++++++++++++++++++++")
    print("Dataset loaded is", dataset)
    print('Task is', task)
    print("++++++++++++++++++++++++++")

    if args.concatenation_factor:
        concat = int(args.concatenation_factor)
    else:
        concat = 50

    if args.epochs:
        epochs = int(args.epochs)
    else:
        epochs = 30

    if args.batch_size:
        batch_size = int(args.batch_size)
    else:
        batch_size = 128

    if args.learning_rate:
        learning_rate = float(args.learning_rate)
    else:
        learning_rate = 0.001
    print("============================")
    print("Number of epochs is", epochs)
    print("Batch size is", batch_size)
    print("Learning rate is", learning_rate)
    print("============================")

    X = loadNumpy(dataset+'_x')
    y = loadData(dataset+'_y')

    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=1024)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=1024)

    from sklearn.preprocessing import RobustScaler
    rbs = RobustScaler(with_centering=True, with_scaling=True, quantile_range=(5.0, 95.0), copy=True)
    if task == "regression":
        y_train_s = rbs.fit_transform(y_train.reshape(-1,1))
        y_val_s = rbs.transform(y_val.reshape(-1,1))
        y_test_s = rbs.transform(y_test.reshape(-1,1))
    else:

        y_train_s, y_val_s, y_test_s = y_train, y_val, y_test

    input_shape = X_train.shape[1:]
    input_img = Input(shape=input_shape)

    x = Inception0(input_img)
    x = Inception(x)
    x = Inception(x)
    od=int(x.shape[1])
    x = MaxPooling2D(pool_size=(od,od), strides=(1,1))(x)
    x = Flatten()(x)
    x = Dense(100, activation='relu')(x)
    if task == "regression":
        output = Dense(1, activation='linear')(x)
    else:
        output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=input_img, outputs=output)
    #Concatenate for longer epochs
    Xt = np.concatenate([X_train]*concat, axis=0)
    yt = np.concatenate([y_train_s]*concat, axis=0)

    g = generator.flow(Xt, yt, batch_size=batch_size, shuffle=True)
    steps_per_epoch = 10000/batch_size

    optimizer = Adam(lr=learning_rate)
    model.summary()
    if task == "regression":
        model.compile(loss="mse", optimizer=optimizer)
    else:
        model.compile(loss='binary_crossentropy',optimizer=optimizer)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,patience=10, min_lr=1e-6, verbose=1)

    start = time.time()
    history = model.fit_generator(g,
                                  steps_per_epoch=len(Xt)//batch_size,
                                  epochs=epochs,
                                  validation_data=(X_val,y_val_s),
                                  callbacks=[reduce_lr])
    stop = time.time()
    time_elapsed = stop - start

    name = "chemception_"+dataset+"_epochs_"+str(epochs)+"_batch_"+str(batch_size)+"_learning_rate_"+str(learning_rate)
    model.save("%s.h5"%name)
    hist = history.history
    pickle.dump(hist, file("%s_history.pickle"%name,"w"))
    print("########################")
    print("model and history saved",name)
    print("########################")
    y_predict = model.predict(X_test)
    if task == "regression":
        r2 = r2_score(y_test_s,y_predict)
        mean_squared_err = mse(y_test_s,y_predict)
        mean_absolute_err = mae(y_test_s, y_predict)
        mean_absolute_percent_err = Mape(y_test_s, y_predict)
        stats = {"mape":mean_absolute_percent_err,  "mae":mean_absolute_err, "mse":mean_squared_err, "r2":r2, "time":time_elapsed}
        print("Test MAPE:", mean_absolute_percent_err)
    else:#classification
        accuracy = accuracy_score(y_test_s,y_predict)
        f1 = f1_score(y_test_s,y_predict)
        precision = precision_score(y_test_s,y_predict)
        recall = recall_score(y_test_s,y_predict)
        roc_auc = roc_auc_score(y_test_s,y_predict)
        stats = { "accuracy":accuracy, "precision":precision, "recall":recall, "f1":f1, "auc":roc_auc, "time":time_elapsed}
        print("Test AUC:", auc)
    print(stats)
    saveData(stats,"stats_"+ name, "model")
    print('stats saved',"model/"+name)

    subject=name
    message=json.dumps(stats)
    send_email(subject, message)
