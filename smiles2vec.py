from dl_util import *
from ml_util import *

embedding_vector_length, max_length, vocab_size = 32, 83, 23


def CNN_GRU_model(X,Y, dropout=0, epochs=10, batch_size=32):
    """
    Module that creates a 1-D CNN followed by GRU for property prediction from SMILES
    """
    model = Sequential()
    model.add(Embedding(max_length, embedding_vector_length, input_length=max_length))
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(GRU(100, dropout=dropout))
    model.add(Dense(1))

    model.compile(loss='mse',optimizer='adam',metrics=['mape'])
    model.summary()
    history = model.fit(X, Y, validation_split=0.1, epochs=epochs, batch_size=batch_size, shuffle=True, verbose=1)
    save_model("cnn_gru", model, history, dropout, epochs, batch_size)
    return model



def CNN_LSTM_model(X,Y, dropout=0, epochs=10, batch_size=32):
    """
    Module that creates a 1-D CNN followed by LSTM for property prediction from SMILES
    """
    model = Sequential()
    model.add(Embedding(max_length, embedding_vector_length, input_length=max_length))
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(100, dropout=dropout))
    model.add(Dense(1))

    model.compile(loss='mse',optimizer='adam',metrics=['mape'])
    model.summary()
    print(X[-1], Y[-1])
    print(X[-100], Y[-100])
    history = model.fit(X, Y, validation_split=0.1, epochs=epochs, batch_size=batch_size, shuffle=True, verbose=1)
    save_model("cnn_lstm", model, history, dropout, epochs, batch_size)
    return model
    # model = Sequential()
    # model.add(Embedding(max_length, embedding_vector_length, input_length=max_length))
    # model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    # model.add(MaxPooling1D(pool_size=2))
    # if gate == "lstm":
    #     model.add(LSTM(num_gated_connections, dropout=dropout))
    # else:#gate="gru"
    #     model.add(GRU(num_gated_connections, dropout=dropout))
    # model.add(Dense(1))
    #
    # if optimizer == "adam" and lr!= 0.001:
    #     print("Setting learning rate to"+str(lr))
    #     optimizer = tf.train.AdamOptimizer(lr)
    #
    # model.compile(loss='mse',optimizer=optimizer,metrics=['mape'])
    # model.summary()
    # return model
    #history = model.fit(X1_train, y_train, shuffle=True, validation_split=0.2, \
    #epochs=epochs, batch_size=batch_size, verbose=1, callbacks=[early_stop])




def GRU2_model(X,Y, dropout=0, epochs=10, batch_size=32):
    """
    Module that creates a 2 layer GRU for property prediction from SMILES
    """
    model = Sequential()
    model.add(Embedding(max_length, embedding_vector_length, input_length=max_length))

    model.add(GRU(100, return_sequences=True, dropout=dropout))
    model.add(GRU(100, dropout=dropout))

    model.add(Dense(1))

    model.compile(loss='mse',optimizer='adam',metrics=['mape'])
    model.summary()
    history = model.fit(X, Y, validation_split=0.1, epochs=epochs, batch_size=batch_size, shuffle=True, verbose=1)
    save_model("gru_2layer", model, history, dropout, epochs, batch_size)
    return model



def LSTM2_model(X,Y, dropout=0, epochs=10, batch_size=32):
    """
    Module that creates a 2 layer LSTM for property prediction from SMILES
    """
    model = Sequential()
    model.add(Embedding(max_length, embedding_vector_length, input_length=max_length))

    model.add(LSTM(100, return_sequences=True, dropout=dropout))
    model.add(LSTM(100, dropout=dropout))

    model.add(Dense(1))

    model.compile(loss='mse',optimizer='adam',metrics=['mape'])
    model.summary()
    history = model.fit(X, Y, validation_split=0.1, epochs=epochs, batch_size=batch_size, shuffle=True, verbose=1)
    save_model("lstm_2layer", model, history, dropout, epochs, batch_size)
    return model

def GRU_model(X,Y, dropout=0, epochs=10, batch_size=32):

    """
    Module that creates a 1 layer GRU for property prediction from SMILES
    """
    model = Sequential()
    model.add(Embedding(max_length, embedding_vector_length, input_length=max_length))

    model.add(GRU(100, dropout=dropout))

    model.add(Dense(1))

    model.compile(loss='mse',optimizer='adam',metrics=['mape'])
    model.summary()
    history = model.fit(X, Y, validation_split=0.1, epochs=epochs, batch_size=batch_size, shuffle=True, verbose=1)
    save_model("gru_1layer", model, history, dropout, epochs, batch_size)
    return model


def LSTM_model(X,Y, dropout=0, epochs=10, batch_size=32):
    """
    Module that creates a 1 layer LSTM for property prediction from SMILES
    """
    model = Sequential()
    model.add(Embedding(max_length, embedding_vector_length, input_length=max_length))

    model.add(LSTM(100, dropout=dropout))

    model.add(Dense(1))

    model.compile(loss='mse',optimizer='adam',metrics=['mape'])
    model.summary()
    history = model.fit(X, Y, validation_split=0.1, epochs=epochs, batch_size=batch_size, shuffle=True, verbose=1)
    save_model("lstm_1layer", model, history, dropout, epochs, batch_size)
    return model

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("-m", "--model", help= "which model to train", required=True)
    parser.add_argument("-d", "--dropout", help= "amount of dropout", required=False)
    parser.add_argument("-s", "--size", help= "percentage of dataset to use", required=False)
    parser.add_argument("-l", "--layers", help= "number of layers", required=False)
    parser.add_argument("-e", "--epochs", help= "epochs", required=False)
    parser.add_argument("-b", "--batch_size", help= "size of batch", required=False)
    #parser.add_argument("-r", "--learning_rate", help= "learning rate", required=False)
    args = parser.parse_args()

    if args.dropout:
        dropout = float(args.dropout)
    else:
        dropout =0

    if args.epochs:
        epochs = int(args.epochs)
    else:
        epochs = 10

    if args.batch_size:
        batch_size = int(args.batch_size)
    else:
        batch_size = 32

    # if args.learning_rate:
    #     learning_rate = float(args.learning_rate)
    # else:
    #     learning_rate = 0.001
    start = time.time()
    #SMILES = loadData("SMILES_1_7")
    smiles_sequences = loadNumpy('SMILES_1_7_sequences')
    HOMO = loadNumpy('HOMO_1_7')
    print("Loaded Data...")


    if args.size:
        size = float(args.size)
        num_records = int(size*len(HOMO))
        X = smiles_sequences[:num_records]
        Y = HOMO[:num_records]

    else:
        size = 1
        X = smiles_sequences
        Y = HOMO

    # print(X[0],Y[0])
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, random_state=1024)
    # print(X_train[0],Y_train[0])

    ## Assigning the architecture based on arguments
    model_type = args.model
    if model_type =="lstm":
        layers = args.layers

        if not layers or int(layers) == 2:
            model = LSTM2_model(X_train, Y_train, dropout, epochs, batch_size)
            model_type = "lstm_2layer"
        else:##layers = 1
            model = LSTM_model(X_train, Y_train, dropout, epochs, batch_size)
            model_type = "lstm_1layer"

    elif model_type =="gru":
        layers = args.layers

        if not layers or int(layers) == 2:
            model = GRU2_model(X_train, Y_train, dropout, epochs, batch_size)
            model_type = "gru_2layer"
        else:
            model = GRU_model(X_train, Y_train, dropout, epochs, batch_size)
            model_type = "gru_1layer"

    elif model_type =="cnn_lstm":
        model = CNN_LSTM_model(X_train, Y_train, dropout, epochs, batch_size)
    elif model_type =="cnn_gru":
        model = CNN_GRU_model(X_train, Y_train, dropout, epochs, batch_size)

    else:
        print("MODEL TYPE UNDEFINED")
        exit(4)

    [loss, mape] = model.evaluate(X_test, Y_test, verbose=0)
    Y_predict = model.predict(X_test)
    r2 = r2_score(Y_test,Y_predict)
    mean_squared_err = mse(Y_test,Y_predict)
    mean_absolute_err = mae(Y_test, Y_predict)

    print("Testing set Mean Abs percentage Error: {:2.4f}".format(mape ))
    print("Testing set Mean Abs Error: {:2.4f}".format(mean_absolute_err))
    print("Testing set Mean R2: {:2.4f}".format(r2 ))
    print("Testing set Mean Squared Error: {:2.4f}".format(mean_squared_err))
    stats = {"mape":mape, "mae":mean_absolute_err, "mse":mean_squared_err, "r2":r2}

    file_suffix = "_"+model_type+"_dropout_"+str(dropout)+"_epochs_"+str(epochs)+"_batch_"+str(batch_size)
    stats_file = "stats"+file_suffix

    saveData(stats,stats_file,"model")
    print("Stats saved in", stats_file+".pkl")

    time_elapsed = str(time.time()-start)
    subject = "smiles2vec_"+model_type+"_dropout_"+str(dropout)+"_epochs_"+str(epochs)+"_"+str(batch_size)
    message = prepare_message(model_type, stats, dropout, epochs, batch_size, time_elapsed, size)
    try:
        send_email(subject, message)
    except:
        print("Unable to send email")
