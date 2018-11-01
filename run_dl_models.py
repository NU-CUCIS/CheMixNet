from dl_util import *
from ml_util import *
embedding_vector_length = 32
max_length = 83
vocab_size = 23

def bidirectional_model(optimizer="adam", lr=0.001, dropout=0, gate = "lstm", gated_layers= 1, num_gated_connections=100):
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_vector_length, input_length=max_length))
    if gated_layers==2:
        if gate == "lstm":
            model.add(Bidirectional(LSTM(num_gated_connections, return_sequences=True, dropout=dropout)))
        else:#gate=gru
            model.add(Bidirectional(GRU(num_gated_connections, return_sequences=True, dropout=dropout)))

    if gate == "lstm":
        model.add(Bidirectional(LSTM(num_gated_connections, dropout=dropout)))
    else:#gate=gru
        model.add(Bidirectional(GRU(num_gated_connections, dropout=dropout)))
    model.add(Dense(1))

    if optimizer == "adam" and lr!= 0.001:
        print("Setting learning rate to"+str(lr))
        optimizer = tf.train.AdamOptimizer(lr)

    model.compile(loss='mse',optimizer=optimizer,metrics=['mape'])
    model.summary()
    return model

def rnn_model(optimizer="adam",lr=0.001, dropout=0,   gate = "lstm", gated_layers= 2, num_gated_connections=100):
    model = Sequential()
    model.add(Embedding(max_length, embedding_vector_length, input_length=max_length))

    if gated_layers==2:
        if gate == "lstm":
            model.add(LSTM(num_gated_connections, return_sequences=True, dropout=dropout))
        else:#gate=gru
            model.add(GRU(num_gated_connections, return_sequences=True, dropout=dropout))

    if gate == "lstm":
        model.add(LSTM(num_gated_connections, dropout=dropout))
    else:#gate=="gru"
        model.add(GRU(num_gated_connections, dropout=dropout))

    model.add(Dense(1))

    if optimizer == "adam" and lr!= 0.001:
        print("Setting learning rate to"+str(lr))
        optimizer = tf.train.AdamOptimizer(lr)

    model.compile(loss='mse',optimizer=optimizer,metrics=['mape'])
    model.summary()
    return model

def cnn_model(optimizer="adam", lr=0.001, dropout=0, layers=2):
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_vector_length, input_length=max_length))
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    if layers==3:
        model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(1))

    if optimizer == "adam" and lr!= 0.001:
        print("Setting learning rate to"+str(lr))
        optimizer = tf.train.AdamOptimizer(lr)

    model.compile(loss='mse',optimizer=optimizer,metrics=['mape'])
    model.summary()
    return model

def cnn_rnn_model(optimizer="adam", lr=0.001, dropout=0, gate = "lstm", num_gated_connections=100):
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_vector_length, input_length=max_length))
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    if gate == "lstm":
        model.add(LSTM(num_gated_connections, dropout=dropout))
    else:#gate="gru"
        model.add(GRU(num_gated_connections, dropout=dropout))
    model.add(Dense(1))

    if optimizer == "adam" and lr!= 0.001:
        print("Setting learning rate to"+str(lr))
        optimizer = tf.train.AdamOptimizer(lr)

    model.compile(loss='mse',optimizer=optimizer,metrics=['mape'])
    model.summary()
    return model

def mlp_model(optimizer="adam", lr=0.001, dropout= 0.0, layers= 2, fp="maccs"):

    fc_model = Sequential()

    if "maccs" in fp:
        if layers == -1:
            fc_model.add(Dense(256, activation= tf.nn.relu, input_dim = 167))
            fc_model.add(Dropout(dropout))
            fc_model.add(Dense(128, activation= tf.nn.relu))
            fc_model.add(Dropout(dropout))
            fc_model.add(Dense(64, activation= tf.nn.relu))
            fc_model.add(Dropout(dropout))
            fc_model.add(Dense(32, activation= tf.nn.relu))
        else:
            fc_model.add(Dense(512, activation= tf.nn.relu, input_dim = 167))
            fc_model.add(Dropout(dropout))
            fc_model.add(Dense(256, activation= tf.nn.relu))
            fc_model.add(Dropout(dropout))
            if layers == 3:
                    fc_model.add(Dense(128, activation= tf.nn.relu))
                    fc_model.add(Dropout(dropout))
            fc_model.add(Dense(64, activation= tf.nn.relu))

    else:#other fingerprints
        if layers == 3:
            fc_model.add(Dense(2048, activation= tf.nn.relu, input_dim = 1024))
            fc_model.add(Dropout(dropout))
            fc_model.add(Dense(1024, activation= tf.nn.relu))

        else:
            fc_model.add(Dense(1024, activation= tf.nn.relu, input_dim = 1024))

        fc_model.add(Dropout(dropout))
        fc_model.add(Dense(512, activation= tf.nn.relu))
        fc_model.add(Dropout(dropout))
        fc_model.add(Dense(256, activation= tf.nn.relu))
        fc_model.add(Dropout(dropout))
        fc_model.add(Dense(64, activation= tf.nn.relu))

    fc_model.add(Dense(1))

    if optimizer == "adam" and lr!= 0.001:
        print("Setting learning rate to"+str(lr))
        optimizer = tf.train.AdamOptimizer(lr)

    fc_model.compile(loss='mse', optimizer=optimizer, metrics=['mape'])
    fc_model.summary()
    return fc_model

def merged_rnn_mlp_model(optimizer="adam", lr=0.001, dropout= 0.0, gate = "lstm", gated_layers= 2, fp="maccs"):
    '''
    merged_lstm_mlp_model is replaced by this and allows both lstm and gru
    '''
    rnn_model = Sequential()
    rnn_model.add(Embedding(vocab_size, embedding_vector_length, input_length=max_length))
    if gated_layers == 2:
        if gate=="gru":
            rnn_model.add(GRU(64, return_sequences=True))
        else:
            rnn_model.add(LSTM(64, return_sequences=True))
    if gate=="gru":
        rnn_model.add(GRU(64))
    else:
        rnn_model.add(LSTM(64))

    fc_model = Sequential()
    if 'maccs' not in fp:
        fc_model.add(Dense(1024, activation= tf.nn.relu, input_dim = 1024))
        fc_model.add(Dropout(dropout))
        fc_model.add(Dense(512, activation= tf.nn.relu))

    else:
        fc_model.add(Dense(512, activation= tf.nn.relu, input_dim = 167))

    fc_model.add(Dense(256, activation= tf.nn.relu))
    fc_model.add(Dropout(dropout))
    fc_model.add(Dense(64, activation= tf.nn.relu))

    model = Sequential()
    model.add(Merge([rnn_model, fc_model], mode='concat'))
    model.add(Dense(64))
    model.add(Dropout(dropout))
    model.add(Dense(1))
    if optimizer == "adam" and lr!= 0.001:
        print("Setting learning rate to"+str(lr))
        optimizer = tf.train.AdamOptimizer(lr)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mape'])
    model.summary()
    return model

def merged_cnn_mlp_model(optimizer="adam", lr=0.001, dropout= 0.0, layers=2, fp="maccs"):
    """
    Replaces the merged cnn_lstm_mlp and cnn_gru_mlp
    """
    cnn_model = Sequential()
    cnn_model.add(Embedding(vocab_size, embedding_vector_length, input_length=max_length))
    cnn_model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    cnn_model.add(MaxPooling1D(pool_size=2))
    cnn_model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    cnn_model.add(MaxPooling1D(pool_size=2))
    if layers==3:
        cnn_model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
        cnn_model.add(MaxPooling1D(pool_size=2))
    cnn_model.add(Flatten())
    cnn_model.add(Dense(64))

    fc_model = Sequential()
    if 'maccs' not in fp:
        fc_model.add(Dense(1024, activation= tf.nn.relu, input_dim = 1024))
        fc_model.add(Dropout(dropout))
        fc_model.add(Dense(512, activation= tf.nn.relu))

    else:
        fc_model.add(Dense(512, activation= tf.nn.relu, input_dim = 167))

    fc_model.add(Dense(256, activation= tf.nn.relu))
    fc_model.add(Dropout(dropout))
    fc_model.add(Dense(64, activation= tf.nn.relu))

    model = Sequential()
    model.add(Merge([cnn_model, fc_model], mode='concat'))
    model.add(Dense(64))
    model.add(Dropout(dropout))
    model.add(Dense(1))
    if optimizer == "adam" and lr!= 0.001:
        print("Setting learning rate to"+str(lr))
        optimizer = tf.train.AdamOptimizer(lr)
    model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mape'])
    model.summary()
    return model

def merged_cnn_rnn_mlp_model(optimizer="adam", lr=0.001, dropout= 0.0, gate = "lstm", fp='maccs'):
    """
    Replaces the merged cnn_lstm_mlp and cnn_gru_mlp
    """
    cnn_rnn_model = Sequential()
    cnn_rnn_model.add(Embedding(vocab_size, embedding_vector_length, input_length=max_length))
    cnn_rnn_model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    cnn_rnn_model.add(MaxPooling1D(pool_size=2))
    if gate == "gru":
        cnn_rnn_model.add(GRU(64))
    else:
        cnn_rnn_model.add(LSTM(64))

    fc_model = Sequential()
    if 'maccs' not in fp:
        fc_model.add(Dense(1024, activation= tf.nn.relu, input_dim = 1024))
        fc_model.add(Dropout(dropout))
        fc_model.add(Dense(512, activation= tf.nn.relu))

    else:
        fc_model.add(Dense(512, activation= tf.nn.relu, input_dim = 167))

    fc_model.add(Dense(256, activation= tf.nn.relu))
    fc_model.add(Dropout(dropout))
    fc_model.add(Dense(64, activation= tf.nn.relu))

    model = Sequential()
    model.add(Merge([cnn_rnn_model, fc_model], mode='concat'))
    model.add(Dense(64))
    model.add(Dropout(dropout))
    model.add(Dense(1))
    if optimizer == "adam" and lr!= 0.001:
        print("Setting learning rate to"+str(lr))
        optimizer = tf.train.AdamOptimizer(lr)
    model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mape'])
    model.summary()
    return model

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("-m", "--model", help= "which model to train", required=True)
    parser.add_argument("-f", "--fingerprint", help= "which fingerprint - if no fingerprint it is SMILES2vec", required=False)
    parser.add_argument("-o", "--optimizer", help= "which optimizer(default is adam)", required=False)
    parser.add_argument("-d", "--dropout", help= "amount of dropout", required=False)
    parser.add_argument("-t", "--dataset", help= "data for training", required=False)
    parser.add_argument("-s", "--size", help= "percentage of dataset to use", required=False)
    parser.add_argument("-l", "--layers", help= "number of layers", required=False)
    parser.add_argument("-e", "--epochs", help= "epochs", required=False)
    parser.add_argument("-b", "--batch_size", help= "size of batch", required=False)
    parser.add_argument("-r", "--learning_rate", help= "learning rate", required=False)
    parser.add_argument("-c", "--recurrent_connections", help= "default is 100", required=False)
    args = parser.parse_args()

    model_type = args.model
    if args.fingerprint:
        fp_type = args.fingerprint
        smiles2vec = False
    else:
        smiles2vec = True


    if args.dropout:
        dropout = float(args.dropout)
    else:
        dropout =0

    if args.dataset or args.dataset=="pce":
        data = loadNumpy('pce_1_7') #pce
        print("Loading pce data")

    else:
        data = loadNumpy('HOMO_1_7') #HOMO

    if args.epochs:
        epochs = int(args.epochs)
    else:
        epochs = 10

    if args.optimizer:
        optimizer = args.optimizer
    else:
        optimizer = "adam"

    if args.batch_size:
        batch_size = int(args.batch_size)
    else:
        batch_size = 32

    if args.learning_rate:
        lr = float(args.learning_rate)
    else:
        lr = 0.001

    if args.recurrent_connections:
        recur_conn = int(args.recurrent_connections)
    else:
        recur_conn = 100

    SMILES = loadNumpy('SMILES_1_7_sequences')

    if not smiles2vec:
        if "maccs" in fp_type:
            if platform.system() == 'Linux':
                fp = loadNumpy(fp_type+"_cep_1_7",fp_type+"_cep")
            else:
                fp = loadNumpy(fp_type+"_cep_1_7")

            print("Loading "+fp_type+"_cep_1_7")
            print
        else:
            if platform.system() == 'Linux':
                fp = loadNumpy(fp_type+"_1024_1_7",fp_type+"_cep")
            else:
                fp = loadNumpy(fp_type+"_1024_1_7")
            print("Loading "+fp_type+"_1024_1_7")
            print

    if args.size:
        size = float(args.size)
        num_records = int(size*len(HOMO))
        X1 = SMILES[:num_records]
        if not smiles2vec:
            X2 = fp[:num_records]
        Y = data[:num_records]

    else:
        size = 1
        X1 = SMILES
        if not smiles2vec:
            X2 = fp
        Y = data
    # print(X1[0],Y[0])
    X1_train, X1_test, y_train, y_test = train_test_split(X1, Y, random_state=1024)
    # print(X1_train[0],y_train[0])
    if not smiles2vec:
        X2_train, X2_test, y_train, y_test = train_test_split(X2, Y, random_state=1024)

    start = time.time()
    if model_type =="lstm" or model_type =="gru":

        layers = args.layers
        if not layers:
            layers = 2
        else:
            layers = 1
        if not smiles2vec:
            model = merged_rnn_mlp_model(optimizer, lr=lr, dropout=dropout, gate=model_type, gated_layers=layers, fp=args.fingerprint)
            model_type = "merged_"+model_type+"_"+str(layers)+"layer_mlp"
        else:
            model = rnn_model(optimizer, lr=lr, dropout=dropout, gate=model_type, gated_layers=layers,num_gated_connections=recur_conn)
            model_type = "smiles2vec_"+model_type+"_"+str(layers)

    elif model_type == "mlp":
        layers = args.layers
        if not layers:
            layers = 2
        else:
            layers = int(layers)
        model = mlp_model(optimizer, lr=lr, dropout=dropout, layers=layers, fp=args.fingerprint)
        model_type = model_type+"_"+str(layers)

    elif model_type == "cnn":
        layers = args.layers
        if not layers:
            layers = 2
        else:
            layers = int(layers)
        model = cnn_model(optimizer, lr=lr, dropout=dropout, layers=layers)
        model_type = model_type+"_"+str(layers)
    elif "bidirectional" in model_type:
        layers = args.layers
        if not layers:
            layers = 1
        else:
            layers = int(layers)
        rnn_type = model_type.split("bidirectional_")[1]
        if rnn_type=="gru":
            gate = "gru"
        else:
            gate="lstm"

        model = bidirectional_model(optimizer, lr=lr, dropout=dropout, gate=model_type, gated_layers=layers,num_gated_connections=recur_conn)
        model_type = model_type+"_"+str(layers)
    else:
        rnn_type = model_type.split("cnn_")[1]
        if rnn_type=="mlp":
            layers = args.layers
            if not layers:
                layers = 2
            else:
                layers = int(layers)
            model = merged_cnn_mlp_model(optimizer, lr=lr, dropout=dropout, layers=layers, fp=args.fingerprint)
            model_type = "merged_"+ model_type
        else:
            if not smiles2vec:
                model = merged_cnn_rnn_mlp_model(optimizer, lr=lr, dropout=dropout, gate=rnn_type, fp=args.fingerprint)
                model_type = "merged_"+ model_type+"_mlp"
            else:
                model = cnn_rnn_model(optimizer, lr=lr, dropout=dropout, gate=rnn_type, num_gated_connections=recur_conn)
                model_type = "smiles2vec_"+ model_type

    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)

    if "mlp_" in model_type:
        history = model.fit(X2_train, y_train, shuffle=True, validation_split=0.1, \
        epochs=epochs, batch_size=batch_size, verbose=1, callbacks=[early_stop])
    elif "merged_" in model_type:
        history = model.fit([X1_train, X2_train], y_train, shuffle=True, validation_split=0.1, \
        epochs=epochs, batch_size=batch_size, verbose=1, callbacks=[early_stop])
    else:#smiles2vec and bidirectional or cnn
        history = model.fit(X1_train, y_train, shuffle=True, validation_split=0.1, \
        epochs=epochs, batch_size=batch_size, verbose=1, callbacks=[early_stop])

    prefix = None
    if optimizer!="adam":
        if smiles2vec:
            prefix = optimizer
        else:
            prefix = optimizer+"_"+fp_type
    else:#optimizer="adam"
        if not smiles2vec:#if smiles2vec and adam, prefix=None
            prefix = fp_type

    save_model(model_type, model, history, dropout, epochs, batch_size, lr, prefix=prefix)

    time_elapsed = time.time() - start

    if "mlp_" in model_type:
        [loss, mape] = model.evaluate(X2_test, y_test, verbose=0)
        y_predict = model.predict(X2_test)

    elif "smiles2vec_" in model_type:
        print("In smiles2vec")
        [loss, mape] = model.evaluate(X1_test, y_test, verbose=0)
        y_predict = model.predict(X1_test)

    else:
        [loss, mape] = model.evaluate([X1_test,X2_test], y_test, verbose=0)
        y_predict = model.predict([X1_test,X2_test])

    r2 = r2_score(y_test,y_predict)
    mean_squared_err = mse(y_test,y_predict)
    mean_absolute_err = mae(y_test, y_predict)

    print("Testing set Mean Abs percentage Error: {:2.4f}".format(mape ))
    print("Testing set Mean Abs Error: {:2.4f}".format(mean_absolute_err))
    print("Testing set Mean R2: {:2.4f}".format(r2 ))
    print("Testing set Mean Squared Error: {:2.4f}".format(mean_squared_err))
    stats = {"mape":mape, "mae":mean_absolute_err, "mse":mean_squared_err, "r2":r2, "time":time_elapsed}

    if smiles2vec:#no fingerprint in file_suffix or subject
        file_suffix = "_"+model_type+"_dropout_"+str(dropout)+"_epochs_"\
                                            +str(epochs)+"_batch_"+str(batch_size)
        subject = model_type+"_dropout_"+str(dropout)+"_epochs_"\
                                +str(epochs)+"_"+str(batch_size)+"_"+str(lr)
    else:
        file_suffix = "_"+fp_type+"_"+model_type+"_dropout_"+str(dropout)+"_epochs_"\
                                            +str(epochs)+"_batch_"+str(batch_size)

        subject = fp_type+"_"+model_type+"_dropout_"+str(dropout)+"_epochs_"\
                                +str(epochs)+"_"+str(batch_size)+"_"+str(lr)

    stats_file = "stats"+file_suffix

    saveData(stats,stats_file,"model")
    print("Stats saved in", stats_file+".pkl")

    message = prepare_message(model_type, stats, dropout, epochs, batch_size, lr, \
                        time_elapsed, size, prefix=prefix)

    try:
        send_email(subject, message)
    except:
        print("Unable to send email")
