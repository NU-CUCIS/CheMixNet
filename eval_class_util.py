from dl_util import *
from ml_util import train_test_split

embedding_vector_length = 32
max_length = 83
vocab_size = 23

def cnn_model(vocab,max_len,embedding_length=embedding_vector_length, optimizer="adam", lr=0.001, dropout=0, layers=2):
    model = Sequential()
    model.add(Embedding(vocab, embedding_length, input_length=max_len))
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    if layers==3:
        model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(1,activation='sigmoid'))

    if optimizer == "adam" and lr!= 0.001:
        print("Setting learning rate to"+str(lr))
        optimizer = tf.train.AdamOptimizer(lr)

    model.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['acc', precision,recall,auc])
    model.summary()
    return model

def rnn_model(optimizer="adam",lr=0.001, dropout=0,   gate = "lstm", gated_layers= 2, num_gated_connections=100,vocab=vocab_size,embedding_length=embedding_vector_length,\
              max_len=max_length):
    model = Sequential()
    model.add(Embedding(vocab, embedding_length, input_length=max_len))

    if gated_layers==2:
        if gate == "lstm":
            model.add(LSTM(num_gated_connections, return_sequences=True, dropout=dropout))
        else:#gate=gru
            model.add(GRU(num_gated_connections, return_sequences=True, dropout=dropout))

    if gate == "lstm":
        model.add(LSTM(num_gated_connections, dropout=dropout))
    else:#gate=="gru"
        model.add(GRU(num_gated_connections, dropout=dropout))

    model.add(Dense(1,activation="sigmoid"))

    if optimizer == "adam" and lr!= 0.001:
        print("Setting learning rate to"+str(lr))
        optimizer = tf.train.AdamOptimizer(lr)

    model.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['acc', precision,recall,auc])
    model.summary()
    return model

def bidirectional_rnn_model(optimizer="adam", lr=0.001, dropout=0, gate = "lstm", gated_layers= 1, num_gated_connections=100,vocab=vocab_size,embedding_length=embedding_vector_length,\
              max_len=max_length):
    model = Sequential()
    model.add(Embedding(vocab, embedding_length, input_length=max_len))
    if gated_layers==2:
        if gate == "lstm":
            model.add(Bidirectional(LSTM(num_gated_connections, return_sequences=True, dropout=dropout)))
        else:#gate=gru
            model.add(Bidirectional(GRU(num_gated_connections, return_sequences=True, dropout=dropout)))

    if gate == "lstm":
        model.add(Bidirectional(LSTM(num_gated_connections, dropout=dropout)))
    else:#gate=gru
        model.add(Bidirectional(GRU(num_gated_connections, dropout=dropout)))
    
    model.add(Dense(1,activation="sigmoid"))

    if optimizer == "adam" and lr!= 0.001:
        print("Setting learning rate to"+str(lr))
        optimizer = tf.train.AdamOptimizer(lr)

    model.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['acc', precision,recall,auc])
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
    fc_model.add(Dense(1,activation="sigmoid"))

    if optimizer == "adam" and lr!= 0.001:
        print("Setting learning rate to"+str(lr))
        optimizer = tf.train.AdamOptimizer(lr)

    fc_model.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['acc', precision,recall,auc])
    fc_model.summary()
    return fc_model

def rnn_mlp_model(optimizer="adam", lr=0.001, dropout= 0.0, gate = "lstm", gated_layers= 2, fp="maccs",vocab=vocab_size,embedding_length=embedding_vector_length,\
              max_len=max_length):
    '''
    merged_lstm_mlp_model is replaced by this and allows both lstm and gru
    '''
    rnn_model = Sequential()
    rnn_model.add(Embedding(vocab, embedding_length, input_length=max_len))
    if gated_layers == 2:
        if gate=="gru":
            rnn_model.add(GRU(64, return_sequences=True))
        else:
            rnn_model.add(LSTM(64, return_sequences=True))
    if gate=="gru":
        rnn_model.add(GRU(64))
    else:
        rnn_model.add(LSTM(64))
    rnn_model.summary()
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
    fc_model.summary()
    model = Sequential()
    model.add(Merge([rnn_model, fc_model], mode='concat'))
    model.add(Dense(64))
    model.add(Dropout(dropout))
    model.add(Dense(1,activation="sigmoid"))
    
    if optimizer == "adam" and lr!= 0.001:
        print("Setting learning rate to"+str(lr))
        optimizer = tf.train.AdamOptimizer(lr)

    model.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['acc', precision,recall,auc])
    model.summary()
    return model

def cnn_mlp_model(optimizer="adam", lr=0.001, dropout= 0.0, layers=2, fp="maccs",vocab=vocab_size,embedding_length=embedding_vector_length,\
              max_len=max_length):
    """
    Replaces the merged cnn_lstm_mlp and cnn_gru_mlp
    """
    cnn_model = Sequential()
    cnn_model.add(Embedding(vocab, embedding_length, input_length=max_len))
    cnn_model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    cnn_model.add(MaxPooling1D(pool_size=2))
    cnn_model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    cnn_model.add(MaxPooling1D(pool_size=2))
    if layers==3:
        cnn_model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
        cnn_model.add(MaxPooling1D(pool_size=2))
    cnn_model.add(Flatten())
    cnn_model.add(Dense(64))
    cnn_model.summary()

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
    fc_model.summary()

    model = Sequential()
    model.add(Merge([cnn_model, fc_model], mode='concat'))
    model.add(Dense(64))
    model.add(Dropout(dropout))
    model.add(Dense(1,activation="sigmoid"))

    if optimizer == "adam" and lr!= 0.001:
        print("Setting learning rate to"+str(lr))
        optimizer = tf.train.AdamOptimizer(lr)

    model.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['acc', precision,recall,auc])
    model.summary()
    return model

def cnn_rnn_model(optimizer="adam", lr=0.001, dropout=0, gate = "lstm", num_gated_connections=100,vocab=vocab_size,embedding_length=embedding_vector_length,\
              max_len=max_length):
    model = Sequential()
    model.add(Embedding(vocab, embedding_length, input_length=max_len))
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    if gate == "lstm":
        model.add(LSTM(num_gated_connections, dropout=dropout))
    else:#gate="gru"
        model.add(GRU(num_gated_connections, dropout=dropout))
    model.add(Dense(1,activation="sigmoid"))

    if optimizer == "adam" and lr!= 0.001:
        print("Setting learning rate to"+str(lr))
        optimizer = tf.train.AdamOptimizer(lr)

    model.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['acc', precision,recall,auc])
    model.summary()
    return model

def cnn_rnn_mlp_model(optimizer="adam", lr=0.001, dropout= 0.0, gate = "lstm", fp='maccs',vocab=vocab_size,embedding_length=embedding_vector_length,\
              max_len=max_length):
    """
    Replaces the merged cnn_lstm_mlp and cnn_gru_mlp
    """
    cnn_rnn_model = Sequential()
    cnn_rnn_model.add(Embedding(vocab, embedding_length, input_length=max_len))
    cnn_rnn_model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    cnn_rnn_model.add(MaxPooling1D(pool_size=2))
    if gate == "gru":
        cnn_rnn_model.add(GRU(64))
    else:
        cnn_rnn_model.add(LSTM(64))
    cnn_rnn_model.summary()

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
    fc_model.summary()

    model = Sequential()
    model.add(Merge([cnn_rnn_model, fc_model], mode='concat'))
    model.add(Dense(64))
    model.add(Dropout(dropout))
    model.add(Dense(1,activation='sigmoid'))

    if optimizer == "adam" and lr!= 0.001:
        print("Setting learning rate to"+str(lr))
        optimizer = tf.train.AdamOptimizer(lr)

    model.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['acc', precision,recall,auc])
    model.summary()
    return model


    