from dl_util import *
from ml_util import *

from chemixnet_util import *

def split_fit_plot_predict(model_arch, X1, X2, Y, vocab, max_len, prefix, dropout=0,\
                        gate=None, optimizer="adam", lr=0.001, epochs=20,batch_size=32):
    X1 = np.array(X1)
    X2 = np.array(X2)
    Y = np.array(Y, dtype=np.float32)

    X1_train, X1_test, y_train, y_test = train_test_split(X1, Y, random_state=1024)
    X2_train, X2_test, y_train, y_test = train_test_split(X2, Y, random_state=1024)

    model_name = model_arch.__name__
    print(model_name)
    if "rnn" in model_name:
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)
        # model_name
    else:
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

    time_start = time.time()

    if "tox" in prefix or "hiv" in prefix:
        metric = "auc"
        model_category = "classification"
    else:
        metric = "mean_absolute_percentage_error"
        model_category = "regression"

    if "mlp" in model_name:
        if "cnn" in model_name or "rnn" in model_name:#merged architecture
            print(gate,model_name)
            if gate:#either lstm or gru
                model = model_arch(dropout=dropout,optimizer=optimizer,lr=lr, vocab=vocab,\
                                                                        max_len=max_len, gate=gate )
                model_name = model_name.replace("rnn",gate)
            else:
                print("here")
                model = model_arch(dropout=dropout,optimizer=optimizer,lr=lr, vocab=vocab,\
                                                                        max_len=max_len)

            history = model.fit([X1_train,X2_train], y_train, shuffle=True, validation_split=0.1, \
                        epochs=epochs, batch_size=batch_size, verbose=1, callbacks=[early_stop])
            X_test = [X1_test, X2_test]
        else:
            model = mlp_model(dropout=dropout,optimizer=optimizer,lr=lr)
            history = model.fit(X2_train, y_train, shuffle=True, validation_split=0.1, \
                        epochs=epochs, batch_size=batch_size, verbose=1, callbacks=[early_stop])
            X_test = X2_test

    else:#just uses SMILES - cnn, rnn, cnn-rnn or bidirectional
        if gate:#rnn either gru or lstm
            model = model_arch(dropout=dropout,optimizer=optimizer,lr=lr,vocab=vocab,\
                max_len=max_len, gate=gate )
            model_name = model_name.replace("rnn",gate)
        else:
             model = model_arch(dropout=dropout,optimizer=optimizer,lr=lr,vocab=vocab,max_len=max_len)

        history = model.fit(X1_train, y_train, shuffle=True, validation_split=0.1,\
                        epochs=epochs, batch_size=batch_size, verbose=1, callbacks=[early_stop])
        X_test = X1_test


    metrics = model.evaluate(X_test, y_test)
    time_end = time.time()
    time_elapsed = time_end - time_start

    if model_category == "regression":
        y_predict = model.predict(X_test).reshape(1,-1)[0]
        r2 = r2_score(y_test,y_predict)
        mean_squared_err = mse(y_test,y_predict)
        mean_absolute_err = mae(y_test, y_predict)
        percent_mean_absolute_err = Mape(y_test, y_predict)
        mean_absolute_percent_err = metrics[1]
        stats = {"mape":percent_mean_absolute_err,  "mean_absolute_percent_error": mean_absolute_percent_err, "mae":mean_absolute_err, "mse":mean_squared_err, "r2":r2, "time":time_elapsed}
        print("Test mape%:", percent_mean_absolute_err)
    else:
        loss, accuracy, precision, recall, auc = metrics
        stats = { "accuracy":accuracy, "precision":precision, "recall":recall, "auc":auc, "time":time_elapsed}
        print("Test AUC:", auc)

    if in_jupyter():
        plot_history(history, metric)
    else:
        if time_elapsed >3600:
            message = prepare_message(model_name, stats, dropout, epochs, batch_size, lr, \
                            time_elapsed, ml_task=model_category, prefix=prefix)
            subject = prefix+"_"+model_name+"_dropout_"+str(dropout)+"_epochs_"\
                                    +str(epochs)+"_"+str(batch_size)+"_"+str(lr)
            try:
                send_email(subject, message)
            except:
                print("Unable to send email")

    file_suffix = prefix+"_"+model_name+"_dropout_"+str(dropout)\
                                        +"_epochs_"+str(epochs)+"_batch_"+str(batch_size)+"_lr_"+str(lr)
    save_history(history, "history_"+file_suffix, "model")
    saveData(stats,"stats_"+ file_suffix, "model")
    print("Stats saved in model/stats_"+ file_suffix)
    return y_test,y_predict

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("-m", "--model", help= "which model to train", required=True)
    parser.add_argument("-s", "--dataset", help= "dataset to use", required=True)

    parser.add_argument("-f", "--fingerprint", help= "which fingerprint - if no fingerprint, it's maccs", required=False)
    parser.add_argument("-o", "--optimizer", help= "which optimizer(default is adam)", required=False)
    parser.add_argument("-d", "--dropout", help= "amount of dropout", required=False)
    parser.add_argument("-l", "--layers", help= "number of layers", required=False)
    parser.add_argument("-e", "--epochs", help= "epochs", required=False)
    parser.add_argument("-b", "--batch_size", help= "size of batch", required=False)
    parser.add_argument("-r", "--learning_rate", help= "learning rate", required=False)
    parser.add_argument("-c", "--recurrent_connections", help= "default is 100", required=False)
    args = parser.parse_args()

    model_type = args.model
    dataset = args.dataset
    if "gru" in model_type:
        gate = "gru"
        model_type = model_type.replace("gru","rnn")
    elif "lstm" in model_type:
        gate = "lstm"
        model_type = model_type.replace("lstm","rnn")
    else:
        gate = None

    if dataset == "tox" or dataset == "hiv":
        X1 = loadNumpy(dataset+'_sequences')
        X2 = loadNumpy(dataset+'_maccs' )

        if "tox" in dataset:
            Y = loadNumpy('tox_nontoxic')
            vocab_size, max_len = 42, 940

        else:#hiv
            Y = loadNumpy('hiv_active')
            vocab_size, max_len = 54, 400


    elif "esol" in dataset:
        X1 = loadNumpy('esol_sequences')
        X2 = loadNumpy('esol_maccs' )

        if "standardized" in dataset:
            Y = loadNumpy('esol_standardized_solubility')#standardized data
        else:
            Y = loadNumpy('esol_solubility')
        vocab_size, max_len = 33, 98

    elif "opv" in dataset:
        if "exp" in dataset:
            X1 = loadNumpy('opv_exp_sequences')
            X2 = loadNumpy('opv_exp_maccs')
            Y = loadNumpy('opv_exp_homo')
            vocab_size, max_len = 32,176
        else:
            X1 = loadNumpy('opv_dft_sequences')
            X2 = loadNumpy('opv_dft_maccs')
            vocab_size, max_len = 31, 186
            dft_type = dataset.split("opv_")[1]
            Y = loadNumpy('opv_'+dft_type+'_homo')

    else:
        print("Dataset not defined")
        exit(4)

    if args.fingerprint:
        fp_type = args.fingerprint
    else:
        fp_type = "maccs"

    if args.dropout:
        dropout = float(args.dropout)
    else:
        dropout =0

    if args.epochs:
        epochs = int(args.epochs)
    else:
        epochs = 20

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



    split_fit_plot_predict(eval(model_type+"_model"),  X1, X2, Y, vocab_size, max_len, args.dataset,dropout=dropout, optimizer=optimizer, lr=lr, epochs=epochs,batch_size=batch_size, gate=gate)
