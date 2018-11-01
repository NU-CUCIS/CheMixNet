import pandas,numpy,keras
from numpy import load, save
import seaborn as sns
import sys
from sys import argv

from keras import initializers,optimizers
from keras.models import Sequential
from keras.layers import Dense,Dropout

from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle

class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open(argv[2], "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass

def loadNumpy(name,path='.'):
    if ".npy" in name:
        fullPath = path+'/'+name
    else:
        fullPath = path+'/'+name+'.npy'
    return load(fullPath)

def saveNumpy(obj, name, path='.'):
    if ".npy" not in name:
        fullPath = path+'/'+name
        save(fullPath, obj)
        print name,'saved successfully in',path
    else:
        fullPath = path+'/'+name.split(".npy")[0]
        save(fullPath, obj)
        print name,'saved successfully in',path

def Normalize(x):
    if len(x.shape)==1:
        return normalize(x)[0]
    else:
        return normalize(x)

seed = 7
numpy.random.seed(seed)
dropout = float(argv[1])

model = Sequential()
model.add(Dense(1024, input_dim=3455, kernel_initializer='normal', activation='relu'))
model.add(Dropout(dropout))

model.add(Dense(256, kernel_initializer='normal', activation='relu'))
model.add(Dropout(dropout))

model.add(Dense(16, kernel_initializer='normal', activation='relu'))
model.add(Dropout(dropout))

model.add(Dense(1, kernel_initializer='normal'))

# RMS = optimizers.RMSprop(lr=0.01, rho=0.9, epsilon=1e-08, decay=0.0)
Adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

model.compile(loss='mean_squared_error', optimizer=Adam)

tbCallBack = keras.callbacks.TensorBoard(log_dir='/tmp/keras_logs', write_graph=True)

# 4
X= loadNumpy('morganUnfolded')
Y = loadNumpy('HOMO')

X,Y = shuffle(X,Y)

x = X
y = Normalize(Y)

model.summary()
print 'The dropout rate is',dropout
model.fit(x,y, epochs=50, batch_size=50,  verbose=1, validation_split=0.2, callbacks=[tbCallBack])

sys.stdout = Logger()
