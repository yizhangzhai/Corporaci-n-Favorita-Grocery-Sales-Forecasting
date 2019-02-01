import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from datetime import date, timedelta
import gc
import lightgbm as lgb
import matplotlib.pyplot as plt
# %matplotlib inline
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from keras.models import Model, Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.advanced_activations import PReLU
from keras.layers.normalization import BatchNormalization
from keras.layers import LSTM,GRU,Input, Conv1D, Lambda
from keras import callbacks
from keras import optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import gc
gc.collect()
from keras.utils.vis_utils import plot_model
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

################################################
X_train = pd.read_pickle('X_train.pkl')
y_train = pd.read_pickle('y_train.pkl')
X_val = pd.read_pickle('X_val.pkl')
y_val = pd.read_pickle('y_val.pkl')
X_test = pd.read_pickle('X_test.pkl')

############################################################################################
n_filters = 20
kernel_size = 2
dilation_rate = [2**n for n in range(5)]

### dilated causal convolution
inputs = Input(shape=(None,1))
dilated_causal_layers = inputs
for R in dilation_rate:
    dilated_causal_layers = Conv1D(filters=n_filters, kernel_size=kernel_size,padding='causal',dilation_rate=R)(dilated_causal_layers)

outputs = Dense(80,activation='relu')(dilated_causal_layers)
outputs = Dropout(0.3)(outputs)
outputs = Dense(1)(outputs)

def sub_outputs(x, steps):
    return x[:,-steps:,:]

outputs_last_16 = Lambda(sub_outputs, arguments={'steps':16})(outputs)

model = Model(inputs=inputs,outputs=outputs_last_16)
model.summary()

plot_model(model, to_file='dilated_casual.png', show_shapes=True, show_layer_names=True)


model.compile(optimizer='Adam',loss='mean_absolute_error')
model.fit(train_X,train_Y,batch_size=512,epochs=30,validation_data=(validation_X,validation_Y))

pred_seq = np.zeros((X_test.shape[0],16,1))
prediction_X = X_test
for i in range(16):
    print('=== Step %s Start ===' %i)
    res = model.predict(prediction_X)[:,-1,:]
    pred_seq[:,i,:] = res
    prediction_X = np.concatenate([prediction_X,res.reshape((-1,1,1))],axis=1)

prediction_X.to_csv('test_pred_model_4.csv',index=False)
