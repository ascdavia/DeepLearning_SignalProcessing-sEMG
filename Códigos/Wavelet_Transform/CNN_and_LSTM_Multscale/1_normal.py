#pip install pretty_confusion_matrix

from IPython.core.display import Pretty
#Data
import pandas as pd 
import numpy as np

#Wavelet Transform 
import pywt

#Train and Test Split
from sklearn.model_selection import train_test_split

#CNN and LSTM Model
import keras 
from keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from keras.layers import Conv1D, LSTM, Input, MaxPool1D, MaxPooling1D, Dropout, AvgPool1D, Reshape, Concatenate, Dense, Flatten

#Metrics
from sklearn.metrics import confusion_matrix, recall_score, precision_score

#Visualization 
import seaborn as sns
import matplotlib.pyplot as plt
from pretty_confusion_matrix import pp_matrix

#Import Data
x1 = pd.read_csv('https://github.com/ascdavia/DeepLearning_SignalProcessing-sEMG/blob/main/Database/sEMG_Basic_Hand_movements_upatras_csv_files/Database_1/df1_mov_all.csv?raw=true', compression = None)
x = x1.drop(x1.columns[0], axis=1)

#Reshape
x = x.values.reshape(x.shape[0], x.shape[1], 1)

#Labels
base = np.ones((150,1), dtype=np.int64)
m_cyl = base*0
m_hook = base*1
m_lat = base*2
m_palm = base*3
m_spher = base*4
m_tip = base*5

y = np.vstack([m_cyl,m_hook,m_lat,m_palm,m_spher,m_tip])
#y = pd.DataFrame(y)

#Train, test and validation split
x_train, x_aux, y_train, y_aux = train_test_split(x,y, test_size=0.30, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_aux, y_aux, test_size=0.50, random_state=23)



#Multscale CNN+LSTM Model

lr = 0.0001
batch_size = 100
epochs = 150

seq_len = 3000
n_classes = 6
n_channels = 1

opt = Adam(learning_rate=lr)

inputs = Input(shape=(seq_len, n_channels))

laywer11 = Conv1D(filters=32, kernel_size=2, activation='relu',padding='same')(inputs)
pool11 = AvgPool1D(pool_size=2, padding='same')(laywer11)
#flat11 = Flatten()(laywer11)
lstm11 = LSTM(6, return_sequences=True)(pool11)
dense11 = Dense(6, activation='relu')(lstm11)
dropout11 = Dropout(0.2)(dense11)

laywer21 = Conv1D(filters=32, kernel_size=11, activation='relu',padding='same')(inputs)
pool21 = AvgPool1D(pool_size=2, padding='same')(laywer21)
#flat21 = Flatten()(laywer21)
lstm21 = LSTM(6, return_sequences=True)(pool21)
dense21 = Dense(6, activation='relu')(lstm21)
dropout21 = Dropout(0.2)(dense21)

merge = Concatenate()([dropout11, dropout21])
#merge = Concatenate()([dense11, dense21])


#lstm1 = LSTM(500)(merge)
final_flat = Flatten()(merge)
final_dense = Dense(n_classes, activation='softmax')(final_flat)

multscale_model = Model(inputs=inputs, outputs=final_dense)
multscale_model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

#Fit Multscale CNN+LSTM Model

multscale_model.fit(x_train, y_train,
          epochs=epochs, 
          batch_size=batch_size, 
          verbose=True, 
          validation_data=(x_val, y_val))


avaliacao2 = multscale_model.evaluate(x_train,y_train)


pred2 = multscale_model.predict(x_test)
y_pred2 = pred2.argmax(axis=-1)

cm2 = confusion_matrix(y_test, y_pred2)
print(cm2)

qualidade2 = cm2.diagonal()/cm2.sum(axis=1)
desvio2 = np.std(qualidade2)
print('Qualidade:', qualidade2)
print('Desvio:', desvio2)


df_cm2 = pd.DataFrame(cm2, range(6),range(6))
pp_matrix(df_cm2)
