import csv
import numpy as np
import matplotlib.pyplot as plt
from keras.layers.core import Dense,Activation,Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
training = []        #training data
data = []           #converting the csv file to an array to better handle data
testing = []        #data for testing

with open('HistoricalQuotes.csv') as csvfile: #same process of extracting data
    csvreader = csv.reader(csvfile)
    next(csvreader)
    next(csvreader)
    for row in csvreader:
        data.append(float(row[3]))
    normalize = MinMaxScaler(feature_range=(0,1))   #normalize data between 0 and 1
    data = np.reshape(data,(len(data),1))
    data = normalize.fit_transform(data)
    counter = 0
    for x in reversed(data):
        if counter <  2* len(data)/3:
            training.append(x)
        else:
            testing.append(x)
        counter+=1

def create_dataset(array,look_back):            #arrange data so that input is one time step behind target output
    data_X,data_Y = [],[]
    for x in range(len(array) - look_back - 1):
        a = array[x:x+look_back]
        data_X.append(a)
        data_Y.append(array[x+look_back])
    return np.array(data_X),np.array(data_Y)

lookBack = 1                                   #look back is one so predict only the immediate next value
trainX,trainY = create_dataset(training,lookBack)               #prepare training data
trainX = np.reshape(trainX,(trainX.shape[0],1,trainX.shape[1]))         #reshape to inputs,time-steps,features format

def train_model():
    model = Sequential()
    model.add(LSTM(800,input_shape=(1,1),return_sequences=True))        #800 LSTM units, returns full sequence of input
    model.add(Dropout(.25))                                             #25% of values will randomly be set to 0
    model.add(LSTM(200,return_sequences=False))                         #only returns last value
    model.add(Dense(units=1))                                           #Standard neural network layer
    model.add(Activation('linear'))                                     #Activation function keeps value the same
    model.compile(loss='mse',optimizer= 'adam')                         #Mean-squared error and Adam optimizer
    model.fit(trainX,trainY,epochs=30,batch_size=512,verbose=2)
    return model


model = train_model()
testing = np.reshape(testing,(len(testing),1,1))                        #reshaping for inverse normalization
predicted= model.predict(testing)
testX = np.reshape(testing,(testing.shape[0],1))
predicted = np.reshape(predicted,(predicted.shape[0],1))
testX = normalize.inverse_transform(testX)                              #inverse normalization
predicted = normalize.inverse_transform(predicted)
testX = np.reshape(testX,(testX.shape[0]))                              #reshaping for plot
predicted = np.reshape(predicted,(predicted.shape[0]))
plt.plot(testX)
plt.plot(predicted)
plt.show()






















