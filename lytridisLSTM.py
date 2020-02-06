import numpy as np
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.model_selection import KFold, StratifiedKFold,StratifiedShuffleSplit
from scipy.io import loadmat
from scipy.signal import butter, lfilter

import scipy.io as sio
from sklearn.preprocessing import StandardScaler
import scipy as sp

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
    

def load_Giga_DB( mat ):
    #mat = sio.loadmat(subject)
    #Import MI Data
    x_left = mat['eeg']['imagery_left'][0][0]
    x_right = mat['eeg']['imagery_right'][0][0]   
    trials = mat['eeg']['n_imagery_trials'][0][0][0][0]
    channels = x_left.shape[0]
    time = x_left.shape[1]
    time_step = int(time / trials)   
    
    # Reshape data per trial for each Class
    x_left = x_left.reshape(channels,trials,time_step)
    x_left = x_left.swapaxes(0,1)
    y_left = np.ones(trials)

    x_right = x_right.reshape(channels,trials,time_step)
    x_right = x_right.swapaxes(0,1)    
    y_right = np.zeros(trials)

    X,y = np.concatenate((x_left, x_right) , axis = 0), np.concatenate((y_left, y_right))
   
                
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
    
    X_train, X_test = epoch_scaling(X_train, X_test)
    return  X_train, X_test, y_train, y_test

def load_BCICIV():
    mat = sio.loadmat("BCICIV_calib_ds1g.mat")
    cnt = mat['cnt']
    pos = mat['mrk']['pos']
    pos = pos[0][0][:]
    pos = pos.swapaxes(0,1)
    pos = pos.reshape(-1)
    y = mat['mrk']['y']
    y = y[0][0][:]
    y = y.swapaxes(0,1)
    y = y.reshape(-1)
    trials = y.shape[0]
    time_step = pos[1] - pos[0]
    channels = cnt.shape[1]   
    
    for i in list(range(trials)):
        if y[i]<0: y[i] = 0
        
    eeg_mean = cnt.mean()     
    cnt = np.subtract(cnt,eeg_mean)
  
    fcnt = butter_bandpass_filter(cnt, 7, 30, 256, order=5)

    X = np.zeros((trials,time_step,channels))
    
    for i in list(range(trials)):
        X[i] = fcnt [ pos[i] : (pos[i] + time_step),:]  
                
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    
    X_train, X_test = epoch_scaling(X_train, X_test)
    return  X_train, X_test, y_train, y_test


def calculate_csp(X, y, labels=[0,1], SCALE=False, dominant_eigen=3):
#    adapted from wrym
    X=np.swapaxes(X, 1,2) # move channels to the end

 
    x1 = X[np.where(y==labels[1])]
    x2 = X[np.where(y==labels[0])]

    x1 = x1.reshape(-1, X.shape[2])
    x2 = x2.reshape(-1, X.shape[2])
           
    c1 = np.cov(x1.transpose())
    c2 = np.cov(x2.transpose())
    # solution of csp objective via generalized eigenvalue problem
    # in matlab the signature is v, d = eig(a, b)
    d, v = sp.linalg.eig(c1-c2, c1+c2)
    d = d.real
    # make sure the eigenvalues and -vectors are correctly sorted
    indx = np.argsort(d)
    # reverse
    indx = indx[::-1]
    d = d.take(indx)
    v = v.take(indx, axis=1) # the spatial filter
    a = sp.linalg.inv(v).transpose()           
           
    return v, a, d


def apply_csp(X, filt, componets=3):

    X=np.swapaxes(X, 1,2) # move channels to the end
 
    f = np.concatenate((filt[:,0:componets], filt[:,-componets:]), axis=1)
    x_csp = np.dot(X, f)
    
    ab = np.var(x_csp, axis=1)
    ac = np.log(ab)    
    
    return ac

def epoch_scaling(X_train, X_test):
    X_train2= np.reshape(X_train, (X_train.shape[0],X_train.shape[1]*X_train.shape[2] ))
    X_test2= np.reshape(X_test,  (X_test.shape[0],X_test.shape[1]*X_test.shape[2] ))     
    scaler = preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True).fit(X_train2)    
    X_train2 = scaler.transform(X_train2)
    X_test2 = scaler.transform(X_test2)
    
    X_train2= np.reshape(X_train2, (X_train.shape[0],X_train.shape[1],X_train.shape[2] ))
    X_test2 = np.reshape(X_test2,  (X_test.shape[0],X_test.shape[1],X_test.shape[2] ))  
    return  X_train2, X_test2  

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def main():
    import os

    dataDir = "GigaDB/"
    mats = []
    for file in os.listdir( dataDir ) :
        mats.append( sio.loadmat( dataDir+file ) )
        
        
    mat = sio.loadmat("GigaDB/s01.mat")    
    X_train, X_test, y_train, y_test = load_Giga_DB(mat)    
        

    
if __name__ == "__main__":
 
    main()




