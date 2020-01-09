
# Importing the libraries
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from sklearn.metrics import accuracy_score


from sklearn.preprocessing import StandardScaler
import scipy.io as sio
import scipy as sp



def calculate_csp(X, y, labels=[0,1], SCALE=False, dominant_eigen=3):
#    adapted from wrym
    X=np.swapaxes(X, 1,2) # move channels to the end
     
    """ 
        Transforms the data using the CSP algorithm.
        
        Parameter X corresponds to the dataset (examples x samples x channels) and y to the labels of the data.
        
    Returns (from wyrm)
    -------
    v : 2d array
        The spatial filters optimized by the ``SPoC_lambda`` algorithm.
        Each column in the matrix is a filter.
    a : 2d array
        The spatial activation patterns that correspond to the filters
        in ``v``. Each column is a spatial pattern. when visualizing the
        SPoC components as scalp maps, plot the spatial patterns and not
        the filters. See also [haufe2014]_.
    d : 1d array
        The lambda values that correspond to the filters/patterns in
        ``v`` and ``a``, sorted from largest (positive covariance) to
        smallest (negative covariance).
    """    

 
 
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
    """Apply the CSP filter.

    Apply the spacial CSP filter to the epoched data.

    Parameters
    ----------
    epo : epoched ``Data`` object
        this method relies on the ``epo`` to have three dimensions in
        the following order: class, time, channel
    filt : 2d array
        the CSP filter (i.e. the ``v`` return value from
        :func:`calculate_csp`)
    columns : array of ints, optional
        the columns of the filter to use. The default is the first and
        the last one.

    Returns
    -------
    epo : epoched ``Data`` object
        The channels from the original have been replaced with the new
        virtual CSP channels.

    Examples
    --------

    >>> w, a, d = calculate_csp(epo)
    >>> epo = apply_csp(epo, w)

    See Also
    --------
    :func:`calculate_csp`, :func:`apply_spatial_filter`

    """
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

def load_cichocki():
    mat = sio.loadmat("SubC_6chan_2LR_s5.mat")
#    info = mat['Info']
    eegdata = mat['EEGDATA']
    eegdata=np.swapaxes(eegdata, 2,0)
    eegdata=np.swapaxes(eegdata, 2,1)        
    labels = mat['LABELS']
    labels=labels[:,0]/2
    for i in list(range(labels.shape[0])):
        if labels[i]<1: labels[i]=0
    
    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(eegdata, labels, test_size = 0.2, random_state = 0)


    X_train, X_test = epoch_scaling(X_train, X_test)
    return  X_train, X_test, y_train, y_test
    
def load_BCICIV():
    # Importing the dataset
    # http://www.bbci.de/competition/iv/desc_1.html Description and format of the dataset
    # sampling rate: fs = 100, channels = 59
    
    #cnt: the continuous EEG signals, size [time x channels]. The array is stored in datatype INT16.  
    cnt = pd.read_csv('Dataset/BCICIV/BCICIV_calib_ds1a_cnt.csv')
    raw_data = cnt.iloc[:,:].values
    
    # pos: vector of positions of the cue in the EEG signals given in unit sample, length #cues 
    pos = pd.read_csv('Dataset/BCICIV/BCICIV_calib_ds1a_pos.csv')
    pos = np.array(pos).T
    
    # y vector of target classes (-1 for class one or 1 for class two), length #cues 
    y = pd.read_csv('Dataset/BCICIV/BCICIV_calib_ds1a_y.csv')
    y = np.array(y).T
    for i in list(range(200)):
        if y[i,0]<0: y[i,0] = 0
        
    y = y.reshape(200,)    
        
    # MI trials starts at time pos[0] so data input must start at pos[0] time too.
    # Χ : input data 200 trials X time X channels
    X = np.zeros((200,800,59))
    
    for i in list(range(200)):
        X[i] = raw_data [ pos[i,0] : (pos[i,0] + 800),:]  
                
    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    
    X_train, X_test = epoch_scaling(X_train, X_test)
    return  X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = load_cichocki()
#X_train, X_test, y_train, y_test = load_BCICIV()
X_train = X_train.reshape(X_train.shape[0],X_train.shape[2],X_train.shape[1])
X_test = X_test.reshape(X_test.shape[0],X_test.shape[2],X_test.shape[1])


filt_csp, _, _, =calculate_csp(X_train,y_train)
    
X_train_flat = apply_csp(X_train, filt_csp, componets=1)
X_test_flat = apply_csp(X_test,   filt_csp, componets=1) 
clf=LogisticRegression('l1')
clf.fit(X_train_flat, y_train)
preds_clf=clf.predict(X_test_flat)
acc_clf=accuracy_score(y_test, preds_clf)
print("--->The accuracy of the CLF is %f " % acc_clf)


clf2 = MLPClassifier(solver='sgd', batch_size ='auto', activation  ='tanh',alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
clf2.fit(X_train_flat, y_train)
preds_clf=clf2.predict(X_test_flat)
acc_clf=accuracy_score(y_test, preds_clf)
print("--->The accuracy of the CLF2 is %f " % acc_clf)    










#
## --------------------- mne csp
##from mne.decoding import CSP  # noqa
##import matplotlib.pyplot as plt
##csp = CSP(n_components=2)
##X_train_flat = csp.fit_transform(X_train, y_train)
##X_test_flat  = csp.transform(X_test)    
##data = csp.patterns_
## 
###-----------------------
##imgplot = plt.imshow(filt_csp)
#
#
#
#
#
