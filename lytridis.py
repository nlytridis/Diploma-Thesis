import numpy as np
import math
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.model_selection import KFold, StratifiedKFold,StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression

#Antonis:
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from scipy.io import loadmat
#from utils import (DownSampler, EpochsVectorizer, CospBoostingClassifier,
#                   epoch_data)
from scipy.signal import butter, lfilter
#from scnet import SCRATCH_NNET as SNET2                
#import cPickle
from sklearn.neural_network import MLPClassifier

import scipy.io as sio
from sklearn.preprocessing import StandardScaler
import scipy as sp

class SCRATCH_NNET(BaseEstimator, TransformerMixin):

    def __init__(self, batch_size=14,nn_segments=10,per_seg_hdim=2,overlap_ratio=0.0, epsilon = 0.001, reg_lambda = 0.001, num_passes=200, nn_output_dim=2,PRINT_LOSS=True):
        self.num_passes=num_passes
        self.batch_size=batch_size
        self.nn_input_dim=100 #dummy value
        self.nn_segments=nn_segments# num of segments we will split the input

        self.per_seg_hdim=per_seg_hdim
        self.overlap_ratio=overlap_ratio
        self.nn_hdim=per_seg_hdim*nn_segments # total hidden neurons
        self.PRINT_LOSS= PRINT_LOSS
        self.model = {}
        self.nn_output_dim = nn_output_dim  # output layer dimensionality

        # Gradient descent parameters 
        self.epsilon = epsilon  # learning rate for gradient descent
        self.reg_lambda = reg_lambda  # regularization strength
        
    
    def sigmoid(self,z):
            #Apply sigmoid activation function to scalar, vector, or matrix
            return 1/(1+np.exp(-z))
        
    
    def sigmoidPrime(self,z):
            #Gradient of sigmoid
            return np.exp(-z)/((1+np.exp(-z))**2)
    

    def activate(self,X,W1,b1):
    # computes the dot product of the input and the weights and returns its tanh(activation function)
        z1 = X.dot(W1) + b1
        a1 = np.tanh(z1)
        #a1 = sigmoid(z1)
        #a1 = np.maximum(0,z1)        
        return a1
    
    
    def predict(self, X):
        
        # training set size
        self.channels=X.shape[1]
        self.time = X.shape[2]
        X_channel=np.array(np.split(X,self.channels,axis=1)).mean(axis=2)
        
        overlap=int(self.time/self.nn_segments*self.overlap_ratio)
        
        if overlap%2==0:
            q=0
        else:
            q=1
            
        x=list(range(self.channels))
        for i in list(range(self.channels)):
            x[i]=[[l] for l in list(range(self.nn_segments))]

        
            # Split input to time segments (for each channel) 
        for i in list(range(self.channels)):
            for l in list(range(self.nn_segments)):
                if l==0:
                    x[i][l]=X_channel[i][:,int(self.time/self.nn_segments*l):int(self.time/self.nn_segments*(l+1)+overlap)]
                elif l== (self.nn_segments-1):
                    x[i][l]=X_channel[i][:,int(self.time/self.nn_segments*l-overlap):int(self.time/self.nn_segments*(l+1))]
                else:   
                    x[i][l]=X_channel[i][:,(int(self.time/self.nn_segments*l-(overlap/2+q))):int(self.time/self.nn_segments*(l+1)+overlap/2)]  

        W1, b1, W2, b2 = self.model['W1'], self.model['b1'],self.model['W2'], self.model['b2']
        
        #x_array = np.asarray(x)
        # Forward propagation to calculate our predictions
        
 #Initialize the activation parameter
        a1=list(range(self.channels))
       
        for k in list(range(self.channels)):#Compute the activation parameters
            a1[k]=[self.activate(x[k][i], W1[k][i], b1[k][i]) for i in list(range(self.nn_segments))]
            #a1[0][0] will have the shape (samples, per_seg_hdim)
                    
        a1=np.asarray(a1)
         
        an1=np.swapaxes(a1, 1,2)           
        an2=np.reshape(an1, (an1.shape[0],an1.shape[1], an1.shape[2]*an1.shape[3]))
        an3=np.swapaxes(an2, 0,1)    
        an5=np.reshape(an3, (an3.shape[0],an3.shape[1]*an3.shape[2]))
        
        a2=an5  
            
        z2 = a2.dot(W2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return np.argmax(probs, axis=1)
    
    
    # This function learns parameters for the neural network and returns the model.

    
    def fit(self,X, y):
        CSP_ENABLE=True
        
        # Initialize the parameters to random values. We need to learn these.
        
        self.samples = X.shape[0]  # training set size
        self.channels= X.shape[1]   
        
        time=X.shape[2]        
        
        np.random.seed(0)
        
        X_channel=np.swapaxes(X,0,1)
        overlap=int(time/self.nn_segments*self.overlap_ratio)

        self.nn_input_dim = time*self.channels
        input_per_channel=self.nn_input_dim/self.channels
        self.per_seg = int(input_per_channel/self.nn_segments + overlap )       
        
        x=list(range(self.channels))
        for i in list(range(self.channels)): # for each channel for each segment
            x[i]=[[l] for l in list(range(self.nn_segments))]
        
        #------Segment Split: For each channel split the input into the desired segments
        # compute overlap between segments     
        if overlap%2==0:
            q=0
        else:
            q=1
        
        for i in list(range(self.channels)):
            for l in list(range(self.nn_segments)):
                if l==0:
                    x[i][l]=X_channel[i][:,int(time/self.nn_segments*l):int(time/self.nn_segments*(l+1)+overlap)]
                elif l== (self.nn_segments-1):
                    x[i][l]=X_channel[i][:,int(time/self.nn_segments*l-overlap):int(time/self.nn_segments*(l+1))]
                else:   
                    x[i][l]=X_channel[i][:,(int(time/self.nn_segments*l-(overlap/2+q))):int(time/self.nn_segments*(l+1)+overlap/2)]

        #x=np.asarray(x)
        #---end Segment split----------------
        
        #Initialize W0
        #Input Layer per segment (self.per_seg)
              
        W1 = np.zeros([self.channels, self.nn_segments, self.per_seg,self.per_seg_hdim]) + np.random.randn(self.channels, self.nn_segments, self.per_seg,self.per_seg_hdim) / np.sqrt(self.per_seg)
        b1 = np.zeros((self.channels, self.nn_segments,1,self.per_seg_hdim))  


               
        #Output Layer
        #Initialize W1 the final weight before the channel connection        
        W2 = np.zeros([self.nn_hdim*self.channels, self.nn_output_dim]) + np.random.randn(self.nn_hdim*self.channels, self.nn_output_dim) / np.sqrt(self.nn_hdim)
        b2 = np.zeros((1, self.nn_output_dim))        
        
        self.model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
        
        
        # Print Network architecture:
#        print( '------Network architecture:----')
#        print('(channels x segments x input X hidden):','(',self.channels,'X',self.nn_segments,'X',self.per_seg,'X',self.per_seg_hdim,')')
#        print('(channels*segments*hidden* X output)  :','(',self.nn_hdim*self.channels,'X',self.nn_output_dim,')')
#        
#        print(' total training parameters:', (self.channels* self.nn_segments* self.per_seg*self.per_seg_hdim) + (self.channels*self.nn_segments*self.per_seg_hdim) + (self.nn_hdim*self.channels*self.nn_output_dim) + self.nn_output_dim)
#        print('---------------------------------')
        
        #compute bathes
        batches = int(math.ceil(self.samples/float(self.batch_size))) 
        
        zf = batches*self.batch_size - self.samples
        temp_x = np.swapaxes(x,2,3)
        #Check if input is divided correct
        # if not use zero filling   
        #TOD0: 
        if zf !=0:
            xzero_fil = np.zeros([x.shape[0],x.shape[1],x.shape[3],zf])
            yrand_fil = np.random.choice([0, 1], size = zf)#nikos--changed from zeros to random.choice
            temp_x = np.append(temp_x,xzero_fil,axis=3)
            y = np.append(y,yrand_fil)
        x=np.swapaxes(temp_x,2,3)
        x = np.split(x,batches,axis=2)
        x = np.asarray(x) 
#       x normaly (no CSP)  is:
#       examples/batches x channels x segments x batches  x time

#       x with CSP should be:
#       examples/batches x   segments x batches  x time/segment x channels
        if CSP_ENABLE:
            x_csp=np.copy(x)
            x_csp=np.swapaxes(x_csp,1,2)
            x_csp=np.swapaxes(x_csp,2,3)
            x_csp=np.swapaxes(x_csp,3,4)
            
            w_csp = np.zeros([self.nn_segments,self.channels,self.channels]) + np.random.randn(self.nn_segments,self.channels,self.channels) / np.sqrt(self.per_seg)        

#            w2_csp = np.zeros([self.channels, self.nn_output_dim]) + np.random.randn(self.channels, self.nn_output_dim) / np.sqrt(self.nn_hdim)
#            b2_csp = np.zeros((1, self.nn_output_dim))     

               
        probs = np.zeros([batches,self.batch_size,self.nn_output_dim])
        probs_csp = np.zeros([batches,self.batch_size,self.nn_output_dim])        
        d3 = np.zeros([batches,self.batch_size,self.nn_output_dim])
        y = np.split(y,batches)
        y = np.asarray(y)
        for l in list(range(0, self.num_passes)):
        #while self.calculate_loss(x, y)>0.01:
            #Initialize the activation parameter
            for batch in list(range(batches)):
                
#------------------------------------------------------------------------------                
#-----------------Forward Propagation for CPS preprocessing-------------------
#                examples/batches x   segments x batches  x time/segment x channels
#                x_csp2=range(self.nn_segments)
                x_csp2 = [np.dot(x_csp[batch][i], w_csp[i]) for i in list(range(self.nn_segments))]
                x_csp2 = np.asarray(x_csp2)
                x_csp3 = np.var(x_csp2, axis=2)
                x_csp4 = np.log(x_csp3)
#                flatten segments and features (out level is common)
                x_csp5= np.swapaxes(x_csp4, 0,1)
                x_csp6= np.reshape(x_csp5, (x_csp5.shape[0],x_csp5.shape[1]*x_csp5.shape[2]))
                f1=x_csp6
                
                w2_csp = np.zeros([self.nn_segments*self.channels, self.nn_output_dim]) + np.random.randn(self.nn_segments*self.channels, self.nn_output_dim) / np.sqrt(self.nn_output_dim)
                b2_csp = np.zeros((1, self.nn_output_dim))

                #Compute softmax for the output layer
                z2_csp = f1.dot(w2_csp) + b2_csp                
                exp_scores = np.exp(z2_csp)
                
                probs_csp[batch] = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)#output probabilities
                d3[batch] = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)   
                
#----------------------Backpropagation for CSP -----------------------------------------
             
                #---End of Forward propagation-------------------------------
                
                
                # Backpropagation
                delta3 = d3[batch] #probs
                delta3[list(range(self.batch_size)), y[batch].astype('int64')] -= 1 #yhat-y
                dW2_csp = (f1.T).dot(delta3)
                db2_csp = np.sum(delta3, axis=0, keepdims=True)
                delta2_csp = delta3.dot(w2_csp.T)*(1 - np.power(f1, 2))
                delta2_csp= np.reshape(delta2_csp, (delta2_csp.shape[0],self.nn_segments,self.channels))
                delta2_csp= np.swapaxes(delta2_csp, 0,1)
#                dd=2*delta2_csp*x_csp2[:,:,0,:]        
                # ------------------var log layer-----------------
                T1=x_csp2.shape[2]
                d_squared=x_csp2*x_csp2
                d_squared_sum=np.sum(x_csp2*x_csp2, axis=2)
                delta1_csp=np.zeros_like(x_csp2)
                for i in list(range(T1)):
                    delta1_csp[:,:,i,:] =    2*delta2_csp* x_csp2[:,:,i,:]/(d_squared_sum) 
                delta1_x_csp2=x_csp2 #debug
                # ----------------------------------------------
 
                dW1_csp=[] #np.zeros_like(w_csp)
                for i in list(range(self.nn_segments)):
                    tmp=[np.dot(x_csp2[i][k].T, delta1_csp[i][k]) for k in list(range(self.batch_size))]
                    tmp=np.sum(tmp, axis=0)
                    dW1_csp.append(tmp)                    
                dW1_csp= np.asarray (dW1_csp)
                
                #---parameters update
                # regularization
                dW2_csp += self.reg_lambda * w2_csp
                w2_csp += -self.epsilon * dW2_csp
                b2_csp += -self.epsilon * db2_csp
                
                for j in list(range(self.nn_segments)): 
                    w_csp[j] += -self.epsilon * dW1_csp[j]

                    
                   
#-------------------------------------------------                
                
#----------------------------END CPS ---------------------
                
                a1=list(range(self.channels))
                #---Forward propagation----------------------------------------
                for k in list(range(self.channels)):#Compute the activation parameters
               
                   a1[k]=[self.activate(x[batch][k][i], W1[k,i,:,:], b1[k,i]) for i in list(range(self.nn_segments))]
                    #a1[0][0] will have the shape (samples, per_seg_hdim)
    
                a1=np.asarray(a1)
                #--some reshaping---
                an1=np.swapaxes(a1, 1,2)           
                an2=np.reshape(an1, (an1.shape[0],an1.shape[1], an1.shape[2]*an1.shape[3]))
                an3=np.swapaxes(an2, 0,1)    
                an5=np.reshape(an3, (an3.shape[0],an3.shape[1]*an3.shape[2]))
                a1=an5    
                #--end reshaping----
 
                z2 = a1.dot(W2) + b2
                               
                #Compute softmax for the output layer
                exp_scores = np.exp(z2)
                probs[batch] = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)#output probabilities
                d3[batch] = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)                
                #---End of Forward propagation-------------------------------
                
                
                # Backpropagation
                delta3 = d3[batch] #probs
                delta3[list(range(self.batch_size)), y[batch].astype('int64')] -= 1 #yhat-y
                # for dW2 (easier as we don't have segments here)                 
                dW2 = (a1.T).dot(delta3)
                db2 = np.sum(delta3, axis=0, keepdims=True)
                delta2 = delta3.dot(W2.T) *(1 - np.power(a1, 2))# (1-tanh^2)* d3(dot)W2.T
                

               
                #We will split delta2 tow time, on on channels, once on segments
                # equiv:delta2 = np.split(delta2,self.channels,axis=1)# split on channel
                delta2 = np.reshape(delta2, (delta2.shape[0], self.channels, -1))
                delta2 = np.swapaxes(delta2, 0,1)
    
                #split delta2 on segment
                d2=[]   
                for i in list(range(self.channels)):
                    d2.append(np.split(np.asarray(delta2[i]),self.nn_segments,axis=1))
                delta2_seg=np.asarray(d2)
                # for dW1 (it is trickier as we have segments)            
                dW1=[]
                db1=[]
                for i in list(range(self.channels)):
                    dW1.append([])
                    db1.append([])                
                    for j in list(range(self.nn_segments)):
                        dW1[i].append(x[batch][i][j].T.dot(delta2_seg[i,j]))
                        db1[i].append(np.sum(delta2_seg[i,j], axis=0))
                
                # Add regularization terms (b1 and b2 don't have regularization terms)
#                l2 regularization
                dW2 += self.reg_lambda * W2
                dW1 += self.reg_lambda * W1   
                
                # Gradient descent parameter update
                for i in list(range(self.channels)):
                    for j in list(range(self.nn_segments)):
                        #W0[i,j] += -self.epsilon * dW0[i][j]
                        #b0[i,j] += -self.epsilon * db0[i][j]
                        W1[i,j] += -self.epsilon * dW1[i][j]
                        b1[i,j] += -self.epsilon * db1[i][j]
    
                W2 += -self.epsilon * dW2
                b2 += -self.epsilon * db2
        
                
                # Assign new parameters to the model
                self.model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
    
            # Calculating the loss
            probs_all = probs.reshape(self.samples+zf,self.nn_output_dim)   
            y_all = y.reshape(self.samples+zf)
            corect_logprobs = -np.log(probs_all[list(range(self.samples+zf)), y_all.astype('int64')])
            data_loss = np.sum(corect_logprobs)
            
            # Add regulatization term to loss (optional)
            #TODO: see if correct
            data_loss += self.reg_lambda / (2 * (np.sum(np.square(W1)) + np.sum(np.square(W2))))
            cross_entropy_loss = (1. / self.samples) * data_loss    
            
            # Optionally print the loss.
            # This is expensive because it uses the whole dataset, so we don't want to do it too often.
            #if self.PRINT_LOSS and l % 10 == 0:# and l>50:
            #print("Loss after iteration %i: %f" % (l, cross_entropy_loss))

            # -------Calculating the CSP loss
            probs_all_csp = probs_csp.reshape(self.samples+zf,self.nn_output_dim)   
            y_all = y.reshape(self.samples+zf)
            corect_logprobs = -np.log(probs_all_csp[list(range(self.samples+zf)), y_all.astype('int64')])
            data_loss = np.sum(corect_logprobs)
            
            # Add regulatization term to loss (optional)
            #TODO: see if correct
            data_loss += self.reg_lambda / (2 * (np.sum(np.square(w_csp)) + np.sum(np.square(w2_csp))))
            cross_entropy_loss = (1. / self.samples) * data_loss    
            
            # Optionally print the loss.
            # This is expensive because it uses the whole dataset, so we don't want to do it too often.
            #if self.PRINT_LOSS and l % 10 == 0:# and l>50:
            #print("CSP Loss after iteration %i: %f" % (l, cross_entropy_loss))             
            
        #l+=1
        return self.model
    



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

def load_Giga_DB():
    mat = sio.loadmat("s05.mat")
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


def main():
    X_train1, X_test1, y_train1, y_test1 = load_cichocki()
    X_train1=np.swapaxes(X_train1, 2,1)
    X_test1=np.swapaxes(X_test1, 2,1)
    X_train2, X_test2, y_train2, y_test2 = load_BCICIV()
    
    X_train3, X_test3, y_train3, y_test3 = load_Giga_DB()
    
    
    filt_csp1, _, _, =calculate_csp(X_train1,y_train1)
    X_train_flat1 = apply_csp(X_train1, filt_csp1, componets=1)
    X_test_flat1 = apply_csp(X_test1,   filt_csp1, componets=1) 
    
    clf=LogisticRegression('l1')
    X_train_flat1 =np.reshape(X_train1, (X_train1.shape[0],X_train1.shape[1]*X_train1.shape[2] )   )
    X_test_flat1 =np.reshape(X_test1, (X_test1.shape[0],X_test1.shape[1]*X_test1.shape[2] )   )
    
    clf.fit(X_train_flat1, y_train1)
    preds_clf1=clf.predict(X_test_flat1)
    acc_clf=accuracy_score(y_test1, preds_clf1)
    print("--->The accuracy of the CSP + LR for Cichocky subC is %f " % acc_clf)
    
    filt_csp2, _, _, =calculate_csp(X_train2,y_train2)
    X_train_flat2 = apply_csp(X_train2, filt_csp2, componets=1)
    X_test_flat2 = apply_csp(X_test2,   filt_csp2, componets=1) 
    
    clf=LogisticRegression('l1')
    X_train_flat2 =np.reshape(X_train2, (X_train2.shape[0],X_train2.shape[1]*X_train2.shape[2] )   )
    X_test_flat2 =np.reshape(X_test2, (X_test2.shape[0],X_test2.shape[1]*X_test2.shape[2] )   )
    
    clf.fit(X_train_flat2, y_train2)
    preds_clf2=clf.predict(X_test_flat2)
    acc_clf=accuracy_score(y_test2, preds_clf2)
    print("--->The accuracy of the CSP + LR for BCICIV is %f " % acc_clf)
    
    filt_csp3, _, _, =calculate_csp(X_train3,y_train3)
    X_train_flat3 = apply_csp(X_train3, filt_csp3, componets=1)
    X_test_flat3 = apply_csp(X_test3,   filt_csp3, componets=1) 
    
    clf=LogisticRegression('l1')
    X_train_flat3 =np.reshape(X_train3, (X_train3.shape[0],X_train3.shape[1]*X_train3.shape[2] )   )
    X_test_flat3 =np.reshape(X_test3, (X_test3.shape[0],X_test3.shape[1]*X_test3.shape[2] )   )
    
    clf.fit(X_train_flat3, y_train3)
    preds_clf3=clf.predict(X_test_flat3)
    acc_clf=accuracy_score(y_test3, preds_clf3)
    print("--->The accuracy of the CSP + LR for GigaDB s05 is %f " % acc_clf)
    
    from sklearn.neural_network import MLPClassifier# with logistic activation and 100 hidden neurons the acc is 96%
    clf2 = MLPClassifier(solver='sgd', batch_size ='auto', activation  ='logistic',alpha=1e-5, hidden_layer_sizes=(100, 2), random_state=1)
    clf2.fit(X_train_flat3, y_train3)
    preds_clf=clf2.predict(X_test_flat3)
    acc_clf=accuracy_score(y_test3, preds_clf)
    print("--->The accuracy of the CLF2 is %f " % acc_clf)  
    
    net = SCRATCH_NNET(batch_size=10,nn_segments=4,per_seg_hdim=3,overlap_ratio=.0, epsilon = 0.001, reg_lambda = 0.1, num_passes=100)
    net.fit(X_train3, y_train3)
    preds=net.predict(X_test3)
    acc=accuracy_score(y_test3, preds)
    print("--->The accuracy of the net is %f " % acc)
    
    
if __name__ == "__main__":
 
    main()

   
    