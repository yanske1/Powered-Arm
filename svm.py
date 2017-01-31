
# coding: utf-8

# In[12]:

import numpy as np
import pandas as pd
import math
from copy import copy
from sklearn.utils import shuffle
from sklearn import svm
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.cross_validation import KFold
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV

random_state = 6598

class LowPassFilter(object):

    def __init__(self, alpha):
        self.__setAlpha(alpha)
        self.__y = self.__s = None

    def __setAlpha(self, alpha):
        alpha = float(alpha)
        if alpha<=0 or alpha>1.0:
            raise ValueError("alpha (%s) should be in (0.0, 1.0]"%alpha)
        self.__alpha = alpha

    def __call__(self, value, timestamp=None, alpha=None):        
        if alpha is not None:
            self.__setAlpha(alpha)
        if self.__y is None:
            s = value
        else:
            s = self.__alpha*value + (1.0-self.__alpha)*self.__s
        self.__y = value
        self.__s = s
        return s

    def lastValue(self):
        return self.__y


class OneEuroFilter(object):

    def __init__(self, freq, mincutoff=1.0, beta=0.0, dcutoff=1.0):
        if freq<=0:
            raise ValueError("freq should be >0")
        if mincutoff<=0:
            raise ValueError("mincutoff should be >0")
        if dcutoff<=0:
            raise ValueError("dcutoff should be >0")
        self.__freq = float(freq)
        self.__mincutoff = float(mincutoff)
        self.__beta = float(beta)
        self.__dcutoff = float(dcutoff)
        self.__x = LowPassFilter(self.__alpha(self.__mincutoff))
        self.__dx = LowPassFilter(self.__alpha(self.__dcutoff))
        self.__lasttime = None
        
    def __alpha(self, cutoff):
        te    = 1.0 / self.__freq
        tau   = 1.0 / (2*math.pi*cutoff)
        return  1.0 / (1.0 + tau/te)

    def __call__(self, x, timestamp=None):
        # ---- update the sampling frequency based on timestamps
        if self.__lasttime and timestamp:
            self.__freq = 1.0 / (timestamp-self.__lasttime)
        self.__lasttime = timestamp
        # ---- estimate the current variation per second
        prev_x = self.__x.lastValue()
        dx = 0.0 if prev_x is None else (x-prev_x)*self.__freq # FIXME: 0.0 or value?
        edx = self.__dx(dx, timestamp, alpha=self.__alpha(self.__dcutoff))
        # ---- use it to update the cutoff frequency
        cutoff = self.__mincutoff + self.__beta*np.fabs(edx)
        # ---- filter the given value
        return self.__x(x, timestamp, alpha=self.__alpha(cutoff))

def checkAccuracy(predicted,goldset):
    predicted=predicted.tolist()
    goldset=goldset.tolist()
    correct=0
    for i in range(0,len(predicted)):
        if goldset[i]==predicted[i]:
            correct+=1
        
    return (float(correct)/len(predicted))*100

def CustomScatter(x,colors,i,title1,xl,yl):
    palette = np.array(sns.color_palette("hls",10))
    
    ax = plt.subplot(1, 2, i)
    plt.title(title1)
    sc=ax.scatter(x[:,0],x[:,1],lw=0,s=40,c=palette[colors.astype(np.int)])
    plt.xlim(xl,yl)
    plt.ylim(xl,yl)

arr=np.load('data/second_third_unprocessed.npy')

train = arr[:, 0:8]
labels = arr[:,8]

duration = 1.0 # seconds
config = {
    'freq': 200,       # Hz
    'mincutoff': 0.8,  # FIXME
    'beta': 0.4,       # FIXME
    'dcutoff': 1.0     # this one should be ok
}
    
f = OneEuroFilter(**config)
timestamp = 20 # seconds
while timestamp<duration:
    filtered = f(train, timestamp)
    timestamp += 1.0/config["freq"]

normalize=preprocessing.Normalizer(norm='l2')
train = normalize.fit_transform(train.T)
train=train.T
train=preprocessing.scale(train)
x, y = shuffle(train, labels, random_state=random_state)
x_t = x[10000:]
y_t = y[10000:]
x=x[0:10000]
y=y[0:10000]
clf_trueacc = svm.SVC(kernel='rbf',C=21, gamma = 0.9)
clf_trueacc.fit(x,y)
y_testpred = clf_trueacc.predict(x_t)
accuracy = checkAccuracy(y_testpred, y_t)
print ('Accuracy is: ', accuracy)


# In[ ]:



