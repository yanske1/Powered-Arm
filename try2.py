
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from copy import copy
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import matplotlib
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.utils import shuffle
from sklearn import svm
from sklearn import preprocessing
from sklearn.cross_validation import KFold
from sklearn.decomposition import PCA


# In[ ]:

arr=np.load('nisarg.npy')
train=arr[:,0:8]
labels=arr[:,8]
print 'train shape is  ',train.shape
print 'train shape is  ',labels.shape
x, y = shuffle(train, labels, random_state=42)
palette=np.array(sns.color_palette("hls",8))
labels=labels.reshape(1,-1)
print 'labels,' ,labels


labels.astype(np.int)
ax=plt.subplot(1,1,1)

ax.plot(x)
plt.show()

