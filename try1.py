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
'''
for i in range(1,7):
    file1=pd.read_csv('data/'+str(i)+'.csv')
    ar2=np.array(file1)
    if arr.shape[0]==0:
        arr=copy(ar2)
    else:
        arr=np.vstack((arr,ar2))
    print 'arr shape is  ',arr.shape'''


arr=np.load('nisarg.npy')

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
    '''labels=[]
    for i in range(7):
        xtext,ytext=np.median(x[colors==i,:],axis=0)
        txt=ax.text(xtext,ytext,str(i),fontsize=24)
        txt.set_path_effects([PathEffects.Stroke(linewidth=5,foreground="w"),PathEffects.Normal()])
        labels.append(txt)'''


train=arr[:,0:8]
labels=arr[:,8]
print 'train shape is  ',train.shape
print 'train shape is  ',labels.shape
x, y = shuffle(train, labels, random_state=42)
x=x[0:10000]
y=y[0:10000]


pca = PCA(n_components=8)
xn=pca.fit_transform(x.T)
print(pca.explained_variance_ratio_)

normalize=preprocessing.Normalizer(norm='l2')
kf = KFold(x.shape[0], n_folds=4)
acc=[]
C=[10,20,30,40,50,60,70,80]
for c in C:
    losses=[]
    clf = svm.SVC(kernel='rbf',C=c)
    print 'training svm with C=',c
    for train_index, test_index in kf:
        X_train, X_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        X_trainNormalized = normalize.fit_transform(X_train.T)
        X_trainNormalized=X_trainNormalized.T
        X_trainNormalized=preprocessing.scale(X_train)
        #print 'type = ',type(X_trainNormalized)
        #
        X_testNormalized=normalize.transform(X_test.T)
        X_testNormalized=X_testNormalized.T
        X_testNormalized=preprocessing.scale(X_test)
        clf.fit(X_trainNormalized,y_train)
        print 'Now testing'
        ypred=clf.predict(X_testNormalized)
        acc1=checkAccuracy(ypred,y_test)
        losses.append(acc1)
        print 'Fold accuracy is ',acc1
    acc.append(np.mean(losses))
    print 'Accuracy with C=',c,' is ',np.mean(losses),'%'
