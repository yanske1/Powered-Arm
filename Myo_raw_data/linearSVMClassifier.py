import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn import svm
style.use("ggplot")

x= [1,5,1.5,8,1,9,5,6.5,3.6]
y= [2,8,1.8,8,0.6,11,5,6,7]


#create list of list
dataList = []
for i in range(0, len(x)):
	temp = [x[i], y[i]]
	dataList.append(temp)

X = np.array(dataList)

#label the data set of y
for i in range(0, len(y)):
	if y[i] < 5:
		y[i] = 0
	else:
		y[i] = 1


clf = svm.SVC(kernel='linear', C = 1.5)
clf.fit(X,y)

print clf.predict([0.53, 0.76])
print clf.predict([10, 6])

#graphing the data
w = clf.coef_[0] 
a = -w[0]/w[1] 
xx = np.linspace(0,15)
yy = a*xx-clf.intercept_[0]/w[1]

h0 = plt.plot(xx, yy, 'k-', label="non weighted div")
plt.scatter(X[:,0], X[:,1], c = y ) #y to give color based on grouping
plt.legend()
plt.show()