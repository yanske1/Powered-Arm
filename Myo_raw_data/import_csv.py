import csv

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

alpha = 0.6
sensor = []
lpFilter = []

sensor1 = []
sensor.append(sensor1)
f1 = LowPassFilter(alpha)
lpFilter.append(f1)

sensor2 = []
sensor.append(sensor2)
f2 = LowPassFilter(alpha)
lpFilter.append(f2)

sensor3 = []
sensor.append(sensor3)
f3 = LowPassFilter(alpha)
lpFilter.append(f3)

sensor4 = []
sensor.append(sensor4)
f4 = LowPassFilter(alpha)
lpFilter.append(f4)

sensor5 = []
sensor.append(sensor5)
f5 = LowPassFilter(alpha)
lpFilter.append(f5)

sensor6 = []
sensor.append(sensor6)
f6 = LowPassFilter(alpha)
lpFilter.append(f6)

sensor7 = []
sensor.append(sensor7)
f7 = LowPassFilter(alpha)
lpFilter.append(f7)

sensor8 = []
sensor.append(sensor8)
f8 = LowPassFilter(alpha)
lpFilter.append(f8)


with open("10to30_every_4s.csv", "rb") as fin:
	reader = csv.reader(fin)
	for row in reader:
		for emg in sensor:
			for i in range(1,9):
				emg.append(float(row[i]))
fin.close()

for i in range(0, len(sensor)):
	for j in range(0, len(sensor[0])):
		#sensor[i][j] = sensor[i][j] - lpFilter[i](sensor[i][j]) high pass filter 
		sensor[i][j] = lpFilter[i](sensor[i][j])








