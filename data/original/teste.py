import numpy

data = numpy.load("lim_aucloss.npy")
data2 = numpy.load("lrs_aucloss.npy")
data3 = numpy.load("lrs_AdaSym_aucloss.npy")

print(data.shape)

