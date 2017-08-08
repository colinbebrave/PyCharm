import numpy,scipy,scipy.spatial

# Exercise on softplus function

softplus_inputs = [-10000,-1000,-100,-10,-1,0,1,10,100,1000,10000]

# Exercise on partition function

w_small = [8.29995516951, 0.0136251921039, -9.0763759345e-06, -1.87986633012e-05, -0.536460202071,
	   0.0796022452808, 4.23454768121, 0.0291246338422, -0.00101219913069, -0.223010484194]
w_medium = [4.33750374543, 1.52257311698, 0.390929449456, 0.13684361743, 1.58873450479,
	    18.0544908209, -22.0165624317, -0.00013062227762, 39.0705138947, 0.346418068333] 
w_big  = [-1.52820754859, -0.000234000845064, -0.00527938881237, 5797.19232191, -6.64682108484,
	   18924.7087966, -69.308158911, 1.1158892974, 1.04454511882, 118.795573742]

# Exercise on Gaussian distribution

S1 = numpy.load('params/S1.npy')
S2 = numpy.load('params/S2.npy')
S3 = numpy.load('params/S3.npy')
m1 = numpy.load('params/m1.npy')
m2 = numpy.load('params/m2.npy')
m3 = numpy.load('params/m3.npy')

X1 = numpy.random.mtrand.RandomState(123).normal(0,1,[5,len(m1)])
X2 = numpy.random.mtrand.RandomState(123).normal(0,1,[20,len(m2)])
X3 = numpy.random.mtrand.RandomState(123).normal(0,1,[100,len(m3)])

