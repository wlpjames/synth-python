
import cython

# import both numpy and the Cython declarations for numpy
import numpy as np
cimport numpy as np
cdef extern void low_pass(float* signal, int siglen, float last, float beta, int reps)
cdef extern float calc(float curr, float last, float betareps)


@cython.boundscheck(False)
@cython.wraparound(False)


def l_p(np.ndarray[float, ndim=1, mode="c"] signal not None, int bufferSize, float last, float beta, int reps):

	""" a wrapper for C low pass function"""
	
	low_pass(&signal[0], bufferSize, last, beta, reps);

	return