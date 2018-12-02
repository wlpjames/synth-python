
import cython
# import both numpy and the Cython declarations for numpy
import numpy as np
cimport numpy as np
cdef extern from "complex.h":
	double complex I
cdef extern double complex getOmega(float freq, float sampleRate)
cdef extern void willsSine(float signal[], double complex last, double complex omega, float volume, int bufferS, float freq, int sampleRate)
cdef extern void square(float signal[], double complex *last, double complex omega, float volume, int bufferS, float freq, int sampleRate) 
cdef extern void saw_tooth(float signal[], int siglen, float freq, int SR, int last, float volume)
cdef extern void tri_wave(float signal[], int siglen, float freq, int SR, int last, float volume)

@cython.boundscheck(False)
@cython.wraparound(False)

def sine_gen(np.ndarray[float, ndim=1, mode="c"] signal not None, double complex last, double complex omega, float volume, int bufferS, float freq, int sampleRate):
	willsSine(&signal[0], &last, omega, volume, bufferS, freq, sampleRate)
	return

def sine_gen(np.ndarray[float, ndim=1, mode="c"] signal not None, double complex last, double complex omega, float volume, int bufferS, float freq, int sampleRate):
	square(&signal[0], &last, omega, volume, bufferS, freq, sampleRate)
	return

def saw_gen(np.ndarray[float, ndim=1, mode="c"] signal not None, int siglen, float freq, int SR, int last):
	saw_tooth(&signal[0], siglen, freq, SR, &last);
	return

def tri_gen(np.ndarray[float, ndim=1, mode="c"] signal not None, int siglen, float freq, int SR, int last):



cdef class WaveGen:
	#declare ctypes here
	cdef double complex omega, last;
	cdef int bufferSize, sampleRate, saw_last;
	cdef float frequency, volume;
	cdef char *type


	def __init__(self, float freq, int BufferSize, char *waveType, float vol, int SR = 44100):
		self.sampleRate = SR
		self.bufferSize = BufferSize
		self.frequency = freq
		self.type = waveType
		
		#for sine
		self.omega = getOmega(self.frequency, self.sampleRate);
		self.last = I; 

		#for saw
		self.saw_last = 0;
		
		
		self.volume = vol
		
		"""
		self.adjCount = 0
		self.adjLim = 1000 #?
        """

	def updateFreq(self, float val):
		self.frequency = val
		self.omega = getOmega(self.frequency, self.sampleRate);

	def nextFrame(self):

		signal = np.zeros(self.bufferSize).astype(np.float32)

		if len(self.type) == 4: #sine
			sine_gen(signal, self.last, self.omega, self.volume, self.bufferSize, self.frequency, self.sampleRate);
		elif len(self.type) == 3: #saw
			saw_gen(signal, self.bufferSize, self. frequency, self.sampleRate, self.saw_last);
		elif len(self.type) == 8:#triangle
			pass;
		elif len(self.type) == 5: #noise
			pass;
		elif len(self.type) == 6: #square
			pass;
		elif len(self.type) == 




		return signal



