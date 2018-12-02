
import pyaudio
import numpy as np
from scipy import signal as sig
import pdb
import time
import cmath
import matplotlib.pyplot as plot
from timeit import default_timer as timer
from scipy.fftpack import rfft, irfft, fftfreq, fft
import pdb

import cylowPass
import sineGen
#import pygame
#import funtools

def main():
	bs = 2016
	master = masterOut()
	

	#sample setup of modular sythisis. Prone to system underruns
	#currently using slower wave genorators, writen in Python 

	master.addChannel()
	master.channels[0].addWave(1760, vol = 1, waveType = "sine")
	
	master.channels[0].addWave(391.995, vol = .6, waveType = "sine")
	master.channels[0].addWave(110, vol = .6, waveType = "sine")
	master.channels[0].addWave(261.6)
	
	master.channels[0].addWave(391.995, vol = .07, waveType = "sine")	
	master.channels[0].addWave(329.6276, vol = .15, waveType = "sine")
	#self, freq, control, , controlObj, BufferSize, waveType = "sine", MinVal = 0 , MaxVal = 1
	
	
	
	#av, al, dl, sv, sl, rl
	master.channels[0].FX.append(env(1, 0.1, 0.005, 0.85, 0.35, 0.03, BS = bs, FPS = 44100))
	
	#Band_Len, Delay, Mix = 0.5, Fade = 2
	master.channels[0].FX.append(echo(88200, 0.5, Fade = 7, BuffSize = bs, FPS = 44100))
	"""

	master.channels[0].FX.append(smpl_flt(.5, master.bufferSize, Reps = 2))
	
	
	master.channels[0].staticConts.append(LFO(2, master.channels[0].FX[1].updatebeta,
									  master.channels[0].FX[0],
									  bs, MaxVal = .8, MinVal = 0.35, waveType = "sine"))
	"""
	master.channels[0].staticConts.append(LFO(10, master.channels[0].inputs[1].updateFreq,
										  master.channels[0].inputs[0],
										  bs, MaxVal = 450, MinVal = 420, waveType = "sine"))
	
	
	
	#a = master.sing_call()
	master.playOutStream()

class masterOut:
	def __init__(self):

		self.channels = []
		self.channelsNum = 0
		self.masterVol = 0.1
		self.sampleRate = 44100
		self.bufferSize = 2016
		
	def getSlice(self, fc):

		data = combineInputs(self.channels, fc)
		return data

	def playOutStream(self): # pyaidio
		#open pyaudio
		p = pyaudio.PyAudio()

		#prepare stream
		stream = p.open(format=pyaudio.paFloat32,
    	            channels=1,
        	        rate=self.sampleRate,
            	    output=True,
            	    stream_callback=callback_maker(self),
            	    frames_per_buffer=self.bufferSize
            	    )
		
		# start the stream (4)		
		stream.start_stream()

		# wait for stream to finish (5)
		while stream.is_active():
			time.sleep(0.1)
		
		#shut everything down nicely
		stream.stop_stream()
		stream.close()
		p.terminate()

	def sing_call(self):
		a = callback_maker(self)
		a(0,self.bufferSize ,0,0,)


	def addChannel(self):
		self.channels += [channel(self.bufferSize)]
		self.channelsNum += 1

	
def callback_maker(master):
	
	def callback(in_data, frame_count, time_info, status):

		data = master.getSlice(frame_count).astype(np.float32)
		return (data*master.masterVol, pyaudio.paContinue)

	return callback







class channel:

	def __init__(self, BS, vol = 1):

		self.inputs = []
		self.staticConts = []
		self.FX = []
		self.bufferSize = BS
		self.volume = vol

	def nextFrame(self):
		self.runStatics()
		
		signal = combineInputs(self.inputs, self.bufferSize)
		signal = self.runFX(signal)

		return self.volume * signal

	def addWave(self, frequency, waveType = "sine", vol = .75):
		self.inputs += [waveGen(frequency, self.bufferSize, waveType, vol)]
		#self.inputs += [sineGen.WaveGen(frequency, self.bufferSize, waveType, vol, 44100)]

	def runStatics(self):
		for i in range(len(self.staticConts)):
			self.staticConts[i].executePerBuffer()

	def runFX(self, signal):
		
		for i in range(len(self.FX)):
			signal = self.FX[i].execute(signal)
		return signal







class waveGen:
	#mostly now replaced by pyx files and C!

	def __init__(self, freq, BufferSize, waveType, vol):
		
		self.sampleRate = 44100
		self.bufferSize = BufferSize
		self.buffer = np.empty(BufferSize, dtype=np.complex)
		self.frequency = freq
		self.omega = 0
		self.updateOmega()
		self.last = complex(1,0)
		self.type = waveType
		self.volume = vol
		self.adjCount = 0
		self.adjLim = 1000
		
		#vals for glider
		self.isGliding = False
		self.glideVal = 0
		self.originalFreq = self.frequency

	def sineGen(self):

		#start = timer() #reduce function!!! functools
		for i in range(0, self.bufferSize):
			self.buffer[i] = self.last
			self.last=self.last*self.omega


		self.glide()
		return self.volume * np.imag(self.buffer).astype(np.float32)

	def squareGen(self):
		
		for i in range(0, self.bufferSize):
			self.buffer[i] = np.sign(self.last)
			self.last=self.last*self.omega
			
		self.glide()
		return np.real(self.volume * self.buffer).astype(np.float32)

	def triangleGen(self):
		#todo
		return self.buffer

	def glide(self): 
			
		if (self.isGliding == True 
				and self.frequency < self.originalFreq*2 
					and self.frequency > 0-(self.originalFreq/3)*2):
			
			self.frequency += self.glideVal
			self.updateOmega()

	def setGlide(self, val):
		self.isGliding = True
		self.glideVal = val

	def updateOmega(self):
		self.omega = cmath.exp(1j*(2*cmath.pi * self.frequency / self.sampleRate))

	def updateFreq(self, val):
		self.frequency = val
		self.updateOmega()

	def nextFrame(self):
		if self.type == "sine":
			return self.sineGen()
		elif self.type == "square":
			return self.squareGen()
		elif self.type == "triangle":
			return self.triangleGen()








def combineInputs(inputs, fc):
	
	#create silent array
	slices = np.zeros(fc, dtype=np.float32)
	#add the output from each channel to array
	for i in range(len(inputs)):
		slices = np.add(slices, inputs[i].nextFrame())

	return slices









class LFO:
	#a class of object that directly controls a source, source can be any value

	def __init__(self, freq, control, controlObj, BufferSize, waveType = "sine", MinVal = 0 , MaxVal = 1):

		self.frequency = freq
		self.maxVal = MaxVal
		self.minVal = MinVal
		self.bufferSize = BufferSize

		#a slow wave to control input
		self.wave = waveGen(freq, BufferSize, waveType, 1)
		#self.wave = sineGen.WaveGen(freq, self.bufferSize, waveType, 1, 44100)
		#associated with another waveGen
		self.output = []
		self.output.append(control)

		self.controlObjs = []
		self.controlObjs.append(controlObj)

	def executePerBuffer(self):
		#sets the control to last point of self.wave
		nextFrame = self.wave.nextFrame()
		lastVal = (nextFrame[self.bufferSize-1] + 1) / 2
		
		#the value of the thing pointed to here becomes set.	
		val = (lastVal*(self.maxVal-self.minVal)) + self.minVal 
		self.output[0](val)


class smpl_flt:

	def __init__(self, Beta, BufferSize, Reps = 1):
		self.beta = Beta
		self.reps = Reps
		self.bufferSize = BufferSize
		self.last = 0


	def Ex_mov_av(self, signal):
		
		#for reps in range(self.reps):
		signal[0] = self.equ(signal[0], self.last)
		#signal[0] = self.last - (self.beta * (self.last - signal[0]))
		for i in range(1, self.bufferSize):
			#signal[i] = signal[i-1] - (self.beta * (signal[i-1] - signal[i]))
			signal[i] = self.equ(signal[i], signal[i-1])		
		self.last = signal[self.bufferSize - 1]

		return signal

	def equ(self, curr, last):
		return last - (self.beta ** self.reps) * last + (self.beta ** self.reps) * curr

	def execute(self, Signal):
		signal = Signal.astype(np.float32)
		cylowPass.l_p(signal, self.bufferSize, self.last, self.beta, self.reps)
		self.last = signal[-1]
		return signal

	def updatebeta(self, val):
		self.beta = float(val)
		#print("beta: ", val)

	def updateReps(self, val):
		self.reps = int(val)

 

class env:

	def __init__(self, av, al, dl, sv, sl, rl, BS = 44100, FPS = 44100):

		#lengths in seconds
		#volumes in percent/100
		self.bs = BS;
		self.fps = FPS;
		self.a_v = av; self.a_l = al;
		self.d_l = dl; self.s_v = sv;
		self.s_l = sl; self.r_l = rl;
		self.curr = 0;

		self.blueprint = self.blueprint();

 

	def blueprint(self):

		a = np.linspace(0, self.a_v, self.a_l * self.fps)
		d= np.linspace(self.a_v, self.s_v, self.d_l * self.fps)
		s = np.full(int(self.s_l * self.fps), self.s_v)
		r = np.linspace(self.s_v, 0, self.r_l * self.fps)

		tot = np.concatenate((a, d, s, r))


		"""
		x = (np.append(np.arange(0, self.a_v, self.a_l * self.fps),

			np.append(np.arange(self.a_v, self.s_v, self.d_l * self.fps),

			np.append(np.full(int(self.s_l * self.fps), self.s_v),

			np.arange(self.s_v, 0, self.r_l * self.fps)))));
		"""
		#plot.plot(np.arange(len(tot)), tot)
		return tot

 

	def execute(self, signal):

	#dogy AF logical programing
		#print("doing env")
		if self.curr <= len(self.blueprint):

 
			if self.curr + self.bs <= len(self.blueprint):
				
				for i in range(self.bs):
					signal[i] *= self.blueprint[self.curr + i]

			else:
				
				for i in range(len(self.blueprint) - self.curr):
					signal[i] *= self.blueprint[self.curr + i]

				for i in range((self.curr + self.bs) - len(self.blueprint)):
					signal[i+len(self.blueprint) - self.curr] = 0;

		else:
			#print("3")
			signal = np.zeros(self.bs);

		self.curr += self.bs
		return signal



	def start_new():
		self.curr = 0;

 

import numpy as np

class echo:

	def __init__(self, Band_Len, Delay,

				Mix = 0.5, Fade = 2,

				BuffSize = 126, FPS = 44100):

		self.fps = FPS

		self.band_len = Band_Len

		self.buffSize = BuffSize

		self.fade = Fade #val between 0 and 1

		self.delay = Delay # val in seconds

		self.S = 0 #start point for playback

		self.R = int(FPS * self.delay) #start point for record

		self.band = np.zeros(self.band_len)

		self.mix = Mix # low = heavy dry, 0-1, high = full wet

 

	def rec(self, signal, band):

		return (band / self.fade) + signal;

 

	def take(self, signal, band):

		return (signal * (1 - self.mix)) + (band * self.mix); 

 

	def proccess(self, signal):

		for i in range(self.buffSize):
			

			signal[i] = self.take(signal[i], self.band[self.S]);

			#bandIndex = (self.band_len // self.R + self.band_len % self.R) -1

			self.band[self.R] = self.rec(signal[i], self.band[self.R]);
			
			self.R += 1
			if self.R >= self.band_len:
				self.R = 0
			
			self.S += 1
			if self.S >= self.band_len:
				self.S = 0

		
		
		return signal

 

	def execute(self, signal):

		return self.proccess(signal);

 



if __name__ == "__main__":
    main()





