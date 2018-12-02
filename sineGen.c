#include <stdio.h>
#include <math.h>
#include <complex.h>

double complex getOmega(float freq, float sampleRate)
{
  //translate to c????
  return cexp(I*(2*M_PI * freq / sampleRate));
}

void willsSine(float signal[], double complex *last, double complex omega, float volume, int bufferS, float freq, int sampleRate) 
{
	//got to make last a float!!!!!
	for (int i = 0; i < bufferS; i++){

	signal[i] = ((double) creal(last)) * volume;

	last *= omega;
	}

	return;
}

void square(float signal[], double complex *last, double complex omega, float volume, int bufferS, float freq, int sampleRate) 
{
	//got to make last a float!!!!!
	for (int i = 0; i < bufferS; i++){

	signal[i] = sign(((double) creal(last))) * volume;

	last *= omega;
	}

	return;
}

float sign(num)
{
	//copied from net
	return (x > 0) - (x < 0);
}