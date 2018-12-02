#include <math.h>

float calc(float curr, float last, float betareps)

{
  //expression not checked!
  return last - (betareps * (last + curr));

}


void low_pass(float* signal, int siglen, float last, float beta, int reps)

{

  //recalculates signal
  signal[0] = calc(signal[0], last, pow(beta, reps));

	for (int i = 1; i < siglen; i++) {
    signal[i] = calc(signal[i], signal[i-1], pow(beta, reps));

	}

  return;

}



 
