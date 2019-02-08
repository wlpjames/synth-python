#include <stdlib.h>
#include <math.h>

//last at begining = 0
float saw_tooth(float signal[], int siglen, float freq, int SR, float last, float volume)
{
	
	float wav_len = SR / freq;
	float half_wave_len = wav_len/2;

	for (int i = 0; i < siglen; i++){
		signal[i] = (( fmod((float) i+last, wav_len) / half_wave_len) - 1) * volume;
	}

	return fmod((float) siglen + last, wav_len);

}

float tri_wave(float signal[], int siglen, float freq, int SR, float last, float volume)
{
	float wav_len = SR / freq;
	float half_wave_len = wav_len/2;

	for (int i = 0; i < siglen; i++) {
		signal[i] = (2 * fabs( (fmod((float) i+last, wav_len) / half_wave_len) - 1) - 1) * volume;
	}

	return fmod((float) siglen + last, wav_len);

}
