
//last at begining = 0
void saw_tooth(float signal[], int siglen, float freq, int SR, int last)
{
	float wav_len = SR / freq;
	float half_wave_len = wav_len/2;


	for (int i = 0; i < siglen; i++){
		signal[i] = (((i+last) % wav_len)/half_wave_len) - 1;
	}

	last = siglen % wavlen;

	return;

}

void tri_wave(float signal[], int siglen, float freq, int SR, int last)
{
	float wav_len = SR / freq;
	float half_wave_len = wav_len/2;


	for (int i = 0; i < siglen; i++){
		signal[i] = 2 * abs((((i+last) % wav_len)/half_wave_len) - 1) - 1;
	}

	last = siglen % wavlen;

	return;
}

void noise(float[] signal, siglen )