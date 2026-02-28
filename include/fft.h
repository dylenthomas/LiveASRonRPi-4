#ifndef FFT_H_
#define FFT_H_

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct{float Re; float Im;} complex;

#ifndef PI
# define PI     3.14159265358979323846264338327950288
#endif

static void print_vector(const char* title, complex* x, int n);
void fft(complex* v, int n, complex* tmp);
void ifft(complex* v, int n, complex* tmp);

#endif
