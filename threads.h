#ifndef __THREADS_H__
#define __THREADS_H__
#include <sys/time.h>
#include "times.h"
#include "consts.h"
#include "colors.h"

typedef struct s_rect {
  double left;
  double top;
  double bottom;
  double right;
} rect;

typedef struct s_complex {
  double re;
  double im;
} complex;

typedef struct s_thread_args {
  struct s_rect *r;
  colorpoint *points;
  color **p_cols;
  int iterations;
} thread_args;

// void start_threads (pthread_t *threads, thread_args *args, int nbr_threads);
void prepare_thread_args (thread_args *args, struct s_rect *r, colorpoint *points, color **p_cols);
void* calculate (struct s_thread_args *arg);
double module (const complex *z);
void times (complex *a, const complex *b);
void plus (complex *a, const complex *b);
// int divergence_speed (const complex *c, int iterations);
__device__ void coord_to_complex (complex *z, int x, int y, rect *r);
void coord_to_complex2 (complex *z, int x, int y, rect *r);
__device__ unsigned int suite (complex *z, complex *c, unsigned int iterations);
__global__ void mandelbrot (struct s_thread_args *args, int max_dimension, int width_offset);
__global__ void julia (struct s_thread_args *args, int max_dimension, int width_offset);

#endif // __THREADS_H__
