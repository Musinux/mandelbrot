#include <sys/time.h>
#include <pthread.h>
#include "consts.h"
#include "colors.h"
#include "threads.h"
#include "times.h"
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void prepare_thread_args (thread_args *args, struct s_rect *r, colorpoint *points, color **p_cols) {
  args->r = r;
  args->points = points;
  args->p_cols = p_cols;
  args->iterations = MIN_ITERATIONS;
}

void* calculate (thread_args *arg) {
  thread_args *args = (thread_args *)arg;

  int max_dimension = HEIGHT > WIDTH ? HEIGHT : WIDTH;

  int max_blocks = 64;
  int curr_blocks;
  curr_blocks = max_blocks;

  struct timeval stop, start;
  float elapsed = 0;
  gettimeofday(&start, NULL);

  for (int i = 0; i < WIDTH; i += max_blocks) {
    if (i + max_blocks >= WIDTH) {
      curr_blocks = WIDTH - i;
    }
    // mandelbrot<<<curr_blocks, diff_height>>>(args, max_dimension, i);
    TYPE<<<curr_blocks, HEIGHT>>>(args, max_dimension, i);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    cudaDeviceSynchronize();
  }

  gettimeofday(&stop, NULL);

  elapsed = timedifference_msec(start, stop);
  printf("calculate took %f milliseconds\n", elapsed);

  return NULL;
}

__global__
void mandelbrot (thread_args *args, int max_dimension, int width_offset) {
  complex c;
  complex z = {0, 0};
  // y = HEIGHT, x = WIDTH
  int y = threadIdx.x;
  // comme on ne peut pas lancer tous les calculs d'un coup, on indique à quel offset on est
  int x = blockIdx.x + width_offset;
  int curr = y * max_dimension + x;

  rect *r = args->r;

  coord_to_complex(&c, x, y, r);

  unsigned int speed = suite(&z, &c, args->iterations - 1);

  color *cols = *(args->p_cols);
  args->points[curr].c = &cols[speed - 1];
  args->points[curr].p.x = x;
  args->points[curr].p.y = y;
}

__global__
void julia (thread_args *args, int max_dimension, int width_offset) {
  complex c = JULIA_COORDS;
  // y = HEIGHT, x = WIDTH
  int y = threadIdx.x;
  // comme on ne peut pas lancer tous les calculs d'un coup, on indique à quel offset on est
  int x = blockIdx.x + width_offset;
  int curr = y * max_dimension + x;

  rect *r = args->r;

  complex z;
  coord_to_complex(&z, x, y, r);

  unsigned int speed = suite(&z, &c, args->iterations - 1);

  color *cols = *(args->p_cols);
  args->points[curr].c = &cols[speed - 1];
  args->points[curr].p.x = x;
  args->points[curr].p.y = y;
}

__device__
void coord_to_complex (complex *z, int x, int y, rect *r) {
  /*
  ici x c'est notre left
  ici y c'est notre top
  re = ((x / largeur) * (droite_precedente - gauche_precedente)) + gauche_precedente
  */
  z->re = (((double)x / (double)WIDTH) * (r->right - r->left)) + r->left;
  z->im = (((double)y / (double)HEIGHT) * (r->top - r->bottom)) + r->bottom;
}

void coord_to_complex2 (complex *z, int x, int y, rect *r) {
  z->re = (((double)x / (double)WIDTH) * (r->right - r->left)) + r->left;
  z->im = (((double)y / (double)HEIGHT) * (r->top - r->bottom)) + r->bottom;
}

__device__
unsigned int suite (complex *z, complex *c, unsigned int iterations) {
  double re;
  double pow_re = pow(z->re, 2);
  double pow_im = pow(z->im, 2);
  unsigned int speed = 0;

  while (speed < iterations) {
    re = pow_re - pow_im;
    z->im = 2 * z->re * z->im;
    z->re = re;

    z->re += c->re;
    z->im += c->im;
    pow_re = pow(z->re, 2);
    pow_im = pow(z->im, 2);
    if (sqrt(pow_im + pow_re) > 4) {
      break;
    }
    speed++;
  }
  return speed;
}
/*
// racine de la partie réelle^2 et de im^2
double module (const complex *z) {
  return sqrt(pow(z->im, 2) + pow(z->re, 2));
}

// re: ab - a'b' (a + ci)(b + di) ab + adi + a'b + a'b'
// im: ab' + ba'
void times (complex *a, const complex *b) {
  double re, im;
  re = pow(a->re, 2) - pow(a->im, 2);
  im = a->re * b->im + b->re * a->im;
  a->re = re;
  a->im = im;
}

void plus (complex *a, const complex *b) {
  a->re += b->re;
  a->im += b->im;
}
int divergence_speed (const complex *c, int iterations) {
  complex z = {0, 0};
  unsigned int n = 0;
  // z = 0 + 0i
  // f(z) = z^2 + c
  while (n < iterations) {
    times(&z, &z);
    plus(&z, c);
    if (module(&z) > 4) return n;
    n++;
  }
  return iterations;
}
*/

// on calcule la valeur du complexe en fonction de la position du cadre
// Si on a 600 PX qu'on veut faire rentrer dans -1 1
// x = 400
// 400 / 600 => 4/6 * (distance -1 1) + décalage
