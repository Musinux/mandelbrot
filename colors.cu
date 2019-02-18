#include "consts.h"
#include "colors.h"

color * calc_speeds (int max_iterations) {
  int i;
  color *cols;
  cudaMallocManaged(&cols, sizeof(color) * max_iterations);
  for (i = 0; i < max_iterations; i++) {
    speed_to_color(&cols[i], i, max_iterations);
    // speed_to_grey(&cols[i], i, max_iterations);
  }
  printf("generated %d colors (@%p -> %p)\n", max_iterations, &cols[0], &cols[max_iterations - 1]);
  return cols;
}

void speed_to_grey (color *c, int n, int iterations) {
  unsigned int calc = ((float)n / (float)iterations) * 255.0;
  c->r = c->g = c-> b = calc;
  c->a = 0;
}

double hue2rgb (double p, double q, double t) {
  if (t < 0) {
    t += 1;
  }
  if (t > 1) {
    t -= 1;
  }
  if (t < 0.16) {
    return p + (q - p) * 6.0 * t;
  }
  if (t < 0.5) {
    return q;
  }
  if (t < 0.66) {
    return p + (q - p) * (0.66 - t) * 6.0;
  }
  return p;
}

void speed_to_color (color *c, int speed, int max_iterations) {
  // b = log(1) - log(max)
  // max = b * log(x)
  double max = ((double)max_iterations * 0.5);
  double m = (double) (speed * 3.0) / max;
  double saturation = 1;
  double light = 0.6;

  double q = light + saturation - light * saturation;
  double p = 2.0 * light - q;
  c->r = hue2rgb(p, q, m + 0.15) * 255.0;
  c->g = hue2rgb(p, q, m) * 255.0;
  c->b = hue2rgb(p, q, m - 0.15) * 255.0;
  c->a = 0;
}

void julia_high_red_to_yellow (color *c, int speed, int max_iterations) {
  double max = ((double)max_iterations * 0.5);
  double m = (double) (speed * 3.0) / max;
  double saturation = 1;
  double light = 0.6;

  double q = light + saturation - light * saturation;
  double p = 2.0 * light - q;
  c->r = hue2rgb(p, q, m + 0.15) * 255.0;
  c->g = hue2rgb(p, q, m) * 255.0;
  c->b = hue2rgb(p, q, m - 0.15) * 255.0;
  c->a = 0;
}

int same_color (const color *a, const color *b) {
  return a->r == b->r && a->g == b->g && a->b == b->b;
}
