#ifndef __COLORS_H_
#define __COLORS_H_
#include <SDL2/SDL.h>

typedef struct s_color {
  unsigned int r, g, b, a;
} color;

typedef struct s_colorpoint {
  SDL_Point p;
  struct s_color *c;
} colorpoint;


void speed_to_grey (color *c, int n, int iterations);
color * calc_speeds (int max_iterations);
double hue2rgb(double p, double q, double t);
void speed_to_color (color *c, int speed, int max_iterations);
int same_color (const color *a, const color *b);

#endif // __COLORS_H_
