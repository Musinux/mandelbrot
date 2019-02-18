#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <SDL2/SDL.h>
#include <sys/time.h>
#include "consts.h"
#include "colors.h"
#include "sdl.h"
#include "threads.h"
#include "times.h"

void render (colorpoint *points, SDL_Renderer *renderer) {
  printf("render started\n");
  struct timeval stop, start;
  float elapsed = 0;
  int p_x = 0;
  int x, y;
  int curr = 0;
  int max_dimension = HEIGHT > WIDTH ? HEIGHT : WIDTH;
  colorpoint *cp = NULL;
  color *col = NULL, *prev_color = NULL;
  SDL_Point *pt = NULL;

  gettimeofday(&start, NULL);

  for (y = 0; y < HEIGHT; y++) {
    p_x = 0;
    for (x = 0; x < WIDTH; x++) {
      curr = y * max_dimension + x;
      cp = &points[curr];
      pt = &points[curr].p;
      col = cp->c;
      if (!col) {
        continue;
      }
      if (x == 0) {
        prev_color = col;
        continue;
      } else if (same_color(prev_color, col)) {
        continue;
      }

      render_renderer(renderer, prev_color, pt, p_x);
      prev_color = col;
      p_x = pt->x;
    }

    if (col) {
      render_renderer(renderer, col, pt, p_x);
    }
  }

  gettimeofday(&stop, NULL);

  elapsed = timedifference_msec(start, stop);
  printf("print took %f milliseconds\n", elapsed);
  SDL_RenderPresent(renderer);
}

void zoom (int x, int y, float zoom, int width, int height, rect *r, complex *z1, complex *z2) {
  // WIDTH et HEIGHT sont constantes sur tout le programme (900 et 1100)
  // zoomed_width = (900 * 0.5) / 2.0;
  /*
  int zoomed_width = (width * zoom) / 2.0;
  int zoomed_height = (height * zoom) / 2.0;
  */

  int left = (float)x - (float)x * zoom;
  int right = (float)x + (float)(width - x) * zoom;
  int top = (float)y - (float)y * zoom;
  int bottom = (float)y + (float)(height - y) * zoom;

  printf("left = %d, right = %d, top = %d, bottom = %d\n",
      left, right, top, bottom);

  /*
  int left = x > zoomed_width ? x - zoomed_width + WIDTH * ratio_x : 0;
  int right = left + zoomed_width * 2;
  int top = y > zoomed_height ? y - zoomed_height : 0;
  int bottom = top + zoomed_height * 2;
  */

  coord_to_complex2(z1, left, top, r);
  coord_to_complex2(z2, right, bottom, r);

  r->left = z1->re;
  r->top = z2->im;
  r->right = z2->re;
  r->bottom = z1->im;
}

void calculate_and_render (thread_args *args, colorpoint *points, SDL_Renderer *renderer) {
  calculate(args);
  render(points, renderer);
  printf("calculate_and_render finished\n");
}

void draw_from_user_events (thread_args *args, colorpoint *points, SDL_Renderer *renderer, rect *r) {
  complex z1, z2;
  SDL_Event event;
  int x, y;
  float zoom_ratio;

  args->iterations = MIN_ITERATIONS;

  while (SDL_WaitEvent(&event)) {
    if (event.button.type == SDL_MOUSEBUTTONDOWN) {
      printf("SDL_MOUSEBUTTONDOWN\n");
      if (SDL_GetMouseState(&x, &y) & SDL_BUTTON(SDL_BUTTON_RIGHT)) {
        printf("SDL_BUTTON_RIGHT\n");
        break;
      }
      args->iterations += (float)MIN_ITERATIONS / 5.0;
      zoom_ratio = ZOOM_RATIO;
    } else if (event.button.type == SDL_MOUSEWHEEL) {
      printf("SDL_MOUSEWHEEL\n");
      SDL_GetMouseState(&x, &y);

      args->iterations += (float)MIN_ITERATIONS / 10.0;
      if (event.wheel.y == 0) continue;

      zoom_ratio = event.wheel.y < 0 ? MOUSE_WHEEL_RATIO : 2 - MOUSE_WHEEL_RATIO;
    } else continue;

    printf("zoom_ratio = %lf\n", zoom_ratio);

    printf("Mouse state x=%d, y=%d, iter=%d\n", x, y, args->iterations);
    printf("Width %lf -> %lf / Height %lf -> %lf\n", r->left, r->right, r->bottom, r->top);

    zoom(x, y, zoom_ratio, WIDTH, HEIGHT, r, &z1, &z2);
    calculate_and_render(args, points, renderer);
  }
}

void draw (rect *r, SDL_Renderer *renderer) {
  color **p_cols;
  cudaMallocManaged(&p_cols, sizeof(color *));
  *p_cols = calc_speeds(MIN_ITERATIONS);
  // on préalloue tous les points d'un seul coup
  colorpoint *points;
  cudaMallocManaged(&points, HEIGHT * WIDTH * sizeof(colorpoint));
  memset(points, 0, HEIGHT * WIDTH * sizeof(colorpoint));

  thread_args *args;

  cudaMallocManaged(&args, sizeof(thread_args));

  prepare_thread_args(args, r, points, p_cols);
  calculate_and_render(args, points, renderer);

  draw_from_user_events(args, points, renderer, r);

  printf("Stopped loop\n");

  cudaFree(points);
  cudaFree(args);
  cudaFree(*p_cols);
  cudaFree(p_cols);
}

int main (int argc, char **argv) {
  SDL_Window * window = NULL;
  SDL_Renderer * renderer = NULL;
  // rect r = {-0.845714, -0.261667, -0.2616671, -0.8457141};
  rect *r;
  cudaMallocManaged(&r, sizeof(rect));
  r->left = -2;
  r->top = 2;
  r->bottom = -2;
  r->right = 2;

  init_sdl(&window, &renderer);

  draw(r, renderer);

  free_sdl(&window, &renderer);
  cudaFree(r);
  return 0;
}
