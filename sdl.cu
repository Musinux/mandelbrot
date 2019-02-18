#include <SDL2/SDL.h>
#include "sdl.h"

void init_sdl (SDL_Window ** window, SDL_Renderer ** renderer) {
  if (SDL_Init(SDL_INIT_VIDEO) == -1) {
    perror(SDL_GetError());
    exit(EXIT_FAILURE);
  }

  *window = SDL_CreateWindow("Mandelbrot",
      SDL_WINDOWPOS_CENTERED,
      SDL_WINDOWPOS_CENTERED,
      WIDTH,
      HEIGHT,
      SDL_WINDOW_RESIZABLE);

  if (!*window) {
    perror(SDL_GetError());
    exit(EXIT_FAILURE);
  }

  *renderer = SDL_CreateRenderer(*window, -1,
      SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);

  if (!*renderer) {
    perror(SDL_GetError());
    exit(EXIT_FAILURE);
  }
  
  // SDL_SetWindowFullscreen(*window, SDL_WINDOW_FULLSCREEN);
  SDL_SetRenderDrawColor(*renderer, 0xFF, 0xFF, 0xFF, 0);
  SDL_RenderClear(*renderer);
}

void free_sdl (SDL_Window ** window, SDL_Renderer ** renderer) {
  SDL_DestroyRenderer(*renderer);
  SDL_DestroyWindow(*window);
  SDL_VideoQuit();
}

void render_renderer (SDL_Renderer *renderer, const color *col, const SDL_Point *pt, int prev_x) {
  SDL_SetRenderDrawColor(renderer, col->r, col->g, col->b, 0);
  if (prev_x < pt->x) {
    SDL_RenderDrawLine(renderer, prev_x, pt->y, pt->x, pt->y);
  } else {
    SDL_RenderDrawPoint(renderer, prev_x, pt->y);
  }
}

int check_mouse_click (SDL_Event *event) {
  int mouse_state;
  if (SDL_PollEvent(event) && SDL_MOUSEBUTTONDOWN) {
    mouse_state = SDL_GetMouseState(NULL, NULL);
    if (mouse_state & SDL_BUTTON(SDL_BUTTON_RIGHT)) {
      printf("Stop (1) !\n");
      return 1;
    } else if (mouse_state & SDL_BUTTON(SDL_BUTTON_LEFT)) {
      printf("Stopping calculate...\n");
      return -1;
    }
  }
  return 0;
}
