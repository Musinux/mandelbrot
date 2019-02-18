#ifndef __SDL_H_
#define __SDL_H_
#include "consts.h"
#include "colors.h"
#include <SDL2/SDL.h>


void init_sdl (SDL_Window ** window, SDL_Renderer ** renderer);
void free_sdl (SDL_Window ** window, SDL_Renderer ** renderer);
void render_renderer (SDL_Renderer *renderer, const color *col, const SDL_Point *pt, int prev_x);
int check_mouse_click (SDL_Event *event);

#endif // __SDL_H_
