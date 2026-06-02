#ifndef AFTERROOTPNGWRITE_H_HEADER_INCLUDED
#define AFTERROOTPNGWRITE_H_HEADER_INCLUDED

#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

int after_root_png_write(FILE *fp, int width, int height,
                         unsigned char color_type, unsigned char bit_depth,
                         unsigned char** row_pointers);

#ifdef __cplusplus
}
#endif

#endif
