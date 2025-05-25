#ifdef _WIN32
#include "win32/config.h"
#else
#include "config.h"
#endif

#include "afterrootpngwrite.h"

# ifdef HAVE_BUILTIN_PNG
#  include "libpng/png.h"
# else
#  include <png.h>
# endif

#include <errno.h>

// 
// Adopted from https://gist.github.com/niw/5963798
// M. Tadel, July 2024.

int after_root_png_write(FILE *fp, int width, int height,
                         unsigned char color_type, unsigned char bit_depth,
                         unsigned char** row_pointers)
{
    png_structp png;
    png_infop   info;

    png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png) return errno;

    info = png_create_info_struct(png);
    if (!info) return errno;

    if (setjmp(png_jmpbuf(png))) return 255;

    png_init_io(png, fp);

    png_set_IHDR(
        png,
        info,
        width, height,
        bit_depth,
        color_type,
        PNG_INTERLACE_NONE,
        PNG_COMPRESSION_TYPE_DEFAULT,
        PNG_FILTER_TYPE_DEFAULT
    );
    png_write_info(png, info);

    // To remove the alpha channel for PNG_COLOR_TYPE_RGB format,
    // Use png_set_filler().
    // png_set_filler(png, 0, PNG_FILLER_AFTER);


    png_write_image(png, row_pointers);
    png_write_end(png, NULL);

    png_destroy_write_struct(&png, &info);

    return 0;
}
