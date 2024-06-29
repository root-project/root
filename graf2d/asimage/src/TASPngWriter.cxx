#include "TASPngWriter.h"

#include <png.h>
#include <cerrno>

// adopted from https://gist.github.com/niw/5963798

int TASPngWriter::write_png_file(std::string_view filename)
{
    FILE *fp = fopen(filename.data(), "w");
    if (!fp) return errno;

    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png) return errno;

    png_infop info = png_create_info_struct(png);
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

    if (row_pointers.empty()) return 1;

    png_write_image(png, row_pointers.data());
    png_write_end(png, NULL);

    fclose(fp);

    png_destroy_write_struct(&png, &info);

    return 0;
}
