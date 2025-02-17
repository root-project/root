#include "TASPngWriter.h"

#include <afterrootpngwrite.h>

#include <cerrno>

/** \class TASPngWriter
\ingroup asimage

C++ wrapper over simple writer of PNG files for standard GL memory formats:
LUMINANCE, LUMINANCE_ALPHA, RGB, and RGBA.
*/

int TASPngWriter::write_png_file(std::string_view filename)
{
   if ((int)row_pointers.size() != height)
      return 1;

   FILE *fp = fopen(filename.data(), "wb");
   if (!fp)
      return errno;

   int ret = after_root_png_write(fp, width, height, color_type, bit_depth, row_pointers.data());
   if (ret)
      return ret;

   fclose(fp);

   return 0;
}
