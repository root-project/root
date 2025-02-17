#ifndef ROOT_TASPngWriter
#define ROOT_TASPngWriter

#include <string_view>
#include <vector>

class TASPngWriter {
   int width = 0;
   int height = 0;
   unsigned char color_type = 0;
   unsigned char bit_depth = 8;
   std::vector<unsigned char *> row_pointers;

public:
   TASPngWriter() = default;
   TASPngWriter(int w, int h, unsigned char t = 2, unsigned char d = 8)
      : width(w), height(h), color_type(t), bit_depth(d)
   {
   }

   void set_type(bool is_rgb, bool has_alpha)
   {
      color_type = is_rgb ? 2 : 0;
      if (has_alpha)
         color_type |= 4;
   }
   void set_luminance() { color_type = 0; }
   void set_luminance_alpha() { color_type = 4; }
   void set_rgb() { color_type = 2; }
   void set_rgba() { color_type = 6; }

   std::vector<unsigned char *> &ref_row_pointers() { return row_pointers; }

   int write_png_file(std::string_view filename);
};

#endif
