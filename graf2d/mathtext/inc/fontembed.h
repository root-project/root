// mathtext - A TeX/LaTeX compatible rendering library. Copyright (C)
// 2008-2012 Yue Shi Lai <ylai@users.sourceforge.net>
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public License
// as published by the Free Software Foundation; either version 2.1 of
// the License, or (at your option) any later version.
//
// This library is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
// 02110-1301 USA

#if defined(_MSC_VER) && (_MSC_VER < 1800)
// Visual C++ 2008 doesn't have stdint.h
typedef __int8 int8_t;
typedef __int16 int16_t;
typedef __int32 int32_t;
typedef __int64 int64_t;
typedef unsigned __int8 uint8_t;
typedef unsigned __int16 uint16_t;
typedef unsigned __int32 uint32_t;
typedef unsigned __int64 uint64_t;
#else
#include <stdint.h>
#endif
#include <stdio.h>
#include <vector>
#include <string>
#include <map>

namespace mathtext {

   class font_embed_t {
   private:
      struct table_data_s {
         char tag[4];
         std::vector<uint8_t> data;
      };
      static void subset_rename_otf_name_table(
         struct table_data_s &table_data, uint8_t *glyph_usage);
      static void subset_ttf_glyf_table(
         struct table_data_s &table_data, uint8_t *glyph_usage);
      static void subset_ttf_loca_table(
         struct table_data_s &table_data, uint8_t *glyph_usage);
      static void subset_ttf_post_table(
         struct table_data_s &table_data, uint8_t *glyph_usage);
      static void subset_otf_cff_table(
         struct table_data_s &table_data, uint8_t *glyph_usage);
      static void parse_ttf_encoding_subtable_format4(
         std::map<wchar_t, uint16_t> &cid_map,
         const std::vector<uint8_t> &font_data,
         const size_t offset, const uint16_t length);
      static unsigned int otf_check_sum(
         const std::vector<unsigned char> &table_data);
   public:
      // I/O
      static std::vector<unsigned char> read_font_data(FILE *);
      static std::vector<unsigned char> read_font_data(const std::string &filename);
      // Font parsing
      static bool parse_otf_cff_header(
         std::string &font_name, unsigned short &cid_encoding_id,
         unsigned int &cff_offset, unsigned int &cff_length,
         const std::vector<unsigned char> &font_data);
      static bool parse_ttf_header(
         std::string &font_name, double *font_bbox,
         std::map<wchar_t, uint16_t> &cid_map,
         std::vector<std::string> &char_strings,
         const std::vector<unsigned char> &font_data);
      // Font subsetting
      static std::vector<unsigned char> subset_otf(
         const std::vector<unsigned char> &font_data,
         const std::map<wchar_t, bool> &glyph_usage);
   };

   class font_embed_postscript_t : public font_embed_t {
   public:
      static void append_asciihex(
         std::string &ascii, const uint8_t *buffer,
         const size_t length);
      static unsigned int ascii85_line_count(
         const uint8_t *buffer, const size_t length);
      static void append_ascii85(
         std::string &ascii, const uint8_t *buffer,
         const size_t length);
   public:
      static std::string font_embed_type_1(
         std::string &font_name,
         const std::vector<unsigned char> &font_data);
      static std::string font_embed_type_2(
         std::string &font_name,
         const std::vector<unsigned char> &font_data);
      static std::string font_embed_type_42(
         std::string &font_name,
         const std::vector<unsigned char> &font_data);
   };

   class font_embed_pdf_t : public font_embed_t {
   public:
      static std::string font_embed_type_1(
         std::string &font_name,
         const std::vector<unsigned char> &font_data);
      static std::string font_embed_type_2(
         std::string &font_name,
         const std::vector<unsigned char> &font_data);
      static std::string font_embed_type_42(
         std::string &font_name,
         const std::vector<unsigned char> &font_data);
   };

   class font_embed_svg_t : public font_embed_t {
      static std::string font_embed_svg(
         std::string &font_name,
         const std::vector<unsigned char> &font_data);
   };

}
