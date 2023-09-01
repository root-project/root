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

#include "../inc/fontembed.h"
#include <algorithm>
#include <string.h>
#include <stdio.h>
#ifdef WIN32
#define snprintf _snprintf
#endif

// ROOT integration
#include <ROOT/RConfig.hxx>
#ifdef R__BYTESWAP
#ifndef LITTLE_ENDIAN
#define LITTLE_ENDIAN 1
#endif // LITTLE_ENDIAN
#include "Byteswap.h"
#define bswap_16(x)   Rbswap_16((x))
#define bswap_32(x)   Rbswap_32((x))
#else // R__BYTESWAP
#ifdef LITTLE_ENDIAN
#undef LITTLE_ENDIAN
#endif // LITTLE_ENDIAN
#endif // R__BYTESWAP

// References:
//
// Adobe Systems, Inc. and Microsoft Corp., OpenType specification
// (2002), version 1.4.
//
// Apple Computer, Inc., TrueType reference ranual (2002)
//
// Microsoft Corp., TrueType 1.0 font files: technical specification
// (1995), version 1.66

namespace mathtext {

   typedef int32_t fixed_t;

   void font_embed_t::parse_ttf_encoding_subtable_format4(
                                                          std::map<wchar_t, uint16_t> &cid_map,
                                                          const std::vector<uint8_t> &font_data, const size_t offset,
                                                          const uint16_t length)
   {
      cid_map.clear();

      size_t offset_current = offset;

      struct ttf_encoding_subtable_format4_s {
         uint16_t seg_count_x2;
         uint16_t search_range;
         uint16_t entry_selector;
         uint16_t range_shift;
      } encoding_subtable_format4;

      memcpy(&encoding_subtable_format4,
             &font_data[offset_current],
             sizeof(struct ttf_encoding_subtable_format4_s));
      offset_current +=
      sizeof(struct ttf_encoding_subtable_format4_s);
#ifdef LITTLE_ENDIAN
      encoding_subtable_format4.seg_count_x2 =
      bswap_16(encoding_subtable_format4.seg_count_x2);
#endif // LITTLE_ENDIAN

      const uint16_t seg_count =
      encoding_subtable_format4.seg_count_x2 >> 1;
      uint16_t *end_code = new uint16_t[seg_count];

      memcpy(end_code, &font_data[offset_current],
             seg_count * sizeof(uint16_t));
      offset_current += seg_count * sizeof(uint16_t);
#ifdef LITTLE_ENDIAN
      for (uint16_t segment = 0; segment < seg_count; segment++) {
         end_code[segment] = bswap_16(end_code[segment]);
      }
#endif // LITTLE_ENDIAN

      uint16_t reserved_pad;

      memcpy(&reserved_pad, &font_data[offset_current],
             sizeof(uint16_t));
      offset_current += sizeof(uint16_t);

      uint16_t *start_code = new uint16_t[seg_count];

      memcpy(start_code, &font_data[offset_current],
             seg_count * sizeof(uint16_t));
      offset_current += seg_count * sizeof(uint16_t);
#ifdef LITTLE_ENDIAN
      for (uint16_t segment = 0; segment < seg_count; segment++) {
         start_code[segment] = bswap_16(start_code[segment]);
      }
#endif // LITTLE_ENDIAN

      uint16_t *id_delta = new uint16_t[seg_count];

      memcpy(id_delta, &font_data[offset_current],
             seg_count * sizeof(uint16_t));
      offset_current += seg_count * sizeof(uint16_t);
#ifdef LITTLE_ENDIAN
      for (uint16_t segment = 0; segment < seg_count; segment++) {
         id_delta[segment] = bswap_16(id_delta[segment]);
      }
#endif // LITTLE_ENDIAN

      const uint16_t variable =
      (length >> 1) - (seg_count << 2) - 8;
      uint16_t *id_range_offset =
      new uint16_t[seg_count + variable];

      memcpy(id_range_offset, &font_data[offset_current],
             (seg_count + variable) * sizeof(uint16_t));
      offset_current += (seg_count + variable) * sizeof(uint16_t);
#ifdef LITTLE_ENDIAN
      for (uint16_t j = 0; j < seg_count + variable; j++) {
         id_range_offset[j] = bswap_16(id_range_offset[j]);
      }
#endif // LITTLE_ENDIAN

      for (uint16_t segment = 0; segment < seg_count; segment++) {
         for (uint32_t code = start_code[segment];
              code <= end_code[segment]; code++) {
            const uint16_t inner_offset = segment +
            (id_range_offset[segment] >> 1) +
            (code - start_code[segment]);
            const uint16_t glyph_index =
            id_range_offset[segment] == 0 ?
            id_delta[segment] + code :
            inner_offset >= seg_count + variable ?
            0 : id_range_offset[inner_offset];

            cid_map[static_cast<wchar_t>(code)] = glyph_index;
         }
      }

      delete [] end_code;
      delete [] start_code;
      delete [] id_delta;
      delete [] id_range_offset;
   }

   /////////////////////////////////////////////////////////////////////
   // Currently unfinished font subsetting code below
   /////////////////////////////////////////////////////////////////////

#if 0
   // Rename:
   // name
   // Subset:
   // glyf, loca, hmtx, vmtx, cmap, hdmx, VDMX, kern, LTSH, VORG
   // Conditional subset:
   // post 2.0
   // Remove completely:
   // DSIG, BASE, GDEF, GPOS, GSUB, JSTF, EBDT, EBLC, EBSC

   void font_embed_t::subset_rename_otf_name_table(
                                                   struct table_data_s &table_data,
                                                   std::map<wchar_t, bool> glyph_usage)
   {
      // Prefix name IDs 1, 4, 6, 16, 19, 20, 21

      // No platform ID other than 1 and 3 is permitted for the name
      // table

      // Platform ID = 1: 1 byte
      // Platform ID = 3: 2 byte

      // Reset UID 4,000,000 and 4,999,999 and XUID
   }

   void font_embed_t::subset_ttf_glyf_table(
                                            struct table_data_s &table_data,
                                            std::map<wchar_t, bool> glyph_usage)
   {
   }

   void font_embed_t::subset_ttf_loca_table(
                                            struct table_data_s &table_data,
                                            std::map<wchar_t, bool> glyph_usage)
   {
   }

   void font_embed_t::subset_ttf_post_table(
                                            struct table_data_s &table_data,
                                            std::map<wchar_t, bool> glyph_usage)
   {
   }

   class cff_index_t {
   public:
      uint16_t count;
      uint8_t off_size;
      std::vector<uint32_t> offset;
      std::vector<uint8_t> data;
      cff_index_t(const uint8_t *input_data)
      {
         memcpy(&count, input_data, sizeof(uint16_t));
#ifdef LITTLE_ENDIAN
         count = bswap_16(count);
#endif // LITTLE_ENDIAN
         memcpy(&off_size, input_data + sizeof(uint16_t),
                sizeof(uint8_t));

         if (!(off_size >= 1 && off_size < 5)) {
            return;
         }

         const uint8_t *input_data_offset =
         input_data + sizeof(uint16_t) + sizeof(uint8_t);

         // The off by one or indexing from one convention in CFF
         // is corrected here, i.e. the resulting offset table is
         // zero-based, and not the CFF one.

         offset.reserve(count + 1);
         switch (off_size) {
            case 1:
               for (size_t i = 0; i < count + 1; i++) {
                  offset.push_back(input_data_offset[i]);
               }
               break;
            case 2:
               for (size_t i = 0; i < count + 1; i++) {
                  offset.push_back(reinterpret_cast<uint16_t *>(
                                                                input_data_offset)[i]);
#ifdef LITTLE_ENDIAN
                  offset.back() = bswap_16(offset.back());
#endif // LITTLE_ENDIAN
               }
               break;
            case 3:
               for (size_t i = 0; i < 3 * (count + 1); i += 3) {
                  const uint32_t value =
                  input_data_offset[3 * i] << 16 |
                  input_data_offset[3 * i + 1] << 8 |
                  input_data_offset[3 * i + 2];

                  offset.push_back(value);
               }
               break;
            case 4:
               for (size_t i = 0; i < count + 1; i++) {
                  offset.push_back(reinterpret_cast<uint32_t *>(
                                                                input_data_offset)[i]);
#ifdef LITTLE_ENDIAN
                  offset.back() = bswap_32(offset.back());
#endif // LITTLE_ENDIAN
               }
               break;
         }

         const uint8_t *input_data_data =
         input_data_offset + off_size * (count + 1);

         data = std::vector<uint8_t>(input_data_data, input_data_data
                                     }
                                     ~cff_index_t(void)
                                     {
                                     }
                                     };

                                     void font_embed_t::subset_otf_cff_table(
                                                                             struct table_data_s &table_data,
                                                                             std::map<wchar_t, bool> glyph_usage)
                                     {
                                     }
#endif

                                     uint32_t font_embed_t::otf_check_sum(
                                                                          const std::vector<uint8_t> &table_data)
                                     {
                                        const uint32_t *table =
                                        reinterpret_cast<const uint32_t *>(&(table_data[0]));
                                        const uint32_t nword = table_data.size() >> 2;
                                        uint32_t sum = 0;

                                        for (size_t i = 0; i < nword; i++) {
#ifdef LITTLE_ENDIAN
                                           sum += bswap_32(table[i]);
#else // LITTLE_ENDIAN
                                           sum += table[i];
#endif // LITTLE_ENDIAN
                                        }

                                        // Do not assume 0x00 padding and calculate partial uint32_t
                                        // checksums directly.
                                        const uint8_t *table_tail =
                                        reinterpret_cast<const uint8_t *>(&(table[nword]));

                                        switch(table_data.size() & 3U) {
                                           case 3:   sum += table_tail[2] << 8;
                                           case 2:   sum += table_tail[1] << 16;
                                           case 1:   sum += table_tail[0] << 24; break;
                                        }

                                        return sum;
                                     }

                                     std::vector<uint8_t> font_embed_t::read_font_data(
                                                                                       FILE *fp)
                                     {
                                        std::vector<uint8_t> font_data;

                                        if (fp == NULL) {
                                           return font_data;
                                        }
                                        if (fseek(fp, 0L, SEEK_SET) == -1) {
                                           perror("fseek");
                                           return font_data;
                                        }
                                        if (fseek(fp, 0L, SEEK_END) == -1) {
                                           perror("fseek");
                                           return font_data;
                                        }

                                        const long length = ftell(fp);

                                        if (length == -1) {
                                           perror("ftell");
                                           return font_data;
                                        }
                                        font_data.resize(length);
                                        if (fseek(fp, 0L, SEEK_SET) == -1) {
                                           perror("fseek");
                                           font_data.clear();
                                           return font_data;
                                        }
                                        if (fread(&font_data[0], sizeof(uint8_t),
                                                  length, fp) != static_cast<unsigned long>(length)) {
                                           perror("fread");
                                           font_data.clear();
                                           return font_data;
                                        }
                                        fseek(fp, 0L, SEEK_SET);

                                        return font_data;
                                     }

                                     std::vector<uint8_t> font_embed_t::read_font_data(
                                                                                       const std::string &filename)
                                     {
                                        FILE *fp = fopen(filename.c_str(), "r");
                                        std::vector<uint8_t> font_data;

                                        if (fp == NULL) {
                                           perror("fopen");
                                           return font_data;
                                        }
                                        font_data = read_font_data(fp);
                                        fclose(fp);

                                        return font_data;
                                     }

                                     bool font_embed_t::parse_otf_cff_header(
                                                                             std::string &font_name, unsigned short &cid_encoding_id,
                                                                             unsigned int &cff_offset, unsigned int &cff_length,
                                                                             const std::vector<uint8_t> &font_data)
                                     {
                                        // OpenType file structure
                                        struct otf_offset_table_s {
                                           char sfnt_version[4];
                                           uint16_t num_tables;
                                           uint16_t search_range;
                                           uint16_t entry_selector;
                                           uint16_t range_shift;
                                        } offset_table;

                                        memcpy(&offset_table, &font_data[0],
                                               sizeof(struct otf_offset_table_s));
                                        if (strncmp(offset_table.sfnt_version, "OTTO", 4) != 0) {
                                           // Not a OpenType CFF/Type 2 font
                                           return false;
                                        }
#ifdef LITTLE_ENDIAN
                                        offset_table.num_tables = bswap_16(offset_table.num_tables);
#endif // LITTLE_ENDIAN

                                        bool name_table_exists = false;
                                        bool cff_table_exists = false;
                                        uint32_t name_offset = 0;

                                        for (uint16_t i = 0; i < offset_table.num_tables; i++) {
                                           struct otf_table_directory_s {
                                              char tag[4];
                                              uint32_t check_sum;
                                              uint32_t offset;
                                              uint32_t length;
                                           } table_directory;

                                           memcpy(&table_directory,
                                                  &font_data[sizeof(struct otf_offset_table_s) + i *
                                                             sizeof(struct otf_table_directory_s)],
                                                  sizeof(struct otf_table_directory_s));
#ifdef LITTLE_ENDIAN
                                           table_directory.offset =
                                           bswap_32(table_directory.offset);
                                           table_directory.length =
                                           bswap_32(table_directory.length);
#endif // LITTLE_ENDIAN
                                           if (strncmp(table_directory.tag, "name", 4) == 0) {
                                              name_offset = table_directory.offset;
                                              name_table_exists = true;
                                           }
                                           else if (strncmp(table_directory.tag, "CFF ", 4) == 0) {
                                              cff_offset = table_directory.offset;
                                              cff_length = table_directory.length;
                                              cff_table_exists = true;
                                           }
                                        }

                                        if (!(name_table_exists && cff_table_exists)) {
                                           return false;
                                        }

                                        // name

                                        struct otf_naming_table_header_s {
                                           uint16_t format;
                                           uint16_t count;
                                           uint16_t string_offset;
                                        } naming_table_header;

                                        memcpy(&naming_table_header, &font_data[name_offset],
                                               sizeof(struct otf_naming_table_header_s));
#ifdef LITTLE_ENDIAN
                                        naming_table_header.format =
                                        bswap_16(naming_table_header.format);
                                        naming_table_header.count =
                                        bswap_16(naming_table_header.count);
                                        naming_table_header.string_offset =
                                        bswap_16(naming_table_header.string_offset);
#endif // LITTLE_ENDIAN

                                        cid_encoding_id = 0xffffU;

                                        for (uint16_t i = 0; i < naming_table_header.count; i++) {
                                           struct otf_name_record_s {
                                              uint16_t platform_id;
                                              uint16_t encoding_id;
                                              uint16_t language_id;
                                              uint16_t name_id;
                                              uint16_t length;
                                              uint16_t offset;
                                           } name_record;
                                           const size_t base_offset = name_offset +
                                           sizeof(struct otf_naming_table_header_s);

                                           memcpy(&name_record,
                                                  &font_data[base_offset + i *
                                                             sizeof(struct otf_name_record_s)],
                                                  sizeof(struct otf_name_record_s));
#ifdef LITTLE_ENDIAN
                                           name_record.platform_id =
                                           bswap_16(name_record.platform_id);
                                           name_record.encoding_id =
                                           bswap_16(name_record.encoding_id);
                                           name_record.name_id = bswap_16(name_record.name_id);
#endif // LITTLE_ENDIAN
                                           if (name_record.platform_id == 1 &&
                                               name_record.encoding_id == 0 &&
                                               name_record.name_id == 6) {
                                              // Postscript name in Mac OS Roman
                                              //
                                              // The font name in Mac OS Roman encoding is
                                              // sufficient to obtain an ASCII PostScript name (and
                                              // is required by OpenType specification), while the
                                              // Windows platform uses a UCS-2 string that would
                                              // require conversion.
#ifdef LITTLE_ENDIAN
                                              name_record.length = bswap_16(name_record.length);
                                              name_record.offset = bswap_16(name_record.offset);
#endif // LITTLE_ENDIAN

                                              char *buffer = new char[name_record.length + 1];

                                              memcpy(buffer,
                                                     &font_data[name_offset +
                                                                naming_table_header.string_offset +
                                                                name_record.offset],
                                                     name_record.length);
                                              buffer[name_record.length] = '\0';
                                              font_name = buffer;

                                              delete [] buffer;
                                           }
                                           if (name_record.platform_id == 1 &&
                                               name_record.name_id == 20) {
                                              // PostScript CID findfont name
                                              //
                                              // encoding_id   Macintosh CMap
                                              // ---------------------------
                                              // 1         83pv-RKSJ-H
                                              // 2         B5pc-H
                                              // 3         KSCpc-EUC-H
                                              // 25         GBpc-EUC-H
                                              cid_encoding_id = name_record.encoding_id;
                                              // The actual Macintosh encoding CMap name is of no
                                              // further use. Note that Adobe currently only
                                              // actively maintains the Unicode based CMaps.
                                           }
                                        }

                                        return true;
                                     }

                                     bool font_embed_t::parse_ttf_header(
                                                                         std::string &font_name, double *font_bbox,
                                                                         std::map<wchar_t, uint16_t> &cid_map,
                                                                         std::vector<std::string> &char_strings,
                                                                         const std::vector<uint8_t> &font_data)
                                     {
                                        cid_map.clear();
                                        char_strings.clear();

                                        struct ttf_offset_table_s {
                                           fixed_t sfnt_version;
                                           uint16_t num_tables;
                                           uint16_t search_range;
                                           uint16_t entry_selector;
                                           uint16_t range_shift;
                                        } offset_table;

                                        memcpy(&offset_table, &font_data[0],
                                               sizeof(struct ttf_offset_table_s));
#ifdef LITTLE_ENDIAN
                                        offset_table.sfnt_version =
                                        bswap_32(offset_table.sfnt_version);
                                        offset_table.num_tables = bswap_16(offset_table.num_tables);
#endif // LITTLE_ENDIAN
                                        if (offset_table.sfnt_version != 0x00010000) {
                                           return false;
                                        }

                                        size_t name_offset = 0;
                                        //size_t name_length = 0;
                                        size_t head_offset = 0;
                                        //size_t head_length = 0;
                                        size_t cmap_offset = 0;
                                        //size_t cmap_length = 0;
                                        size_t post_offset = 0;
                                        //size_t post_length = 0;

                                        for (uint16_t i = 0; i < offset_table.num_tables; i++) {
                                           struct ttf_table_directory_s {
                                              char tag[4];
                                              uint32_t check_sum;
                                              uint32_t offset;
                                              uint32_t length;
                                           } table_directory;

                                           memcpy(&table_directory,
                                                  &font_data[sizeof(struct ttf_offset_table_s) + i *
                                                             sizeof(struct ttf_table_directory_s)],
                                                  sizeof(struct ttf_table_directory_s));
#ifdef LITTLE_ENDIAN
                                           table_directory.offset =
                                           bswap_32(table_directory.offset);
                                           table_directory.length =
                                           bswap_32(table_directory.length);
#endif // LITTLE_ENDIAN
#if 0
                                           fprintf(stderr, "%s:%d: tag = %c%c%c%c, offset = %u, "
                                                   "length = %u\n", __FILE__, __LINE__,
                                                   table_directory.tag[0], table_directory.tag[1],
                                                   table_directory.tag[2], table_directory.tag[3],
                                                   table_directory.offset, table_directory.length);
#endif
                                           if (strncmp(table_directory.tag, "name", 4) == 0) {
                                              name_offset = table_directory.offset;
                                              //name_length = table_directory.length;
                                           }
                                           else if (strncmp(table_directory.tag, "head", 4) == 0) {
                                              head_offset = table_directory.offset;
                                              //head_length = table_directory.length;
                                           }
                                           else if (strncmp(table_directory.tag, "cmap", 4) == 0) {
                                              cmap_offset = table_directory.offset;
                                              //cmap_length = table_directory.length;
                                           }
                                           else if (strncmp(table_directory.tag, "post", 4) == 0) {
                                              post_offset = table_directory.offset;
                                              //post_length = table_directory.length;
                                           }
                                        }

                                        // name

                                        struct ttf_naming_table_header_s {
                                           uint16_t format;
                                           uint16_t count;
                                           uint16_t string_offset;
                                        } naming_table_header;

                                        memcpy(&naming_table_header,
                                               &font_data[name_offset],
                                               sizeof(struct ttf_naming_table_header_s));
#ifdef LITTLE_ENDIAN
                                        naming_table_header.format =
                                        bswap_16(naming_table_header.format);
                                        naming_table_header.count =
                                        bswap_16(naming_table_header.count);
                                        naming_table_header.string_offset =
                                        bswap_16(naming_table_header.string_offset);
#endif // LITTLE_ENDIAN

                                        for (uint16_t i = 0; i < naming_table_header.count; i++) {
                                           struct ttf_name_record_s {
                                              uint16_t platform_id;
                                              uint16_t encoding_id;
                                              uint16_t language_id;
                                              uint16_t name_id;
                                              uint16_t length;
                                              uint16_t offset;
                                           } name_record;

                                           memcpy(
                                                  &name_record,
                                                  &font_data[name_offset +
                                                             sizeof(struct ttf_naming_table_header_s) +
                                                             i * sizeof(struct ttf_name_record_s)],
                                                  sizeof(struct ttf_name_record_s));
#ifdef LITTLE_ENDIAN
                                           name_record.platform_id =
                                           bswap_16(name_record.platform_id);
                                           name_record.encoding_id =
                                           bswap_16(name_record.encoding_id);
                                           name_record.name_id = bswap_16(name_record.name_id);
#endif // LITTLE_ENDIAN
       // the font name in mac os roman encoding is good enough
       // to obtain an ASCII post_script name, while the windows
       // platform uses a utF-16 string that would require
       // conversion.
                                           if (name_record.platform_id == 1 &&
                                               name_record.encoding_id == 0 &&
                                               name_record.name_id == 6) {
#ifdef LITTLE_ENDIAN
                                              name_record.length = bswap_16(name_record.length);
                                              name_record.offset = bswap_16(name_record.offset);
#endif // LITTLE_ENDIAN

                                              char *buffer = new char[name_record.length + 1];

                                              memcpy(buffer,
                                                     &font_data[name_offset +
                                                                naming_table_header.string_offset +
                                                                name_record.offset],
                                                     name_record.length * sizeof(char));
                                              buffer[name_record.length] = '\0';
                                              font_name = buffer;

                                              delete [] buffer;
                                           }
                                           else if (name_record.platform_id == 3 &&
                                                    name_record.encoding_id == 1 &&
                                                    name_record.name_id == 6) {
#ifdef LITTLE_ENDIAN
                                              name_record.length = bswap_16(name_record.length);
                                              name_record.offset = bswap_16(name_record.offset);
#endif // LITTLE_ENDIAN

                                              // Very ugly UCS-2 to ASCII conversion, but should
                                              // work for most font names
                                              char *buffer =
                                              new char[(name_record.length >> 1) + 1];

                                              for (uint16_t j = 0; j < (name_record.length >> 1);
                                                   j++) {
                                                 buffer[j] =
                                                 font_data[name_offset +
                                                           naming_table_header.string_offset +
                                                           name_record.offset + j * 2 + 1];
                                              }
                                              buffer[name_record.length >> 1] = '\0';
                                              font_name = buffer;

                                              delete [] buffer;
                                           }
                                        }

                                        // head

                                        struct ttf_head_table_s {
                                           fixed_t version;
                                           fixed_t font_revision;
                                           uint32_t check_sum_adjustment;
                                           uint32_t magic_number;
                                           uint16_t flags;
                                           uint16_t units_per_em;
                                           char created[8];
                                           char modified[8];
                                           int16_t x_min;
                                           int16_t y_min;
                                           int16_t x_max;
                                           int16_t y_max;
                                           uint16_t mac_style;
                                           uint16_t lowest_rec_ppem;
                                           int16_t font_direction_hint;
                                           int16_t index_to_loc_format;
                                           int16_t glyph_data_format;
                                        } head_table;

                                        memcpy(&head_table, &font_data[head_offset],
                                               sizeof(struct ttf_head_table_s));
#ifdef LITTLE_ENDIAN
                                        head_table.units_per_em = bswap_16(head_table.units_per_em);
                                        head_table.x_min = bswap_16(head_table.x_min);
                                        head_table.y_min = bswap_16(head_table.y_min);
                                        head_table.x_max = bswap_16(head_table.x_max);
                                        head_table.y_max = bswap_16(head_table.y_max);
#endif // LITTLE_ENDIAN

                                        font_bbox[0] =
                                        (double)head_table.x_min / head_table.units_per_em;
                                        font_bbox[1] =
                                        (double)head_table.y_min / head_table.units_per_em;
                                        font_bbox[2] =
                                        (double)head_table.x_max / head_table.units_per_em;
                                        font_bbox[3] =
                                        (double)head_table.y_max / head_table.units_per_em;

                                        // post

                                        struct ttf_post_script_table_s {
                                           fixed_t format_type;
                                           fixed_t italic_angle;
                                           int16_t underline_position;
                                           int16_t underline_thickness;
                                           uint32_t is_fixed_pitch;
                                           uint32_t min_mem_type42;
                                           uint32_t max_mem_type42;
                                           uint32_t min_mem_type1;
                                           uint32_t max_mem_type1;
                                        } post_script_table;

                                        memcpy(&post_script_table,
                                               &font_data[post_offset],
                                               sizeof(struct ttf_post_script_table_s));

#ifdef LITTLE_ENDIAN
                                        post_script_table.format_type =
                                        bswap_32(post_script_table.format_type);
                                        post_script_table.min_mem_type42 =
                                        bswap_32(post_script_table.min_mem_type42);
                                        post_script_table.max_mem_type42 =
                                        bswap_32(post_script_table.max_mem_type42);
#endif // LITTLE_ENDIAN

                                        size_t offset_current = post_offset;

#if 0
                                        if (post_script_table.format_type == 0x00010000) {
                                           // Exactly the 258 glyphs in the standard Macintosh glyph
                                           // set
                                        }
#endif
                                        if (post_script_table.format_type == 0x00020000) {
                                           // Version required by TrueType-based fonts to be used on
                                           // Windows
                                           //
                                           // numberOfGlyphs, glyphNameIndex[numGlyphs],
                                           // names[numberNewGlyphs] table

                                           uint16_t num_glyphs;

                                           memcpy(&num_glyphs,
                                                  &font_data[post_offset +
                                                             sizeof(struct ttf_post_script_table_s)],
                                                  sizeof(uint16_t));
#ifdef LITTLE_ENDIAN
                                           num_glyphs = bswap_16(num_glyphs);
#endif // LITTLE_ENDIAN

                                           uint16_t *glyph_name_index = new uint16_t[num_glyphs];

                                           memcpy(glyph_name_index,
                                                  &font_data[post_offset +
                                                             sizeof(struct ttf_post_script_table_s) +
                                                             sizeof(uint16_t)],
                                                  num_glyphs * sizeof(uint16_t));
#ifdef LITTLE_ENDIAN
                                           for (uint16_t i = 0; i < num_glyphs; i++) {
                                              glyph_name_index[i] = bswap_16(glyph_name_index[i]);
                                           }
#endif // LITTLE_ENDIAN

                                           size_t max_glyph_name_index = 0;
                                           for (int i = num_glyphs - 1; i >= 0; i--) {
                                              if (glyph_name_index[i] > max_glyph_name_index) {
                                                 max_glyph_name_index = glyph_name_index[i];
                                              }
                                           }

                                           std::string *glyph_name =
                                           new std::string[max_glyph_name_index - 258 + 1];

                                           offset_current +=
                                           sizeof(struct ttf_post_script_table_s) +
                                           (num_glyphs + 1) * sizeof(uint16_t);
                                           for (uint16_t i = 0; i <= max_glyph_name_index - 258; i++) {
                                              uint8_t length;

                                              memcpy(&length, &font_data[offset_current],
                                                     sizeof(uint8_t));
                                              offset_current += sizeof(uint8_t);

                                              char *buffer = new char[length + 1UL];

                                              memcpy(buffer, &font_data[offset_current],
                                                     length * sizeof(uint8_t));
                                              offset_current += length * sizeof(uint8_t);
                                              buffer[length] = '\0';
                                              glyph_name[i] = buffer;

                                              delete [] buffer;
                                           }

                                           char_strings.resize(num_glyphs);
#include "table/macintoshordering.h"
                                           for (uint16_t glyph = 0; glyph < num_glyphs; glyph++) {
                                              char_strings[glyph] = glyph_name_index[glyph] >= 258 ?
                                              glyph_name[glyph_name_index[glyph] - 258].c_str() :
                                              macintosh_ordering[glyph_name_index[glyph]];
                                           }

                                           delete [] glyph_name_index;
                                           delete [] glyph_name;
                                        }
                                        else if (post_script_table.format_type == 0x00030000) {
                                           // No PostScript name information is provided for the
                                           // glyphs

                                           // Do nothing, cid_map will be initialized with standard
                                           // Adobe glyph names once cmap is read
                                        }
                                        else {
                                           fprintf(stderr, "%s:%d: unsupported post table format "
                                                   "0x%08x\n", __FILE__, __LINE__,
                                                   post_script_table.format_type);

                                           return false;
                                        }
#if 0
                                        if (post_script_table.format_type == 0x00025000) {
                                           // Pure subset/simple reordering of the standard Macintosh
                                           // glyph set. Deprecated as of OpenType Specification v1.3
                                           //
                                           // numberOfGlyphs, offset[numGlyphs]
                                           return false;
                                        }
#endif

                                        // cmap

                                        struct ttf_mapping_table_s {
                                           uint16_t version;
                                           uint16_t num_encoding_tables;
                                        } mapping_table;

                                        memcpy(&mapping_table, &font_data[cmap_offset],
                                               sizeof(struct ttf_mapping_table_s));
#ifdef LITTLE_ENDIAN
                                        mapping_table.num_encoding_tables =
                                        bswap_16(mapping_table.num_encoding_tables);
#endif // LITTLE_ENDIAN

                                        uint32_t *subtable_offset =
                                        new uint32_t[mapping_table.num_encoding_tables];

                                        for (uint16_t i = 0;
                                             i < mapping_table.num_encoding_tables; i++) {
                                           struct ttf_encoding_table_s {
                                              uint16_t platform_id;
                                              uint16_t encoding_id;
                                              uint32_t offset;
                                           } encoding_table;

                                           memcpy(
                                                  &encoding_table,
                                                  &font_data[cmap_offset +
                                                             sizeof(struct ttf_mapping_table_s) +
                                                             i * sizeof(struct ttf_encoding_table_s)],
                                                  sizeof(struct ttf_encoding_table_s));
#ifdef LITTLE_ENDIAN
                                           encoding_table.platform_id =
                                           bswap_16(encoding_table.platform_id);
                                           encoding_table.encoding_id =
                                           bswap_16(encoding_table.encoding_id);
                                           encoding_table.offset = bswap_32(encoding_table.offset);
#endif // LITTLE_ENDIAN
                                           subtable_offset[i] = cmap_offset + encoding_table.offset;
                                        }

                                        int priority_max = 0;

                                        for (uint16_t i = 0;
                                             i < mapping_table.num_encoding_tables; i++) {
                                           struct ttf_encoding_subtable_common_s {
                                              uint16_t format;
                                              uint16_t length;
                                              uint16_t language;
                                           } encoding_subtable_common;

                                           memcpy(&encoding_subtable_common,
                                                  &font_data[subtable_offset[i]],
                                                  sizeof(struct ttf_encoding_subtable_common_s));
#ifdef LITTLE_ENDIAN
                                           encoding_subtable_common.format =
                                           bswap_16(encoding_subtable_common.format);
                                           encoding_subtable_common.length =
                                           bswap_16(encoding_subtable_common.length);
                                           encoding_subtable_common.language =
                                           bswap_16(encoding_subtable_common.language);
#endif // LITTLE_ENDIAN

                                           offset_current = subtable_offset[i] +
                                           sizeof(struct ttf_encoding_subtable_common_s);
#if 0
                                           fprintf(stderr, "%s:%d: encoding_subtable_common.format "
                                                   "= %hu\n", __FILE__, __LINE__,
                                                   encoding_subtable_common.format);
                                           fprintf(stderr, "%s:%d: encoding_subtable_common.length "
                                                   "= %hu\n", __FILE__, __LINE__,
                                                   encoding_subtable_common.length);
                                           fprintf(stderr, "%s:%d: encoding_subtable_common.language "
                                                   "= %hu\n", __FILE__, __LINE__,
                                                   encoding_subtable_common.language);
#endif

                                           int priority;

                                           switch(encoding_subtable_common.format) {
                                                 /////////////////////////////////////////////////////
                                                 // 8 and 16 bit mappings
                                                 // Priority range 1, 3..5 (2 reserved for format 13)
                                              case 0:
                                                 priority = 1;
                                                 // Byte encoding table
                                                 break;
                                              case 2:
                                                 // High-byte mapping through table
                                                 priority = 3;
                                                 break;
                                              case 4:
                                                 // Segment mapping to delta values
                                                 priority = 5;
#if 0
                                                 fprintf(stderr, "%s:%d: priority = %d, priority_max "
                                                         "= %d\n", __FILE__, __LINE__, priority,
                                                         priority_max);
#endif
                                                 if (priority_max <= priority) {
                                                    parse_ttf_encoding_subtable_format4(
                                                                                        cid_map, font_data, offset_current,
                                                                                        encoding_subtable_common.length);
                                                    priority_max = priority;
                                                 }
                                                 break;
                                              case 6:
                                                 // Trimmed table mapping
                                                 priority = 5;
                                                 break;
                                                 /////////////////////////////////////////////////////
                                                 // 32-bit mappings
                                                 // Priority range 6..9 (2 reserved for format 13)
                                              case 8:
                                                 // Mixed 16-bit and 32-bit coverage
                                                 priority = 6;
                                                 break;
                                              case 10:
                                                 // Trimmed array
                                                 priority = 6;
                                                 break;
                                              case 12:
                                                 // Segmented coverage
                                                 priority = 6;
                                                 break;
                                              case 13:
                                                 // Last resort font
                                                 priority = 2;
                                                 break;
                                              case 14:
                                                 // Unicode variation sequences
                                                 priority = 9;
                                                 break;
                                              default:
                                                 delete [] subtable_offset;
                                                 return false;
                                           }
                                        }

                                        delete [] subtable_offset;

                                        // Regenerate cid_map from the Adobe glyph list

                                        if (char_strings.empty() && !cid_map.empty()) {
                                           char_strings.resize(cid_map.size());
                                           for (std::map<wchar_t, uint16_t>::const_iterator iterator = cid_map.begin();
                                                iterator != cid_map.end(); ++iterator) {
                                              if (iterator->second < char_strings.size()) {
#include "table/adobeglyphlist.h"

                                                 const wchar_t *lower =
                                                 std::lower_bound(
                                                                  adobe_glyph_ucs,
                                                                  adobe_glyph_ucs + nadobe_glyph,
                                                                  iterator->first);
                                                 // The longest Adobe glyph name is 20 characters
                                                 // long (0x03b0 = upsilondieresistonos)
                                                 char buf[21];

                                                 if (iterator->first == L'\uffff') {
                                                    strncpy(buf, ".notdef", 8);
                                                 }
                                                 else if (lower < adobe_glyph_ucs + nadobe_glyph &&
                                                          *lower == iterator->first) {
                                                    const size_t index =
                                                    lower - adobe_glyph_ucs;

                                                    snprintf(buf, 21, "%s", adobe_glyph_name[index]);
                                                 }
                                                 else {
                                                    snprintf(buf, 21, "uni%04X", iterator->first);
                                                 }
                                                 char_strings[iterator->second] = buf;
                                              }
                                           }
                                        }

                                        return true;
                                     }

#if 0
                                     std::vector<uint8_t> font_embed_t::subset_otf(
                                                                                   const std::vector<uint8_t> &font_data,
                                                                                   const std::map<wchar_t, bool> &glyph_usage)
                                     {
                                        std::vector<uint8_t> retval;
                                        struct otf_offset_table_s {
                                           char sfnt_version[4];
                                           uint16_t num_tables;
                                           uint16_t search_range;
                                           uint16_t entry_selector;
                                           uint16_t range_shift;
                                        } offset_table;

                                        memcpy(&offset_table, &font_data[0],
                                               sizeof(struct otf_offset_table_s));
                                        if (strncmp(offset_table.sfnt_version, "OTTO", 4) != 0 ||
                                            strncmp(offset_table.sfnt_version, "\0\1\0\0", 4) != 0) {
                                           // Neither a OpenType, nor TrueType font
#if 0
                                           fprintf(stderr, "%s:%d: error: unknown sfnt_version = "
                                                   "0x%02x%02x%02x%02x\n", __FILE__, __LINE__,
                                                   offset_table.sfnt_version[0],
                                                   offset_table.sfnt_version[1],
                                                   offset_table.sfnt_version[2],
                                                   offset_table.sfnt_version[3]);
#endif
                                           return retval;
                                        }
#ifdef LITTLE_ENDIAN
                                        offset_table.num_tables = bswap_16(offset_table.num_tables);
#endif // LITTLE_ENDIAN

                                        struct otf_table_directory_s {
                                           char tag[4];
                                           uint32_t check_sum;
                                           uint32_t offset;
                                           uint32_t length;
                                        };
                                        struct table_data_s *table_data =
                                        new struct table_data_s[offset_table.num_tables];

                                        for (uint16_t i = 0; i < offset_table.num_tables; i++) {
                                           struct otf_table_directory_s table_directory;

                                           memcpy(&table_directory,
                                                  &font_data[sizeof(struct otf_offset_table_s) + i *
                                                             sizeof(struct otf_table_directory_s)],
                                                  sizeof(struct otf_table_directory_s));
#ifdef LITTLE_ENDIAN
                                           table_directory.offset =
                                           bswap_32(table_directory.offset);
                                           table_directory.length =
                                           bswap_32(table_directory.length);
#endif // LITTLE_ENDIAN
#if 0
                                           fprintf(stderr, "%s:%d: tag = %c%c%c%c, offset = %u, "
                                                   "length = %u\n", __FILE__, __LINE__,
                                                   table_directory.tag[0], table_directory.tag[1],
                                                   table_directory.tag[2], table_directory.tag[3],
                                                   table_directory.offset, table_directory.length);
#endif
                                           memcpy(table_data[i].tag, table_directory.tag,
                                                  4 * sizeof(char));
                                           table_data[i].data.resize(table_directory.length);
                                           memcpy(&(table_data[i].data[0]),
                                                  &font_data[table_directory.offset],
                                                  table_directory.length);
                                        }

                                        size_t size_count;

                                        size_count = sizeof(struct otf_offset_table_s) +
                                        offset_table.num_tables *
                                        sizeof(struct otf_table_directory_s);
                                        for (size_t i = 0; i < offset_table.num_tables; i++) {
                                           size_count += table_data[i].data.size();
                                        }

                                        size_t offset_current = sizeof(struct otf_offset_table_s);
                                        size_t offset_check_sum_adjustment = 0;
                                        bool head_table_exists = false;

                                        retval.resize(size_count);
                                        memcpy(&retval[0], &font_data[0],
                                               sizeof(struct otf_offset_table_s));
                                        for (size_t i = 0; i < offset_table.num_tables; i++) {
                                           struct otf_table_directory_s table_directory;
                                           const bool head_table =
                                           strncmp(table_directory.tag, "head", 4) == 0;

                                           memcpy(table_directory.tag, table_data[i].tag,
                                                  4 * sizeof(char));
                                           if (head_table) {
                                              // Reset checkSumAdjustment in order to calculate the
                                              // check sum of the head table
                                              offset_check_sum_adjustment = 2 * sizeof(fixed_t);
                                              *reinterpret_cast<uint32_t *>(&(table_data[i].data[
                                                                                                 offset_check_sum_adjustment])) = 0U;
                                              // Change the offset for checkSumAdjustment to the
                                              // global indexing
                                              offset_check_sum_adjustment += offset_current;
                                              head_table_exists = true;
                                           }
                                           table_directory.check_sum =
                                           otf_check_sum(table_data[i].data);
                                           table_directory.offset = size_count;
                                           table_directory.length = table_data[i].data.size();

                                           memcpy(&retval[offset_current], &table_directory,
                                                  sizeof(struct otf_table_directory_s));
                                           size_count += table_data[i].data.size();
                                           offset_current += sizeof(struct otf_table_directory_s);
                                        }

                                        for (size_t i = 0; i < offset_table.num_tables; i++) {
                                           memcpy(&retval[offset_current], &(table_data[i].data[0]),
                                                  table_data[i].data.size());
                                           offset_current += table_data[i].data.size();

                                           const size_t padding_size =
                                           (4U - table_data[i].data.size()) & 3U;

                                           memset(&retval[offset_current], '\0', padding_size);
                                           offset_current += padding_size;
                                        }

                                        if (head_table_exists) {
                                           // Set checkSumAdjustment in the head table
                                           *reinterpret_cast<uint32_t *>(&(retval[
                                                                                  offset_check_sum_adjustment])) =
                                           0xb1b0afbaU - otf_check_sum(retval);
                                        }

                                        return retval;
                                     }
#endif

         }
