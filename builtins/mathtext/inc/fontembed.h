// mathtext - A TeX/LaTeX compatible rendering library. Copyright (C)
// 2008-2016 Yue Shi Lai <ylai@users.sourceforge.net>
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

#include <vector>
#include <string>
#include <map>
#include <cstdio>
#include <stdint.h>

namespace mathtext {

	class font_embed_t {
	protected:
		struct table_data_s {
			char tag[4];
			std::vector<uint8_t> data;
		};
		static void protected_memcpy(
			void *destination,
			std::vector<uint8_t>::const_iterator source,
			std::vector<uint8_t>::const_iterator source_end,
			size_t length, const char *location);
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
			const size_t offset);
		static void parse_ttf_encoding_subtable_format12(
			std::map<wchar_t, uint16_t> &cid_map,
			const std::vector<uint8_t> &font_data,
			const size_t offset);
		static unsigned int otf_check_sum(
			const std::vector<unsigned char> &table_data);
	public:
		// I/O
		static std::vector<unsigned char> read_font_data(FILE *);
		static std::vector<unsigned char> read_font_data(
			const std::string &filename);
		// Font parsing
		// OTF/TTF tables
		static std::map<std::string, std::pair<uint32_t, uint32_t> >
		parse_ttf_offset_table(
			const std::vector<uint8_t> &font_data,
			size_t offset_table_size, uint16_t num_tables);
		static void parse_ttf_cmap(
			std::map<wchar_t, uint16_t> &cid_map,
			const std::vector<uint8_t> &font_data,
			uint32_t cmap_offset);
		static void parse_ttf_head(
			double *font_bbox, uint16_t &units_per_em,
			const std::vector<uint8_t> &font_data,
			uint32_t head_offset);
		static uint16_t parse_ttf_hhea(
			const std::vector<uint8_t> &font_data,
			uint32_t hhea_offset);
		static std::vector<uint16_t> parse_ttf_hmtx(
			const std::vector<uint8_t> &font_data,
			uint32_t hmtx_offset, uint16_t number_of_h_metrics);
		static uint16_t parse_ttf_maxp(
			const std::vector<uint8_t> &font_data,
			uint32_t maxp_offset);
		static void parse_ttf_name(
			std::string &font_name, uint16_t &cid_encoding_id,
			const std::vector<uint8_t> &font_data,
			uint32_t name_offset);
		static void parse_ttf_os_2(
			uint32_t &font_descriptor_flag, double &ascent,
			double &descent, double &leading, double &cap_height,
			double &x_height, double &stem_v, double &avg_width,
			const std::vector<uint8_t> &font_data,
			uint32_t os_2_offset, uint16_t units_per_em);
		static void parse_ttf_post(
			std::vector<std::string> &charset,
			double &italic_angle, uint32_t &font_descriptor_flags,
			const std::vector<uint8_t> &font_data,
			uint32_t name_offset);
		// CFF parsing
		static std::vector<std::vector<uint8_t> > parse_cff_index(
			const std::vector<uint8_t> &font_data,
			uint32_t &current_offset, uint32_t *skip_psize = NULL);
		static double parse_cff_dict_number(
			std::vector<uint8_t>::const_iterator &data,
			std::vector<uint8_t>::const_iterator end);
		static double search_cff_top_dict_number(
			const std::vector<uint8_t> &top_dict,
			uint8_t operator_value_1, uint8_t operator_value_2 = 255);
		static std::string cff_sid_to_string(
			uint16_t sid,
			const std::vector<std::vector<uint8_t> > &string_index);
		static std::vector<std::string> parse_cff_charset(
			const std::vector<uint8_t> &data,
			uint32_t charset_offset, uint32_t cff_nglyph,
			const std::vector<std::vector<uint8_t> > &string_index);
		static void parse_cff(
			std::vector<std::string> &charset,
			const std::vector<uint8_t> &font_data,
			uint32_t cff_offset);
		// OTF CFF parsing interface
		static bool parse_otf_cff_header(
			std::string &font_name, unsigned short &cid_encoding_id,
			uint32_t font_descriptor_flags, double *font_bbox,
			double &italic_angle, double &ascent, double &descent,
			double &leading, double &cap_height, double &x_height,
			double &stem_v,double &avg_width,
			std::map<wchar_t, uint16_t> &cid_map,
			std::vector<std::string> &charset,
			std::vector<uint16_t> &advance_width,
			unsigned int &cff_offset, unsigned int &cff_length,
			const std::vector<uint8_t> &font_data);
		static bool parse_otf_cff_header(
			std::string &font_name, unsigned short &cid_encoding_id,
			unsigned int &cff_offset, unsigned int &cff_length,
			const std::vector<unsigned char> &font_data);
		static std::vector<std::string>
			charset_from_adobe_glyph_list(
			std::map<wchar_t, uint16_t> &cid_map);
		static bool parse_ttf_header(
			std::string &font_name, unsigned short &cid_encoding_id,
			uint32_t font_descriptor_flags, double *font_bbox,
			double &italic_angle, double &ascent, double &descent,
			double &leading, double &cap_height, double &x_height,
			double &stem_v,double &avg_width,
			std::map<wchar_t, uint16_t> &cid_map,
			std::vector<std::string> &charset,
			std::vector<uint16_t> &advance_width,
			const std::vector<uint8_t> &font_data);
		static bool parse_ttf_header(
			std::string &font_name, double *font_bbox,
			std::map<wchar_t, uint16_t> &cid_map,
			std::vector<std::string> &charset,
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
	protected:
		struct cid_char_s {
			uint32_t code;
			uint16_t cid;
		};
		struct cid_range_s {
			uint32_t code_first;
			uint32_t code_last;
			uint16_t cid;
		};
		static uint16_t utf16_high_surrogate(uint32_t code);
		static uint16_t utf16_low_surrogate(uint32_t code);
		static std::string utf16be_str(uint32_t code);
		static std::string uint16_str(uint16_t code);
		static void add_cidfont_w_token(std::string &s, const std::string &t);
		static std::string cidfont_w(
			const std::vector<uint16_t> &advance_width);
		static std::string pdf_vector_differences(
			const std::string &key,
			const std::map<wchar_t, uint16_t> &cid_map,
			const std::vector<std::string> &charset);
		static std::map<std::string, std::string>
		font_embed_cid(
			std::string &font_name,
			const std::vector<unsigned char> &font_data,
			unsigned int type);
	public:
		static std::map<std::string, std::string>
		font_embed_type_2(
			std::string &font_name,
			const std::vector<unsigned char> &font_data);
		static std::map<std::string, std::string>
		font_embed_type_42(
			std::string &font_name,
			const std::vector<unsigned char> &font_data);
	};

	class font_embed_svg_t : public font_embed_t {
		static std::string font_embed_svg(
			std::string &font_name,
			const std::vector<unsigned char> &font_data);
	};

}
