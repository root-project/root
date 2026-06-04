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

#include <mathtext/fontembed.h>
#include <algorithm>
#include <cstring>
#include <cstdio>
#include <cmath>
#include <byteswap.h>

// References:
//
// Adobe Systems, Inc. and Microsoft Corp., OpenType specification
// (2002), version 1.4.
//
// Apple Computer, Inc., TrueType reference ranual (2002)
//
// Microsoft Corp., TrueType 1.0 font files: technical specification
// (1995), version 1.66

#define ERROR_ACCESS(t) \
	(fprintf(stderr, "%s:%d: error: access out of bound in %s\n", \
			 __FILE__, __LINE__, t));
#define ERROR_UNSUPPORTED(t) \
	(fprintf(stderr, "%s:%d: error: %s is not supported\n", \
			 __FILE__, __LINE__, t));

namespace mathtext {

	typedef int32_t fixed_t;

	void font_embed_t::protected_memcpy(
		void *destination,
		std::vector<uint8_t>::const_iterator source,
		std::vector<uint8_t>::const_iterator source_end,
		size_t length, const char *location)
	{
		if (source + length > source_end) {
			ERROR_ACCESS(location);
		}
		memcpy(destination, &source[0], length);
	}

	void font_embed_t::parse_ttf_encoding_subtable_format4(
		std::map<wchar_t, uint16_t> &cid_map,
		const std::vector<uint8_t> &font_data, const size_t offset)
	{
		static const char *location =
			"TrueType encoding table format 4";

		cid_map.clear();

		size_t offset_current = offset;

		struct ttf_encoding_subtable_format4_s {
			uint16_t format;
			uint16_t length;
			uint16_t language;
			uint16_t seg_count_x2;
			uint16_t search_range;
			uint16_t entry_selector;
			uint16_t range_shift;
		} encoding_subtable_format4;

		protected_memcpy(&encoding_subtable_format4,
						 font_data.begin() + offset_current,
						 font_data.end(),
						 sizeof(struct ttf_encoding_subtable_format4_s),
						 location);
		offset_current +=
			sizeof(struct ttf_encoding_subtable_format4_s);
#ifdef LITTLE_ENDIAN
		encoding_subtable_format4.length =
			bswap_16(encoding_subtable_format4.length);
		encoding_subtable_format4.seg_count_x2 =
			bswap_16(encoding_subtable_format4.seg_count_x2);
#endif // LITTLE_ENDIAN

		const uint16_t seg_count =
			encoding_subtable_format4.seg_count_x2 >> 1;
		uint16_t *end_code = new uint16_t[seg_count];

		protected_memcpy(end_code,
						 font_data.begin() + offset_current,
						 font_data.end(),
						 seg_count * sizeof(uint16_t),
						 location);
		offset_current += seg_count * sizeof(uint16_t);
#ifdef LITTLE_ENDIAN
		for (uint16_t segment = 0; segment < seg_count; segment++) {
			end_code[segment] = bswap_16(end_code[segment]);
		}
#endif // LITTLE_ENDIAN

		uint16_t reserved_pad;

		protected_memcpy(&reserved_pad,
						 font_data.begin() + offset_current,
						 font_data.end(), sizeof(uint16_t),
						 location);
		offset_current += sizeof(uint16_t);

		uint16_t *start_code = new uint16_t[seg_count];

		protected_memcpy(start_code,
						 font_data.begin() + offset_current,
						 font_data.end(),
						 seg_count * sizeof(uint16_t), location);
		offset_current += seg_count * sizeof(uint16_t);
#ifdef LITTLE_ENDIAN
		for (uint16_t segment = 0; segment < seg_count; segment++) {
			start_code[segment] = bswap_16(start_code[segment]);
		}
#endif // LITTLE_ENDIAN

		uint16_t *id_delta = new uint16_t[seg_count];

		protected_memcpy(id_delta,
						 font_data.begin() + offset_current,
						 font_data.end(),
						 seg_count * sizeof(uint16_t), location);
		offset_current += seg_count * sizeof(uint16_t);
#ifdef LITTLE_ENDIAN
		for (uint16_t segment = 0; segment < seg_count; segment++) {
			id_delta[segment] = bswap_16(id_delta[segment]);
		}
#endif // LITTLE_ENDIAN

		const uint16_t variable =
			(encoding_subtable_format4.length >> 1) -
			(seg_count << 2) - 8;
		uint16_t *id_range_offset =
			new uint16_t[seg_count + variable];

		protected_memcpy(id_range_offset,
						 font_data.begin() + offset_current,
						 font_data.end(),
						 (seg_count + variable) * sizeof(uint16_t),
						 location);
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
				const uint16_t glyph_id =
					id_range_offset[segment] == 0 ?
					id_delta[segment] + code :
					inner_offset >= seg_count + variable ?
					0 : id_range_offset[inner_offset];

				cid_map[static_cast<wchar_t>(code)] = glyph_id;
			}
		}

		delete [] end_code;
		delete [] start_code;
		delete [] id_delta;
		delete [] id_range_offset;
	}

	void font_embed_t::parse_ttf_encoding_subtable_format12(
		std::map<wchar_t, uint16_t> &cid_map,
		const std::vector<uint8_t> &font_data, const size_t offset)
	{
		static const char *location =
			"TrueType encoding table format 12";

		cid_map.clear();

		size_t offset_current = offset;

		struct ttf_encoding_subtable_format12_s {
			uint16_t format;
			uint16_t reserved;
			uint32_t length;
			uint32_t language;
			uint32_t ngroups;
		} encoding_subtable_format12;

		protected_memcpy(&encoding_subtable_format12,
						 font_data.begin() + offset_current,
						 font_data.end(),
						 sizeof(struct ttf_encoding_subtable_format12_s),
						 location);
		offset_current +=
			sizeof(struct ttf_encoding_subtable_format12_s);
#ifdef LITTLE_ENDIAN
		encoding_subtable_format12.ngroups =
			bswap_32(encoding_subtable_format12.ngroups);
#endif // LITTLE_ENDIAN

		struct ttf_encoding_subtable_format12_group_s {
			uint32_t start_char_code;
			uint32_t end_char_code;
			uint32_t start_glyph_id;
		} encoding_subtable_format12_group;

		for (uint32_t group = 0;
			 group < encoding_subtable_format12.ngroups; group++) {
			protected_memcpy(
				&encoding_subtable_format12_group,
				font_data.begin() + offset_current,
				font_data.end(),
				sizeof(struct ttf_encoding_subtable_format12_group_s),
				location);
			offset_current +=
				sizeof(struct ttf_encoding_subtable_format12_group_s);
#ifdef LITTLE_ENDIAN
			encoding_subtable_format12_group.start_char_code =
				bswap_32(encoding_subtable_format12_group.start_char_code);
			encoding_subtable_format12_group.end_char_code =
				bswap_32(encoding_subtable_format12_group.end_char_code);
			encoding_subtable_format12_group.start_glyph_id =
				bswap_32(encoding_subtable_format12_group.start_glyph_id);
#endif // LITTLE_ENDIAN
			for (uint32_t code =
					 encoding_subtable_format12_group.start_char_code,
					 glyph_id =
					 encoding_subtable_format12_group.start_glyph_id;
				 code <= encoding_subtable_format12_group.end_char_code;
				 code++, glyph_id++) {
				cid_map[static_cast<wchar_t>(code)] = glyph_id;
			}
		}
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

			data = std::vector<uint8_t>(input_data_data, input_data_data);
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

			protected_memcpy(
				&table_directory,
				font_data.begin() +
				sizeof(struct otf_offset_table_s) + i *
				sizeof(struct otf_table_directory_s),
				font_data.end(),
				sizeof(struct otf_table_directory_s),
				location);
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
#endif //////////////////////////////////////////////////////////////

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

	std::vector<uint8_t> font_embed_t::read_font_data(FILE *fp)
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

	std::map<std::string, std::pair<uint32_t, uint32_t> >
	font_embed_t::parse_ttf_offset_table(
		const std::vector<uint8_t> &font_data,
		size_t offset_table_size, uint16_t num_tables)
	{
		std::map<std::string, std::pair<uint32_t, uint32_t> > table;

		for (uint16_t i = 0; i < num_tables; i++) {
			struct ttf_table_directory_s {
				char tag[4];
				uint32_t check_sum;
				uint32_t offset;
				uint32_t length;
			} table_directory;

			if (!(offset_table_size + (i + 1) *
				  sizeof(struct ttf_table_directory_s) <=
				  font_data.size())) {
				ERROR_ACCESS("table directory");
				continue;
			}
			memcpy(&table_directory,
				   &font_data[offset_table_size + i *
							  sizeof(struct ttf_table_directory_s)],
				   sizeof(struct ttf_table_directory_s));
#ifdef LITTLE_ENDIAN
			table_directory.offset =
				bswap_32(table_directory.offset);
			table_directory.length =
				bswap_32(table_directory.length);
#endif // LITTLE_ENDIAN
			if (!(table_directory.offset + table_directory.length <=
				  font_data.size())) {
				ERROR_ACCESS("table directory");
				continue;
			}
			table[std::string(table_directory.tag,
							  table_directory.tag + 4)] =
				std::pair<uint32_t, uint32_t>(
					table_directory.offset,
					table_directory.length);
		}

		return table;
	}

	void font_embed_t::parse_ttf_cmap(
		std::map<wchar_t, uint16_t> &cid_map,
		const std::vector<uint8_t> &font_data,
		uint32_t cmap_offset)
	{
		if (cmap_offset == 0) {
			return;
		}

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

			size_t offset_current = subtable_offset[i];
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
						cid_map, font_data, offset_current);
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
				if (priority_max <= priority) {
					parse_ttf_encoding_subtable_format12(
						cid_map, font_data, offset_current);
					priority_max = priority;
				}
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
				return;
			}
		}

		delete [] subtable_offset;
	}

	void font_embed_t::parse_ttf_head(
		double *font_bbox, uint16_t &units_per_em,
		const std::vector<uint8_t> &font_data,
		uint32_t head_offset)
	{
		if (head_offset == 0) {
			return;
		}

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

		if (!(head_offset + sizeof(struct ttf_head_table_s) <=
			  font_data.size())) {
			ERROR_ACCESS("head table");
			return;
		}
		memcpy(&head_table, &font_data[head_offset],
			   sizeof(struct ttf_head_table_s));
#ifdef LITTLE_ENDIAN
		head_table.units_per_em = bswap_16(head_table.units_per_em);
		head_table.x_min = bswap_16(head_table.x_min);
		head_table.y_min = bswap_16(head_table.y_min);
		head_table.x_max = bswap_16(head_table.x_max);
		head_table.y_max = bswap_16(head_table.y_max);
#endif // LITTLE_ENDIAN

		units_per_em = head_table.units_per_em;
		font_bbox[0] = head_table.x_min * 1000.0 / units_per_em;
		font_bbox[1] = head_table.y_min * 1000.0 / units_per_em;
		font_bbox[2] = head_table.x_max * 1000.0 / units_per_em;
		font_bbox[3] = head_table.y_max * 1000.0 / units_per_em;
	}

	uint16_t font_embed_t::parse_ttf_hhea(
		const std::vector<uint8_t> &font_data,
		uint32_t hhea_offset)
	{
		if (!(hhea_offset + 36 <= font_data.size())) {
			ERROR_ACCESS("hhea table");
		}

		// We don't care about anything other than numberOfHMetrics, which
		// sits at a common position for both versions 0.5 (OTF CFF)
		// and 1.0 (TTF).

		uint16_t number_of_h_metrics;

		memcpy(&number_of_h_metrics, &font_data[hhea_offset + 34], 2);
#ifdef LITTLE_ENDIAN
		number_of_h_metrics = bswap_16(number_of_h_metrics);
#endif // LITTLE_ENDIAN

		return number_of_h_metrics;
	}

	std::vector<uint16_t> font_embed_t::parse_ttf_hmtx(
		const std::vector<uint8_t> &font_data,
		uint32_t hmtx_offset, uint16_t number_of_h_metrics)
	{
		std::vector<uint16_t> advance_width;

		advance_width.resize(number_of_h_metrics);
		for (size_t i = 0; i < number_of_h_metrics; i++) {
			memcpy(&advance_width[i], &font_data[hmtx_offset + 4 * i], 2);
#ifdef LITTLE_ENDIAN
			advance_width[i] = bswap_16(advance_width[i]);
#endif // LITTLE_ENDIAN
		}

		return advance_width;
	}

	uint16_t font_embed_t::parse_ttf_maxp(
		const std::vector<uint8_t> &font_data,
		uint32_t maxp_offset)
	{
		if (!(maxp_offset + 6 <= font_data.size())) {
			ERROR_ACCESS("maxp table");
		}

		// We don't care about anything other than numGlyphs, which
		// sits at a common position for both versions 0.5 (OTF CFF)
		// and 1.0 (TTF).

		uint16_t num_glyphs;

		memcpy(&num_glyphs, &font_data[maxp_offset + 4], 2);
#ifdef LITTLE_ENDIAN
		num_glyphs = bswap_16(num_glyphs);
#endif // LITTLE_ENDIAN

		return num_glyphs;
	}

	void font_embed_t::parse_ttf_name(
		std::string &font_name, uint16_t &cid_encoding_id,
		const std::vector<uint8_t> &font_data,
		uint32_t name_offset)
	{
		if (name_offset == 0) {
			return;
		}

		struct ttf_naming_table_header_s {
			uint16_t format;
			uint16_t count;
			uint16_t string_offset;
		} naming_table_header;

		if (!(name_offset +
			  sizeof(struct ttf_naming_table_header_s) <=
			  font_data.size())) {
			ERROR_ACCESS("name table");
			return;
		}

		memcpy(&naming_table_header, &font_data[name_offset],
			   sizeof(struct ttf_naming_table_header_s));
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
			struct ttf_name_record_s {
				uint16_t platform_id;
				uint16_t encoding_id;
				uint16_t language_id;
				uint16_t name_id;
				uint16_t length;
				uint16_t offset;
			} name_record;
			const size_t base_offset = name_offset +
				sizeof(struct ttf_naming_table_header_s);

			if (!(base_offset + (i + 1) *
				  sizeof(struct ttf_name_record_s) <=
				  font_data.size())) {
				ERROR_ACCESS("name table");
				continue;
			}
			memcpy(&name_record,
				   &font_data[base_offset + i *
							  sizeof(struct ttf_name_record_s)],
				   sizeof(struct ttf_name_record_s));
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
				if (!(name_offset +
					  naming_table_header.string_offset +
					  name_record.offset + name_record.length <
					  font_data.size())) {
					ERROR_ACCESS("name table");
					continue;
				}

				const char *p = reinterpret_cast<const char *>(
					&font_data[name_offset +
							   naming_table_header.string_offset +
							   name_record.offset]);

				font_name = std::string(p, p + name_record.length);
			}
			else if (name_record.platform_id == 3 &&
					 name_record.encoding_id == 1 &&
					 name_record.name_id == 6) {
#ifdef LITTLE_ENDIAN
				name_record.length = bswap_16(name_record.length);
				name_record.offset = bswap_16(name_record.offset);
#endif // LITTLE_ENDIAN
				if (!(name_offset +
					  naming_table_header.string_offset +
					  name_record.offset + name_record.length <
					  font_data.size())) {
					ERROR_ACCESS("name table");
					continue;
				}
				// Very ugly UCS-2 to ASCII conversion, but should
				// work for most font names
				font_name.resize(name_record.length >> 1);

				for (uint16_t j = 0; j < (name_record.length >> 1);
					 j++) {
					font_name[j] =
						font_data[name_offset +
								  naming_table_header.string_offset +
								  name_record.offset + j * 2 + 1];
				}
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
	}

	void font_embed_t::parse_ttf_os_2(
		uint32_t &font_descriptor_flag, double &ascent,
		double &descent, double &leading, double &cap_height,
		double &x_height, double &stem_v, double &avg_width,
		const std::vector<uint8_t> &font_data,
		uint32_t os_2_offset, uint16_t units_per_em)
	{
		if (os_2_offset == 0) {
			return;
		}

		struct ttf_os_2_table_s {
			uint16_t version;
			int16_t x_avg_char_width;
			uint16_t us_weight_class;
			uint16_t us_width_class;
			uint16_t fs_type;
			int16_t y_subscript_x_size;
			int16_t y_subscript_y_size;
			int16_t y_subscript_x_offset;
			int16_t y_subscript_y_offset;
			int16_t y_superscript_x_size;
			int16_t y_superscript_y_size;
			int16_t y_superscript_x_offset;
			int16_t y_superscript_y_offset;
			int16_t y_strikeout_size;
			int16_t y_strikeout_position;
			int16_t s_family_class;
			uint8_t panose[10];
			uint32_t ul_unicode_range1;
			uint32_t ul_unicode_range2;
			uint32_t ul_unicode_range3;
			uint32_t ul_unicode_range4;
			char ach_vend_id[4];
			uint16_t fs_selection;
			uint16_t us_first_char_index;
			uint16_t us_last_char_index;
			int16_t s_typo_ascender;
			int16_t s_typo_descender;
			int16_t s_typo_line_gap;
			uint16_t us_win_ascent;
			uint16_t us_win_descent;
			uint32_t ul_code_page_range1;
			uint32_t ul_code_page_range2;
			int16_t s_x_height;
			int16_t s_cap_height;
			uint16_t us_default_char;
			uint16_t us_break_char;
			uint16_t us_max_context;
			uint16_t us_lower_optical_point_size;
			uint16_t us_upper_optical_point_size;
		} os_2_table;

		if (!(os_2_offset + sizeof(struct ttf_os_2_table_s) <=
			  font_data.size())) {
			ERROR_ACCESS("OS/2 table");
			return;
		}
		memcpy(&os_2_table, &font_data[os_2_offset],
			   sizeof(struct ttf_os_2_table_s));
#ifdef LITTLE_ENDIAN
		os_2_table.x_avg_char_width =
			bswap_16(os_2_table.x_avg_char_width);
		os_2_table.s_family_class =
			bswap_16(os_2_table.s_family_class);
		os_2_table.s_typo_ascender =
			bswap_16(os_2_table.s_typo_ascender);
		os_2_table.s_typo_descender =
			bswap_16(os_2_table.s_typo_descender);
		os_2_table.s_typo_line_gap =
			bswap_16(os_2_table.s_typo_line_gap);
		os_2_table.s_x_height = bswap_16(os_2_table.s_x_height);
		os_2_table.s_cap_height = bswap_16(os_2_table.s_cap_height);
#endif // LITTLE_ENDIAN

		switch (os_2_table.s_family_class >> 8) {
		case 1:
		case 2:
		case 3:
		case 4:
		case 5:
		case 7:
			// Serif
			font_descriptor_flag |= (1U << 1);
			break;
		case 8:
			// Sans serif
			font_descriptor_flag &= ~(1U << 1);
			break;
		case 10:
			// Script
			font_descriptor_flag |= (1U << 3);
			break;
		case 12:
			// Symbolic
			font_descriptor_flag |= (1U << 2);
			// Nonsymbolic
			font_descriptor_flag &= ~(1U << 2);
			break;
		}
		if (os_2_table.fs_selection & (1U << 0) ||
			os_2_table.fs_selection & (1U << 9)) {
			// Italic (or oblique)
			font_descriptor_flag |= (1U << 6);
		}
		ascent = os_2_table.s_typo_ascender * 1000.0 / units_per_em;
		descent = os_2_table.s_typo_descender * 1000.0 / units_per_em;
		leading = os_2_table.s_typo_line_gap * 1000.0 / units_per_em;
		cap_height = os_2_table.s_cap_height * 1000.0 / units_per_em;
		x_height = os_2_table.s_x_height * 1000.0 / units_per_em;

		// The PDFLib values
		static const uint16_t weight_class_stem_v[] = {
			50, 71, 109, 125, 135, 165, 201, 241
		};

		stem_v = weight_class_stem_v[
			std::max(0, std::min(7,
				os_2_table.us_weight_class / 100)) - 3];
		avg_width = os_2_table.x_avg_char_width * 1000.0 /
			units_per_em;
	}

	void font_embed_t::parse_ttf_post(
		std::vector<std::string> &charset,
		double &italic_angle, uint32_t &font_descriptor_flags,
		const std::vector<uint8_t> &font_data,
		uint32_t post_offset)
	{
		if (post_offset == 0) {
			return;
		}

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

		if (!(post_offset +
			  sizeof(struct ttf_post_script_table_s) <=
			  font_data.size())) {
			ERROR_ACCESS("post table");
			return;
		}

		memcpy(&post_script_table,
			   &font_data[post_offset],
			   sizeof(struct ttf_post_script_table_s));

#ifdef LITTLE_ENDIAN
		post_script_table.format_type =
			bswap_32(post_script_table.format_type);
		post_script_table.italic_angle =
			bswap_32(post_script_table.italic_angle);
		post_script_table.is_fixed_pitch =
			bswap_32(post_script_table.is_fixed_pitch);
		post_script_table.min_mem_type42 =
			bswap_32(post_script_table.min_mem_type42);
		post_script_table.max_mem_type42 =
			bswap_32(post_script_table.max_mem_type42);
#endif // LITTLE_ENDIAN

		italic_angle = post_script_table.italic_angle *
			(1.0 / (1U << 16));
		if (post_script_table.italic_angle != 0) {
			font_descriptor_flags |= (1U << 6);
		}
		if (post_script_table.is_fixed_pitch != 0) {
			font_descriptor_flags |= (1U << 0);
		}

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

			charset.resize(num_glyphs);
#include "table/macintoshordering.h"
			for (uint16_t glyph = 0; glyph < num_glyphs; glyph++) {
				charset[glyph] = glyph_name_index[glyph] >= 258 ?
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

			return;
		}
#if 0
		if (post_script_table.format_type == 0x00025000) {
			// Pure subset/simple reordering of the standard Macintosh
			// glyph set. Deprecated as of OpenType Specification v1.3
			//
			// numberOfGlyphs, offset[numGlyphs]
			return;
		}
#endif
	}

	std::vector<std::vector<uint8_t> >
	font_embed_t::parse_cff_index(
		const std::vector<uint8_t> &font_data,
		uint32_t &current_offset, uint32_t *skip)
	{
		struct cff_index_s {
			uint16_t count;
			uint8_t off_size;
			std::vector<uint32_t> offset;
			std::vector<std::vector<uint8_t> > data;
		} cff_index;

		memcpy(&cff_index.count, &font_data[current_offset], 2);
#ifdef LITTLE_ENDIAN
		cff_index.count = bswap_16(cff_index.count);
#endif // LITTLE_ENDIAN
		current_offset += 2;
		if (!(cff_index.count > 0)) {
			return cff_index.data;
		}

		cff_index.off_size = font_data[current_offset];
		current_offset++;

		for (size_t i = 0; i < cff_index.count + 1U; i++) {
			cff_index.offset.push_back(font_data[current_offset]);
			current_offset++;
			for (size_t j = 1; j < cff_index.off_size; j++) {
				cff_index.offset.back() <<= 8;
				cff_index.offset.back() |= font_data[current_offset];
				current_offset++;
			}
		}
		if (skip == NULL) {
			for (size_t i = 0; i < cff_index.count; i++) {
				cff_index.data.push_back(std::vector<uint8_t>(
					&font_data[current_offset +
							   cff_index.offset[i] - 1],
					&font_data[current_offset +
							   cff_index.offset[i + 1] - 1]));
			}
		}
		else {
			*skip = cff_index.count;
		}
		current_offset += cff_index.offset.back() - 1;

		return cff_index.data;
	}

	double font_embed_t::parse_cff_dict_number(
		std::vector<uint8_t>::const_iterator &data,
		std::vector<uint8_t>::const_iterator end)
	{
		const char *nibble_chr[16] = {
			"0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
			".", "E", "E-", "", "-", ""
		};
		std::string nibble_str;
		uint8_t nibble = 0;
		bool nibble_high = true;
		double ret = NAN;
		int16_t ret_int16;
		int32_t ret_int32;

		if (!(data < end)) {
			ERROR_ACCESS("CFF DICT number");
			return NAN;
		}
		switch (data[0]) {
		case 0x1e:
			// Floating point value
			while (!(nibble_high && nibble == 0xf) && data < end) {
				nibble = nibble_high ? (data[0] >> 4) :
					data[0] & 0xf;
				nibble_str += nibble_chr[nibble];
				nibble_high = !nibble_high;
				if (nibble_high) {
					data++;
					if (!(data < end)) {
						ERROR_ACCESS("CFF DICT floating-point "
									 "number");
						return NAN;
					}
				}
			};
			sscanf(nibble_str.c_str(), "%lf", &ret);
			break;
		case 28:
			if (!(data + 3 <= end)) {
				ERROR_ACCESS("CFF DICT 3 byte integer number");
				return NAN;
			}
			ret_int16 = (data[1] << 8) | data[2];
			ret = ret_int16;
			data += 3;
			break;
		case 29:
			if (!(data + 5 <= end)) {
				ERROR_ACCESS("CFF DICT 5 byte integer number");
				return NAN;
			}
			ret_int32 =
				(data[1] << 24) |
				(data[2] << 16) |
				(data[3] << 8) |
				data[4];
			ret = ret_int32;
			data += 5;
			break;
		default:
			if (data[0] >= 32 && data[0] <= 246) {
				ret = data[0] - 139;
				data++;
			}
			else if (data[0] >= 247 && data[0] <= 250) {
				if (!(data + 2 <= end)) {
					ERROR_ACCESS("CFF DICT 2 byte positive integer "
								 "number");
					return NAN;
				}
				ret = (data[0] - 247) * 256 +
					data[1] + 108;
				data += 2;
			}
			else if (data[0] >= 251 && data[0] <= 254) {
				if (!(data + 2 <= end)) {
					ERROR_ACCESS("CFF DICT 2 byte negative integer "
								 "number");
					return NAN;
				}
				ret = -(data[0] - 251) * 256 -
					data[1] - 108;
				data += 2;
			}
			else {
				fprintf(stderr, "%s:%d: Illegal CFF DICT number, "
						"b0 = %d, 0x%02x\n",
						__FILE__, __LINE__, data[0], data[0]);
			}
		}

		return ret;
	}

	double font_embed_t::search_cff_top_dict_number(
		const std::vector<uint8_t> &top_dict,
		uint8_t operator_value_1, uint8_t operator_value_2)
	{
		if (!(operator_value_1 <= 21)) {
			fprintf(stderr, "%s:%d: Illegal CFF operator %d, "
					"0x%02x requested\n", __FILE__, __LINE__,
					operator_value_1, operator_value_1);
			return NAN;
		}

		double last = NAN;

		for (std::vector<uint8_t>::const_iterator iterator =
				 top_dict.begin();
			 iterator != top_dict.end();) {
			if (operator_value_1 == 12 ?
				iterator[0] == 12 && iterator[1] == operator_value_2 :
				iterator[0] == operator_value_1) {
				return last;
			}
			else if (iterator[0] <= 21) {
				// Some other operators
				last = NAN;
				iterator += iterator[0] == 12 ? 2 : 1;
			}
			else if ((iterator[0] >= 28 && iterator[0] <= 30) ||
					 (iterator[0] >= 32 && iterator[0] <= 254)) {
				last = parse_cff_dict_number(iterator, top_dict.end());
			}
			else {
				fprintf(stderr, "%s:%d: error: unexpected byte %d, "
						"0x%02x in CFF DICT (attempt to skip...)\n",
						__FILE__, __LINE__, iterator[0], iterator[0]);
				iterator++;
			}
		}

		return NAN;
	}

	std::string font_embed_t::cff_sid_to_string(
		uint16_t sid,
		const std::vector<std::vector<uint8_t> > &string_index)
	{
#include "table/cffstdstr.h"
		if (!(sid < ncff_standard_string + string_index.size())) {
			ERROR_ACCESS("CFF SID");
		}

		return sid < ncff_standard_string ?
			cff_standard_string[sid] :
			sid - ncff_standard_string < string_index.size() ?
			std::string(
				string_index[sid - ncff_standard_string].begin(),
				string_index[sid - ncff_standard_string].end()) :
			"";
	}

	std::vector<std::string> font_embed_t::parse_cff_charset(
		const std::vector<uint8_t> &data,
		uint32_t charset_offset, uint32_t nglyph,
		const std::vector<std::vector<uint8_t> > &string_index)
	{
		if (!(charset_offset < data.size())) {
			ERROR_ACCESS("CFF Charset");
		}

		const uint8_t format = data[charset_offset];
		std::vector<std::string> charset;

		switch (format) {
		case 0:
			charset.push_back(".notdef");
			for (size_t i = 0; i < nglyph - 1; i++) {
				uint16_t sid;

				if (!(charset_offset + 2 * i + 3 <= data.size())) {
					ERROR_ACCESS("CFF Charset");
				}
				memcpy(&sid, &data[charset_offset + 2 * i + 1], 2);
#ifdef LITTLE_ENDIAN
				sid = bswap_16(sid);
#endif // LITTLE_ENDIAN
				charset.push_back(cff_sid_to_string(sid, string_index));
			}
			break;
		default:
			fprintf(stderr, "%s:%d: error: unsupported Charset "
					"format %d\n", __FILE__, __LINE__, format);
			break;
		}

		return charset;
	}

	void font_embed_t::parse_cff(
		std::vector<std::string> &charset,
		const std::vector<uint8_t> &font_data,
		uint32_t cff_offset)
	{
		struct cff_header_s {
			uint8_t major;
			uint8_t minor;
			uint8_t hdr_size;
			uint8_t off_size;
		} cff_header;

		uint32_t current_offset = cff_offset;

		if (!(current_offset + sizeof(struct cff_header_s) <=
			  font_data.size())) {
			ERROR_ACCESS("CFF header");
		}

		memcpy(&cff_header, &font_data[current_offset],
			   sizeof(struct cff_header_s));
		current_offset += sizeof(struct cff_header_s);

		const std::vector<std::vector<uint8_t> > cff_name_index =
			parse_cff_index(font_data, current_offset);
		const std::vector<std::vector<uint8_t> > cff_top_dict_index =
			parse_cff_index(font_data, current_offset);

		if (cff_top_dict_index.size() != 1) {
			fprintf(stderr, "%s:%d: error: CFF FontSet is not "
					"supported\n", __FILE__, __LINE__);
			return;
		}

		const uint32_t charset_offset = cff_offset +
			search_cff_top_dict_number(cff_top_dict_index[0], 15);
		uint32_t charstrings_offset = cff_offset +
			search_cff_top_dict_number(cff_top_dict_index[0], 17);
		uint32_t cff_nglyph = 0;

		parse_cff_index(font_data, charstrings_offset, &cff_nglyph);

		const std::vector<std::vector<uint8_t> > string_index =
			parse_cff_index(font_data, current_offset);

		charset = parse_cff_charset(font_data, charset_offset,
			cff_nglyph, string_index);
	}

	bool font_embed_t::parse_otf_cff_header(
		std::string &font_name, unsigned short &cid_encoding_id,
		uint32_t font_descriptor_flags, double *font_bbox,
		double &italic_angle, double &ascent, double &descent,
		double &leading, double &cap_height, double &x_height,
		double &stem_v,double &avg_width,
		std::map<wchar_t, uint16_t> &cid_map,
		std::vector<std::string> &charset,
		std::vector<uint16_t> &advance_width,
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

		if (!(sizeof(struct otf_offset_table_s) <=
			  font_data.size())) {
			ERROR_ACCESS("OTF offset table");
		}
		memcpy(&offset_table, &font_data[0],
			   sizeof(struct otf_offset_table_s));
		if (strncmp(offset_table.sfnt_version, "OTTO", 4) != 0) {
			// Not a OpenType CFF/Type 2 font
			return false;
		}
#ifdef LITTLE_ENDIAN
		offset_table.num_tables = bswap_16(offset_table.num_tables);
#endif // LITTLE_ENDIAN

		std::map<std::string, std::pair<uint32_t, uint32_t> > table =
			parse_ttf_offset_table(
				font_data, sizeof(struct otf_offset_table_s),
				offset_table.num_tables);

		const uint32_t cmap_offset = table["cmap"].first;
		const uint32_t head_offset = table["head"].first;
		const uint32_t hhea_offset = table["hhea"].first;
		const uint32_t hmtx_offset = table["hmtx"].first;
		const uint32_t maxp_offset = table["maxp"].first;
		const uint32_t name_offset = table["name"].first;
		const uint32_t os_2_offset = table["OS/2"].first;
		const uint32_t post_offset = table["post"].first;

		cff_offset = table["CFF "].first;
		cff_length = table["CFF "].second;

		if (!(cmap_offset != 0 && head_offset != 0 &&
			  hhea_offset != 0 && hmtx_offset != 0 &&
			  maxp_offset != 0 && name_offset != 0 &&
			  os_2_offset != 0 && cff_offset != 0)) {
			fprintf(stderr, "%s:%d: error: OTF CFF font is "
					"missing required tables\n", __FILE__, __LINE__);
			return false;
		}

		parse_ttf_name(font_name, cid_encoding_id, font_data,
					   name_offset);

		uint16_t units_per_em;

		parse_ttf_head(font_bbox, units_per_em, font_data,
					   head_offset);
		font_descriptor_flags = 0;
		parse_ttf_post(charset, italic_angle, font_descriptor_flags,
					   font_data, post_offset);
		parse_ttf_cmap(cid_map, font_data, cmap_offset);
		parse_ttf_os_2(font_descriptor_flags, ascent, descent,
					   leading, cap_height, x_height, stem_v,
					   avg_width, font_data, os_2_offset,
					   units_per_em);

		const uint16_t num_glyphs =
			parse_ttf_maxp(font_data, maxp_offset);
		advance_width =
			parse_ttf_hmtx(font_data, hmtx_offset,
						   parse_ttf_hhea(font_data, hhea_offset));

		if (advance_width.empty()) {
			fprintf(stderr, "%s:%d: error: hMetrics in hmtx is "
					"empty\n", __FILE__, __LINE__);
			return false;
		}
		advance_width.resize(num_glyphs, advance_width.back());

		if (charset.empty()) {
			parse_cff(charset, font_data, cff_offset);
		}

		cid_map.erase(L'\uffff');

		return true;
	}

	bool font_embed_t::parse_otf_cff_header(
		std::string &font_name, unsigned short &cid_encoding_id,
		unsigned int &cff_offset, unsigned int &cff_length,
		const std::vector<uint8_t> &font_data)
	{
		uint32_t font_descriptor_flags = 0;
		double font_bbox[4];
		double italic_angle;
		double ascent;
		double descent;
		double leading;
		double cap_height;
		double x_height;
		double stem_v;
		double avg_width;
		std::map<wchar_t, uint16_t> cid_map;
		std::vector<std::string> charset;
		std::vector<uint16_t> advance_width;

		return parse_otf_cff_header(
			font_name, cid_encoding_id, font_descriptor_flags,
			font_bbox, italic_angle, ascent, descent, leading,
			cap_height, x_height, stem_v, avg_width, cid_map,
			charset, advance_width, cff_offset, cff_length,
			font_data);
	}

	std::vector<std::string>
	font_embed_t::charset_from_adobe_glyph_list(
		std::map<wchar_t, uint16_t> &cid_map)
	{
		// Regenerate cid_map from the Adobe glyph list

		std::vector<std::string> charset;

		if (cid_map.empty()) {
			return std::vector<std::string>(1, ".notdef");
		}
		charset.resize(cid_map.size());
		for (std::map<wchar_t, uint16_t>::const_iterator iterator =
				 cid_map.begin();
			 iterator != cid_map.end(); iterator++) {
			if (iterator->second < charset.size()) {
#include "table/adobeglyphlist.h"

				const wchar_t *lower = std::lower_bound(
					adobe_glyph_ucs, adobe_glyph_ucs + nadobe_glyph,
					iterator->first);
				// The longest Adobe glyph name is 20 characters long
				// (0x03b0 = upsilondieresistonos)
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
				charset[iterator->second] = buf;
			}
		}

		return charset;
	}

	bool font_embed_t::parse_ttf_header(
		std::string &font_name, unsigned short &cid_encoding_id,
		uint32_t font_descriptor_flags, double *font_bbox,
		double &italic_angle, double &ascent, double &descent,
		double &leading, double &cap_height, double &x_height,
		double &stem_v,double &avg_width,
		std::map<wchar_t, uint16_t> &cid_map,
		std::vector<std::string> &charset,
		std::vector<uint16_t> &advance_width,
		const std::vector<uint8_t> &font_data)
	{
		cid_map.clear();
		charset.clear();

		struct ttf_offset_table_s {
			fixed_t sfnt_version;
			uint16_t num_tables;
			uint16_t search_range;
			uint16_t entry_selector;
			uint16_t range_shift;
		} offset_table;

		if (!(sizeof(struct ttf_offset_table_s) <=
			  font_data.size())) {
			ERROR_ACCESS("TTF offset table");
		}
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

		std::map<std::string, std::pair<uint32_t, uint32_t> > table =
			parse_ttf_offset_table(
				font_data, sizeof(struct ttf_offset_table_s),
				offset_table.num_tables);

		const uint32_t cmap_offset = table["cmap"].first;
		const uint32_t head_offset = table["head"].first;
		const uint32_t hhea_offset = table["hhea"].first;
		const uint32_t hmtx_offset = table["hmtx"].first;
		const uint32_t maxp_offset = table["maxp"].first;
		const uint32_t name_offset = table["name"].first;
		const uint32_t os_2_offset = table["OS/2"].first;
		const uint32_t post_offset = table["post"].first;

		if (!(cmap_offset != 0 && head_offset != 0 &&
			  hhea_offset != 0 && hmtx_offset != 0 &&
			  maxp_offset != 0 && name_offset != 0 &&
			  os_2_offset != 0)) {
			fprintf(stderr, "%s:%d: error: TTF font is missing "
					"required tables\n", __FILE__, __LINE__);
			return false;
		}

		parse_ttf_name(font_name, cid_encoding_id, font_data,
					   name_offset);

		uint16_t units_per_em;

		parse_ttf_head(font_bbox, units_per_em, font_data,
					   head_offset);
		font_descriptor_flags = 0;
		parse_ttf_post(charset, italic_angle, font_descriptor_flags,
					   font_data, post_offset);
		parse_ttf_cmap(cid_map, font_data, cmap_offset);
		parse_ttf_os_2(font_descriptor_flags, ascent, descent,
					   leading, cap_height, x_height, stem_v,
					   avg_width, font_data, os_2_offset,
					   units_per_em);

		const uint16_t num_glyphs =
			parse_ttf_maxp(font_data, maxp_offset);
		advance_width =
			parse_ttf_hmtx(font_data, hmtx_offset,
						   parse_ttf_hhea(font_data, hhea_offset));

		if (advance_width.empty()) {
			fprintf(stderr, "%s:%d: error: hMetrics in hmtx is "
					"empty\n", __FILE__, __LINE__);
			return false;
		}
		advance_width.resize(num_glyphs, advance_width.back());

		if (charset.empty()) {
			charset = charset_from_adobe_glyph_list(cid_map);
		}

		cid_map.erase(L'\uffff');

		return true;
	}

	bool font_embed_t::parse_ttf_header(
		std::string &font_name, double *font_bbox,
		std::map<wchar_t, uint16_t> &cid_map,
		std::vector<std::string> &charset,
		const std::vector<uint8_t> &font_data)
	{
		uint16_t cid_encoding_id;
		uint32_t font_descriptor_flags = 0;
		double italic_angle;
		double ascent;
		double descent;
		double leading;
		double cap_height;
		double x_height;
		double stem_v;
		double avg_width;
		std::vector<uint16_t> advance_width;

		return parse_ttf_header(
			font_name, cid_encoding_id, font_descriptor_flags,
			font_bbox, italic_angle, ascent, descent, leading,
			cap_height, x_height, stem_v, avg_width, cid_map,
			charset, advance_width, font_data);
	}

}
