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

#include "fontembed.h"
#include <string.h>
#include <stdio.h>
#ifdef WIN32
#define snprintf _snprintf
#endif

// ROOT integration
#include "RConfig.h"
#ifdef R__BYTESWAP
#ifndef LITTLE_ENDIAN
#define LITTLE_ENDIAN 1
#endif // LITTLE_ENDIAN
#include "Byteswap.h"
#define bswap_16(x)	Rbswap_16((x))
#define bswap_32(x)	Rbswap_32((x))
#else // R__BYTESWAP
#ifdef LITTLE_ENDIAN
#undef LITTLE_ENDIAN
#endif // LITTLE_ENDIAN
#endif // R__BYTESWAP

// References:
//
// Adobe Systems, Inc., PostScript language Document Structuring
// Convention specification (Adobe Systems, Inc., San Jose, CA, 1992),
// version 3.0, section 5.1.
//
// Adobe Systems, Inc., PostScript language reference manual
// (Addison-Wesley, Reading, MA, 1999), 3rd edition, section 5.8.1.
//
// Adobe Systems, Inc., Adobe Type 1 Font Format (Addison-Wesley,
// Reading, MA, 1993), version 1.1
//
// Adobe Systems, Inc., The Compact Font Format specification, Adobe
// Technical Note 5176 (Adobe, Mountain View, CA, 2003), 4 December
// 2003 document
//
// Adobe Systems, Inc., Type 2 charstring format, Adobe Technical Note
// 5177 (Adobe, San Jose, CA, 2000), 16 March 2000 document

namespace mathtext {

	void font_embed_postscript_t::append_asciihex(
		std::string &ascii, const unsigned char *buffer,
		const size_t length)
	{
		const int width = 64;
		int column = 0;

		for(size_t i = 0; i < length; i++) {
			char str[3];

			snprintf(str, 3, "%02hhX", buffer[i]);
			ascii.append(str, 2);
			column += 2;
			if(column >= width) {
				ascii.append(1, '\n');
				column = 0;
			}
		}
	}

	unsigned int font_embed_postscript_t::ascii85_line_count(
		const uint8_t *buffer, const size_t length)
	{
		const unsigned int width = 64;
		unsigned int column = 0;
		unsigned int line = 0;

		if (length >= 4) {
			for (size_t i = 0; i < length - 3; i += 4) {
				unsigned int b = reinterpret_cast<
					const unsigned int *>(buffer)[i >> 2];

				if (b == 0) {
					column++;
					if (column == width - 1) {
						line++;
						column = 0;
					}
				}
				else {
					if (column + 5 >= width) {
						column += 5 - width;
						line++;
					}
					else {
						column += 5;
					}
				}
			}
		}
		if (column + (length & 3) + 3 >= width) {
			line++;
		}

		return line;
	}

	void font_embed_postscript_t::append_ascii85(
		std::string &ascii, const uint8_t *buffer,
		const size_t length)
	{
		const int width = 64;
		int column = 0;

		if (length >= 4) {
			for (size_t i = 0; i < length - 3; i += 4) {
				unsigned int dword = reinterpret_cast<
					const unsigned int *>(buffer)[i >> 2];

				if (dword == 0) {
					ascii.append(1, 'z');
					column++;
					if (column == width - 1) {
						ascii.append(1, '\n');
						column = 0;
					}
				}
				else {
#ifdef LITTLE_ENDIAN
					dword = bswap_32(dword);
#endif // LITTLE_ENDIAN

					char str[5];

					str[4] = static_cast<char>(dword % 85 + '!');
					dword /= 85;
					str[3] = static_cast<char>(dword % 85 + '!');
					dword /= 85;
					str[2] = static_cast<char>(dword % 85 + '!');
					dword /= 85;
					str[1] = static_cast<char>(dword % 85 + '!');
					dword /= 85;
					str[0] = static_cast<char>(dword % 85 + '!');
					for (size_t j = 0; j < 5; j++) {
						ascii.append(1, str[j]);
						column++;
						if(column == width) {
							ascii.append(1, '\n');
							column = 0;
						}
					}
				}
			}
		}

		int k = length & 3;

		if(k > 0) {
			unsigned int dword = 0;

			memcpy(&dword, buffer + (length & ~3), k);
#ifdef LITTLE_ENDIAN
			dword = bswap_32(dword);
#endif // LITTLE_ENDIAN

			char str[5];

			str[4] = static_cast<char>(dword % 85 + '!');
			dword /= 85;
			str[3] = static_cast<char>(dword % 85 + '!');
			dword /= 85;
			str[2] = static_cast<char>(dword % 85 + '!');
			dword /= 85;
			str[1] = static_cast<char>(dword % 85 + '!');
			dword /= 85;
			str[0] = static_cast<char>(dword % 85 + '!');
			for(int j = 0; j < k + 1; j++) {
				ascii.append(1, str[j]);
				column++;
				if(column == width) {
					ascii.append(1, '\n');
					column = 0;
				}
			}

		}
		if(column > width - 2)
			ascii.append(1, '\n');
		ascii.append("~>");
	}

	std::string font_embed_postscript_t::font_embed_type_1(
		std::string &font_name,
		const std::vector<unsigned char> &font_data)
	{
		// Embed font type 1

		struct pfb_segment_header_s {
			char always_128;
			char type;
			unsigned int length;
		};
		enum {
			TYPE_ASCII = 1,
			TYPE_BINARY,
			TYPE_EOF
		};

		char magic_number[2];
		std::string ret;

		memcpy(magic_number, &font_data[0], 2);
		if(magic_number[0] == '\200') {
			// IBM PC format printer font binary

			// FIXME: Maybe the real name can be parsed out of the
			// file
			font_name = "";

			struct pfb_segment_header_s segment_header;
			size_t offset = 2;

			// The two char elements of struct pfb_segment_header_s
			// are most likely aligned to larger than 1 byte
			// boundaries, so copy all the elements individually
			segment_header.always_128 = font_data[offset];
			segment_header.type = font_data[offset + 1];
			memcpy(&segment_header.length, &font_data[offset + 2],
				   sizeof(unsigned int));
			offset += sizeof(unsigned int) + 2;
			while (segment_header.type != TYPE_EOF) {
#ifdef LITTLE_ENDIAN
				segment_header.length =
					bswap_32(segment_header.length);
#endif // LITTLE_ENDIAN
				char *buffer = new char[segment_header.length];

				memcpy(buffer, &font_data[offset],
					   segment_header.length);
				offset += segment_header.length;

				switch(segment_header.type) {
				case TYPE_ASCII:
					// Simple CR -> LF conversion
					for (int i = 0;
						 i < (int)(segment_header.length) - 1; i++) {
						if(buffer[i] == '\r' &&
						   buffer[i + 1] != '\n') {
							buffer[i] = '\n';
						}
					}
					if (buffer[segment_header.length - 1] == '\r') {
						buffer[segment_header.length - 1] = '\n';
					}
					ret.append(buffer, segment_header.length);
					break;
				case TYPE_BINARY:
					append_asciihex(
						ret, reinterpret_cast<uint8_t *>(buffer),
						segment_header.length);
					break;
				default:
					{}
				}

				delete [] buffer;
			}

			return ret;
		}
		else if(strncmp(magic_number, "%!", 2) == 0) {
			// Printer font ASCII
			fprintf(stderr, "%s:%d: Printer font ASCII is not "
					"implemented\n", __FILE__, __LINE__);
			return std::string();
		}

		return std::string();
	}

	std::string font_embed_postscript_t::font_embed_type_2(
		std::string &font_name,
		const std::vector<unsigned char> &font_data)
	{
		// Embed an OpenType CFF (Type 2) file in ASCII85 encoding
		// with the PostScript syntax

		unsigned short cid_encoding_id;
		unsigned int cff_offset;
		unsigned int cff_length;

		if (!parse_otf_cff_header(font_name, cid_encoding_id,
								  cff_offset, cff_length,
								  font_data)) {
			return std::string();
		}

		std::vector<unsigned char> cff;

		cff.resize(cff_length + 10);
		memcpy(&cff[0], "StartData\r", 10);
		memcpy(&cff[10], &font_data[cff_offset], cff_length);

		char linebuf[BUFSIZ];
		std::string ret;

		snprintf(linebuf, BUFSIZ, "%%%%BeginResource: FontSet (%s)\n",
				 font_name.c_str());
		ret.append(linebuf);
		ret.append("%%VMusage: 0 0\n");
		ret.append("/FontSetInit /ProcSet findresource begin\n");
		snprintf(linebuf, BUFSIZ, "%%%%BeginData: %u ASCII Lines\n",
				 ascii85_line_count(&cff[0], cff_length) + 2);
		ret.append(linebuf);
		snprintf(linebuf, BUFSIZ,
				 "/%s %u currentfile /ASCII85Decode filter cvx exec\n",
				 font_name.c_str(), cff_length);
		ret.append(linebuf);
		append_ascii85(ret, &cff[0], cff_length + 10);
		ret.append(1, '\n');
		ret.append("%%EndData\n");
		ret.append("%%EndResource\n");

		return ret;
	}

	std::string font_embed_postscript_t::font_embed_type_42(
		std::string &font_name,
		const std::vector<unsigned char> &font_data)
	{
		// Embed an TrueType as Type 42 with the PostScript syntax

		double font_bbox[4];
		std::map<wchar_t, uint16_t> cid_map;
		std::vector<std::string> char_strings;

		if (!parse_ttf_header(font_name, font_bbox, cid_map,
							  char_strings, font_data)) {
			fprintf(stderr, "%s:%d:\n", __FILE__, __LINE__);
			return std::string();
		}

		char linebuf[BUFSIZ];
		std::string ret;

		snprintf(linebuf, BUFSIZ, "%%%%BeginResource: FontSet (%s)\n",
				 font_name.c_str());
		ret.append(linebuf);
		ret.append("%%VMusage: 0 0\n");
		ret.append("11 dict begin\n");
		snprintf(linebuf, BUFSIZ, "/FontName /%s def\n",
				 font_name.c_str());
		ret.append(linebuf);
		ret.append("/Encoding 256 array\n");
		snprintf(linebuf, BUFSIZ,
				 "0 1 255 { 1 index exch /%s put } for\n",
				 char_strings[0].c_str());
		ret.append(linebuf);
		for (unsigned int code_point = 0; code_point < 256;
			 code_point++) {
			unsigned int glyph_index = cid_map[code_point];

			if (char_strings[glyph_index] != ".notdef" &&
				char_strings[glyph_index] != "") {
				snprintf(linebuf, BUFSIZ, "dup %u /%s put\n",
						 code_point,
						 char_strings[glyph_index].c_str());
				ret.append(linebuf);
			}
		}
		ret.append("readonly def\n");
		ret.append("/PaintType 0 def\n");	// 0 for filled, 2 for stroked
		ret.append("/FontMatrix [1 0 0 1 0 0] def\n");
		snprintf(linebuf, BUFSIZ, "/FontBBox [%f %f %f %f] def\n",
				 font_bbox[0], font_bbox[1], font_bbox[2], font_bbox[3]);
		ret.append(linebuf);
		ret.append("/FontType 42 def\n");
		// FIXME: XUID generation using the font data's MD5
		ret.append("/sfnts [\n");

		const size_t block_size = 32262;
		size_t offset = 0;

		while (offset < font_data.size()) {
			const size_t output_length =
				std::min(block_size, font_data.size() - offset);

			ret.append("<\n");
			append_asciihex(ret, &font_data[offset], output_length);
			ret.append(">\n");
			offset += output_length;
		}
		ret.append("] def\n");

		unsigned int char_strings_count = 0;

		for (std::vector<std::string>::const_iterator iterator =
				 char_strings.begin();
			 iterator < char_strings.end(); iterator++) {
			if (!iterator->empty()) {
				char_strings_count++;
			}
		}

		snprintf(linebuf, BUFSIZ, "/CharStrings %u dict dup begin\n",
				 char_strings_count);
		ret.append(linebuf);
		for (unsigned int glyph_index = 0;
			 glyph_index < char_strings.size(); glyph_index++) {
			if (!char_strings[glyph_index].empty()) {
				snprintf(linebuf, BUFSIZ, "/%s %u def\n",
						 char_strings[glyph_index].c_str(),
						 glyph_index);
				ret.append(linebuf);
			}
		}
		ret.append("end readonly def\n");
		ret.append("FontName currentdict end definefont pop\n");
		ret.append("%%EndResource\n");

		return ret;
	}
}
