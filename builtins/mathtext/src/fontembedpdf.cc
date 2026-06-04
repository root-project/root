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
#include <string.h>
#include <stdio.h>
#include <algorithm>

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

	uint16_t font_embed_pdf_t::utf16_high_surrogate(uint32_t code)
	{
		return ((code - 0x10000) >> 10) + 0xd800;
	}

	uint16_t font_embed_pdf_t::utf16_low_surrogate(uint32_t code)
	{
		return ((code - 0x10000) & ((1U << 10) - 1U)) + 0xdc00;
	}

	std::string font_embed_pdf_t::utf16be_str(uint32_t code)
	{
		char buffer[9];

		if (code < 0x10000) {
			snprintf(buffer, 5, "%04x", code);
		}
		else if (code < 0x110000) {
			snprintf(buffer, 9, "%04x%04x",
					 utf16_high_surrogate(code),
					 utf16_low_surrogate(code));
		}
		else {
			snprintf(buffer, 5, "????");
		}
		buffer[8] = '\0';

		return buffer;
	}

	std::string font_embed_pdf_t::uint16_str(uint16_t cid)
	{
		char buffer[6];

		snprintf(buffer, 6, "%u", cid);
		buffer[5] = '\0';

		return buffer;
	}

	void font_embed_pdf_t::add_cidfont_w_token(
		std::string &s, const std::string &t)
	{
		while (!s.empty() &&
			   (s.rbegin()[0] == ' ' || s.rbegin()[0] == '\n')) {
			s.resize(s.size() - 1);
		}

		if (!((s.rbegin()[0] == '[' &&
			   t.begin()[0] >= '0' && s.begin()[0] <= '9') ||
			  (s.rbegin()[0] >= '0' && s.rbegin()[0] <= '9' &&
			   t.begin()[0] == ']'))) {
			s += " ";
		}
		s += t;

		static const size_t pdf_line_width = 79;
		const size_t last_break = s.rfind('\n');

		if (last_break != std::string::npos &&
			s.size() - last_break - 1 <= pdf_line_width) {
			return;
		}

		const size_t new_break =
			s.rfind(' ',
					last_break == std::string::npos ? s.size() :
					last_break + pdf_line_width + 1);

		if (new_break != std::string::npos) {
			s[new_break] = '\n';
		}
	}

	std::string font_embed_pdf_t::cidfont_w(
		const std::vector<uint16_t> &advance_width)
	{
		// Format 1 and 2 are the two formats on PDF 1.4 Reference, p.
		// 340:
		//
		// c [w1 w2 ... wn]   <-- "format 1"
		// cfirst clast w     <-- "format 2"
		bool in_format_1 = false;
		std::string str;

		str = "/W [";
		for (size_t cid = 1, cid_next = 2;
			 cid < advance_width.size(); cid = cid_next) {
			while (cid_next < advance_width.size() &&
				   advance_width[cid_next] == advance_width[cid]) {
				cid_next++;
			}

			if (cid_next == cid + 1) {
				if (!in_format_1) {
					add_cidfont_w_token(str, uint16_str(cid));
					add_cidfont_w_token(str, "[");
				}
				in_format_1 = true;
			}
			else {
				if (in_format_1) {
					add_cidfont_w_token(str, "]");
				}
				add_cidfont_w_token(str, uint16_str(cid));
				add_cidfont_w_token(str, uint16_str(cid_next - 1));
				in_format_1 = false;
			}
			add_cidfont_w_token(str, uint16_str(advance_width[cid]));
		}
		if (in_format_1) {
			add_cidfont_w_token(str, "]");
		}
		add_cidfont_w_token(str, "]\n");

		return str;
	}

	std::map<std::string, std::string>
	font_embed_pdf_t::font_embed_cid(
		std::string &font_name,
		const std::vector<unsigned char> &font_data,
		unsigned int type)
	{
		unsigned short cid_encoding_id;
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
		unsigned int cff_offset;
		unsigned int cff_length;

		std::map<std::string, std::string> pdf_object;

		if (!(type == 0 ?
			  parse_otf_cff_header(
				font_name, cid_encoding_id, font_descriptor_flags,
				font_bbox, italic_angle, ascent, descent, leading,
				cap_height, x_height, stem_v, avg_width, cid_map,
				charset, advance_width, cff_offset, cff_length,
				font_data) :
			  parse_ttf_header(
				font_name, cid_encoding_id, font_descriptor_flags,
				font_bbox, italic_angle, ascent, descent, leading,
				cap_height, x_height, stem_v, avg_width, cid_map,
				charset, advance_width, font_data))) {
			return pdf_object;
		}

		// Layout of a embedded CID Type 0/2 font
		//
		// /Font (/Subtype /Type0/2)
		// | /Encoding
		// +-- /CMap
		// | /DescendantFonts
		// +-- /Font (/Subtype /CIDFontType0/2)
		//     | /FontDescriptor
		//     +-- /FontDescriptor
		//         | /FontFile3/2
		//         +-- stream (plain or /SubType /CIDFontType0C)

		const uint16_t max_width =
			*std::max_element(advance_width.begin(),
							  advance_width.end());

		std::string font_file32 =
			"<<\n";

		if (type == 0) {
			font_file32 += "/Subtype /CIDFontType0C\n";
		}

		char buffer[4096];

		snprintf(buffer, 4096,
				 "/Length %u\n"
				 ">>\n"
				 "stream\n",
				 type == 0 ? cff_length :
				 static_cast<unsigned int>(font_data.size()));
		font_file32 += buffer;

		if (type == 0) {
			font_file32.insert(
				font_file32.end(), font_data.begin() + cff_offset,
				font_data.begin() + cff_offset + cff_length);
		}
		else {
			font_file32.insert(
				font_file32.end(), font_data.begin(),
				font_data.end());
		}
		font_file32 += "\nendstream\n";

		pdf_object["/FontFile32"] = font_file32;

		std::string font_descriptor =
			"<<\n"
			"/Type /FontDescriptor\n";

		snprintf(buffer, 4096,
				 "/FontName /%s\n"
				 "/Flags %u\n"
				 "/FontBBox [%g %g %g %g]\n"
				 "/ItalicAngle %g\n"
				 "/Ascent %g\n"
				 "/Descent %g\n"
				 "/Leading %g\n"
				 "/CapHeight %g\n"
				 "/XHeight %g\n"
				 "/StemV %g\n"
				 "/StemH %g\n"
				 "/AvgWidth %g\n"
				 "/MaxWidth %hu\n",
				 font_name.c_str(), font_descriptor_flags,
				 font_bbox[0], font_bbox[1], font_bbox[2],
				 font_bbox[3], italic_angle, ascent, descent,
				 leading, cap_height, x_height, stem_v, 0.0,
				 avg_width, max_width);
		font_descriptor += buffer;
		font_descriptor += "/FontFile" +
			std::string(type == 0 ? "3" : "2") + " %u %u R\n";
		font_descriptor += ">>\n";

		pdf_object["/FontDescriptor"] = font_descriptor;

		const std::string registry = "Adobe";
		const std::string ordering = "Identity";
		const std::string supplement = "0";

		pdf_object["/CIDFont"] =
			"<<\n"
			"/Type /Font\n"
			"/Subtype /CIDFontType" +
			std::string(type == 0 ? "0" : "2") + "\n"
			"/BaseFont /" + font_name + "\n"
			"/CIDSystemInfo\n"
			"<<\n"
			"/Registry (" + registry + ")\n"
			"/Ordering (" + ordering + ")\n"
			"/Supplement " + supplement + "\n"
			">>\n"
			"/FontDescriptor %u %u R\n" +
			cidfont_w(advance_width) +
			">>\n";

		const std::string cmap_name = "UniMT-UTF16-H";
		std::string cmap_stream;

		cmap_stream =
			"%!PS-Adobe-3.0 Resource-CMap\n"
			"%%DocumentNeededResources: ProcSet (CIDInit)\n"
			"%%IncludeResource: ProcSet (CIDInit)\n"
			"%%BeginResource: CMap (" + cmap_name + ")\n"
			"%%Title: (" + cmap_name + " " + registry + " " +
			ordering + " " + supplement + ")\n"
			"%%Version: 1.000\n"
			"%%EndComments\n"
			"\n"
			"/CIDInit /ProcSet findresource begin\n"
			"\n"
			"12 dict begin\n"
			"\n"
			"begincmap\n"
			"\n"
			"/CIDSystemInfo 3 dict dup begin\n"
			"  /Registry (" + registry + ") def\n"
			"  /Ordering (" + ordering + ") def\n"
			"  /Supplement " + supplement + " def\n"
			"end def\n"
			"\n"
			"/CMapName /" + cmap_name + " def\n"
			"/CMapVersion 1.000 def\n"
			"/CMapType 1 def\n"
			"\n"
			"/XUID [1000000 10] def\n"
			"\n"
			"/WMode 0 def\n"
			"\n";

		// UTF-16BE

		cmap_stream +=
			"3 begincodespacerange\n"
			"  <0000>     <D7FF>\n"
			"  <D800DC00> <DBFFDFFF>\n"
			"  <E000>     <FFFF>\n"
			"endcodespacerange\n"
			"\n"
			"1 beginnotdefrange\n"
			"<0000> <001f> 1\n"
			"endnotdefrange\n"
			"\n";

		// Split into begin/endcidchar and begin/endcidrange

		std::vector<struct cid_char_s> cid_char;
		std::vector<struct cid_range_s> cid_range;

		{
			std::map<wchar_t, uint16_t>::const_iterator iterator =
				cid_map.begin();
			const struct cid_range_s r0 = {
				static_cast<uint32_t>(iterator->first),
				static_cast<uint32_t>(iterator->first),
				iterator->second
			};

			cid_range.push_back(r0);
			iterator++;
			for (; iterator != cid_map.end(); iterator++) {
				const unsigned int code = iterator->first;
				if (code == cid_range.back().code_last + 1U &&
					iterator->second ==
					iterator->first - cid_range.back().code_first +
					cid_range.back().cid &&
					// Split around UTF-16BE byte overflow (<..ff> to
					// <..00>)
					(code & ((1U << 8) - 1U)) != 0U &&
					(code < 0x10000 ? true :
					 ((code >> 10) & ((1U << 8) - 1U)) != 0U)) {
					cid_range.back().code_last++;
				}
				else {
					if (cid_range.back().code_first ==
						cid_range.back().code_last) {
						const struct cid_char_s c = {
							cid_range.back().code_first,
							cid_range.back().cid
						};

						cid_char.push_back(c);
						cid_range.pop_back();
					}

					const struct cid_range_s r = {
						static_cast<uint32_t>(iterator->first),
						static_cast<uint32_t>(iterator->first),
						iterator->second
					};

					cid_range.push_back(r);
				}
			}
		}
		if (cid_range.back().code_first ==
			cid_range.back().code_last) {
			const struct cid_char_s c = {
				cid_range.back().code_first,
				cid_range.back().cid
			};

			cid_char.push_back(c);
			cid_range.pop_back();
		}

		for (std::vector<struct cid_char_s>::const_iterator
				 iterator = cid_char.begin();
			 iterator != cid_char.end(); iterator++) {
			if ((iterator - cid_char.begin()) % 100U == 0U) {
				if (iterator != cid_char.begin()) {
					cmap_stream +=
						"endcidchar\n"
						"\n";
				}

				const unsigned int chunk =
					std::min(100U, static_cast<unsigned int>
							 (cid_char.end() - iterator) / 2);

				snprintf(buffer, 4096, "%u begincidchar\n",
						 chunk);
				cmap_stream += buffer;
			}

			cmap_stream += "<" + utf16be_str(iterator->code) + "> " +
				uint16_str(iterator->cid) + "\n";
		}
		if (!cid_char.empty()) {
			cmap_stream +=
				"endcidchar\n"
				"\n";
		}

		for (std::vector<struct cid_range_s>::const_iterator
				 iterator = cid_range.begin();
			 iterator != cid_range.end(); iterator++) {
			if ((iterator - cid_range.begin()) % 100U == 0U) {
				if (iterator != cid_range.begin()) {
					cmap_stream +=
						"endcidrange\n"
						"\n";
				}

				const unsigned int chunk =
					std::min(100U, static_cast<unsigned int>
							 (cid_range.end() - iterator) / 3);

				snprintf(buffer, 4096, "%u begincidrange\n",
						 chunk);
				cmap_stream += buffer;
			}
			cmap_stream += "<" + utf16be_str(iterator->code_first) +
				"> <" + utf16be_str(iterator->code_last) + "> " +
				uint16_str(iterator->cid) + "\n";
		}
		if (!cid_range.empty()) {
			cmap_stream +=
				"endcidrange\n"
				"\n";
		}

		cmap_stream +=
			"endcmap\n"
			"CMapName currentdict /CMap defineresource pop\n"
			"end\n"
			"end\n"
			"\n"
			"%%EndResource\n"
			"%%EOF";

		std::string cmap;

		snprintf(buffer, 4096,
				 "<<\n"
				 "/CMapName /%s\n",
				 cmap_name.c_str());
		cmap = buffer;
		cmap +=
			"/Type /CMap\n"
			"/CIDSystemInfo\n"
			"<<\n"
			"/Registry (" + registry + ")\n"
			"/Ordering (" + ordering + ")\n"
			"/Supplement " + supplement + "\n"
			">>\n"
			"/WMode 0\n";
		pdf_object["/CMap.dict_part"] = cmap;
		snprintf(buffer, 4096,
				 "/Length %u\n"
				 ">>\n"
				 "stream\n",
				 static_cast<unsigned int>(cmap_stream.size()));
		cmap += buffer;
		pdf_object["/CMap.stream"] = cmap_stream;
		cmap += cmap_stream;
		cmap += "\nendstream\n";

		pdf_object["/CMap"] = cmap;

		// PDF 1.4 Reference, p. 353, Table 5.17

		pdf_object["/Font"] =
			"<<\n"
			"/Type /Font\n"
			"/Subtype /Type0\n"
			"/BaseFont /" + font_name + "\n"
			"/Encoding %u %u R\n"
			"/DescendantFonts [%u %u R]\n"
			">>\n";

		return pdf_object;
	}

	std::map<std::string, std::string>
	font_embed_pdf_t::font_embed_type_2(
		std::string &font_name,
		const std::vector<unsigned char> &font_data)
	{
		return font_embed_cid(font_name, font_data, 0);
	}

	std::map<std::string, std::string>
	font_embed_pdf_t::font_embed_type_42(
		std::string &font_name,
		const std::vector<unsigned char> &font_data)
	{
		return font_embed_cid(font_name, font_data, 2);
	}

}
