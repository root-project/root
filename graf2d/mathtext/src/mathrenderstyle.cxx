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

#include <math.h>
#include <algorithm>
#include "mathrender.h"

/////////////////////////////////////////////////////////////////////

namespace mathtext {

	/////////////////////////////////////////////////////////////////

	float math_text_renderer_t::
	style_size(const unsigned int style) const
	{
		// Intel C++ Compiler 10.1 chokes on this if declared static
		// const and using -O1:
		const float size[math_text_t::item_t::NSTYLE - 1] = {
			script_script_ratio, script_script_ratio,
			script_ratio, script_ratio, 1.0F, 1.0F, 1.0F, 1.0F
		};

		if(style == math_text_t::item_t::STYLE_UNKNOWN ||
		   style >= math_text_t::item_t::NSTYLE)
			return font_size();

		return size[style - 1] * font_size();
	}

	bool math_text_renderer_t::
	is_display_style(const unsigned int style) const
	{
		switch(style) {
		case math_text_t::item_t::STYLE_DISPLAY:
		case math_text_t::item_t::STYLE_DISPLAY_PRIME:
			return true;
		default:
			return false;
		}
	}

	bool math_text_renderer_t::
	is_script_style(const unsigned int style) const
	{
		switch(style) {
		case math_text_t::item_t::STYLE_SCRIPT:
		case math_text_t::item_t::STYLE_SCRIPT_PRIME:
		case math_text_t::item_t::STYLE_SCRIPT_SCRIPT:
		case math_text_t::item_t::STYLE_SCRIPT_SCRIPT_PRIME:
			return true;
		default:
			return false;
		}
	}

	unsigned int math_text_renderer_t::
	prime_style(const unsigned int style) const
	{
		switch(style) {
		case math_text_t::item_t::STYLE_DISPLAY:
			return math_text_t::item_t::STYLE_DISPLAY_PRIME;
		case math_text_t::item_t::STYLE_TEXT:
			return math_text_t::item_t::STYLE_TEXT_PRIME;
		case math_text_t::item_t::STYLE_SCRIPT:
			return math_text_t::item_t::STYLE_SCRIPT_PRIME;
		case math_text_t::item_t::STYLE_SCRIPT_SCRIPT:
			return math_text_t::item_t::STYLE_SCRIPT_SCRIPT_PRIME;
		default:
			return style;
		}
	}

	bool math_text_renderer_t::
	is_prime_style(const unsigned int style) const
	{
		switch(style) {
		case math_text_t::item_t::STYLE_DISPLAY_PRIME:
		case math_text_t::item_t::STYLE_TEXT_PRIME:
		case math_text_t::item_t::STYLE_SCRIPT_PRIME:
		case math_text_t::item_t::STYLE_SCRIPT_SCRIPT_PRIME:
			return true;
		default:
			return false;
		}
	}

	// Knuth, The TeXbook (1986), p. 441

	unsigned int math_text_renderer_t::
	next_superscript_style(const unsigned int style) const
	{
		switch(style) {
		case math_text_t::item_t::STYLE_DISPLAY:
		case math_text_t::item_t::STYLE_TEXT:
			return math_text_t::item_t::STYLE_SCRIPT;
		case math_text_t::item_t::STYLE_DISPLAY_PRIME:
		case math_text_t::item_t::STYLE_TEXT_PRIME:
			return math_text_t::item_t::STYLE_SCRIPT_PRIME;
		case math_text_t::item_t::STYLE_SCRIPT:
		case math_text_t::item_t::STYLE_SCRIPT_SCRIPT:
			return math_text_t::item_t::STYLE_SCRIPT_SCRIPT;
		case math_text_t::item_t::STYLE_SCRIPT_PRIME:
		case math_text_t::item_t::STYLE_SCRIPT_SCRIPT_PRIME:
			return math_text_t::item_t::STYLE_SCRIPT_SCRIPT_PRIME;
		default:
			return style;
		}
	}

	unsigned int math_text_renderer_t::
	next_subscript_style(const unsigned int style) const
	{
		return prime_style(next_superscript_style(style));
	}

	unsigned int math_text_renderer_t::
	next_numerator_style(const unsigned int style) const
	{
		switch(style) {
		case math_text_t::item_t::STYLE_DISPLAY:
			return math_text_t::item_t::STYLE_TEXT;
		case math_text_t::item_t::STYLE_DISPLAY_PRIME:
			return math_text_t::item_t::STYLE_TEXT_PRIME;
		default:
			return next_superscript_style(style);
		}
	}

	unsigned int math_text_renderer_t::
	next_denominator_style(const unsigned int style) const
	{
		return if_else_display(
			style,
			static_cast<unsigned int>(
				math_text_t::item_t::STYLE_TEXT_PRIME),
			next_subscript_style(style));
	}

	// The most elementary building block is the math symbol
	float math_text_renderer_t::x_height(const unsigned int style)
	{
		const unsigned int family = FAMILY_ITALIC;
		const float size = style_size(style);

		set_font_size(size, family);

		bounding_box_t math_symbol_bounding_box =
			bounding_box(L"x", family);

		reset_font_size(family);

		return math_symbol_bounding_box.ascent();
	}

	float math_text_renderer_t::quad(const unsigned int style)
		const
	{
		const float size = style_size(style);

		return size;
	}

	void math_text_renderer_t::
	post_process_atom_type_initial(unsigned int &atom_type) const
	{
		// Rule 5, initial atom
		if(atom_type == math_text_t::atom_t::TYPE_BIN)
			atom_type = math_text_t::atom_t::TYPE_ORD;
	}

	void math_text_renderer_t::
	post_process_atom_type_interior(unsigned int &
									previous_atom_type,
									unsigned int &atom_type) const
	{
		// Rule 5, interior/final atom
		if(atom_type == math_text_t::atom_t::TYPE_BIN)
			switch(previous_atom_type) {
			case math_text_t::atom_t::TYPE_BIN:
			case math_text_t::atom_t::TYPE_OP:
			case math_text_t::atom_t::TYPE_REL:
			case math_text_t::atom_t::TYPE_OPEN:
			case math_text_t::atom_t::TYPE_PUNCT:
				atom_type = math_text_t::atom_t::TYPE_ORD;
				break;
			}
		// Rule 6
		else if(previous_atom_type == math_text_t::atom_t::TYPE_BIN)
			switch(atom_type) {
			case math_text_t::atom_t::TYPE_REL:
			case math_text_t::atom_t::TYPE_CLOSE:
			case math_text_t::atom_t::TYPE_PUNCT:
				previous_atom_type = math_text_t::atom_t::TYPE_ORD;
				break;
			}
	}

	bool math_text_renderer_t::
	valid_accent(bool &vertical_alignment,
				 const std::vector<math_text_t::item_t>::
				 const_iterator &iterator, 
				 const std::vector<math_text_t::item_t>::
				 const_iterator &math_list_end)
		const
	{
		if(iterator->_atom._type == math_text_t::atom_t::TYPE_ACC) {
			std::vector<math_text_t::item_t>::const_iterator
				iterator_next = iterator + 1;

			vertical_alignment = true;
			return iterator_next != math_list_end &&
				iterator_next->_type ==
				math_text_t::item_t::TYPE_ATOM;
		}
		else if(iterator->_atom.is_combining_diacritical()) {
			std::vector<math_text_t::item_t>::const_iterator
				iterator_next = iterator + 1;

			vertical_alignment = false;
			return iterator_next != math_list_end &&
				iterator_next->_type ==
				math_text_t::item_t::TYPE_ATOM;
		}
		else
			return false;
	}

	float math_text_renderer_t::kerning_mu(const float amount) const
	{
		// Rule 2
		return amount / 18.0F * font_size();
	}

	float math_text_renderer_t::
	math_spacing(unsigned int left_type, unsigned int right_type,
				 unsigned int style) const
	{
		const unsigned int left_type_modified =
			left_type <= (unsigned int) math_text_t::atom_t::TYPE_INNER ?
			left_type : (unsigned int) math_text_t::atom_t::TYPE_ORD;
		const unsigned int right_type_modified =
			right_type <= (unsigned int) math_text_t::atom_t::TYPE_INNER ?
			right_type : (unsigned int) math_text_t::atom_t::TYPE_ORD;
		const unsigned int space = math_text_t::atom_t::
			spacing(left_type_modified, right_type_modified,
					is_script_style(style));
		float mu_skip;

		switch(space) {
		case 1:		mu_skip = thin_mu_skip; break;
		case 2:		mu_skip = med_mu_skip; break;
		case 3:		mu_skip = thick_mu_skip; break;
		default:	mu_skip = 0.0F;
		}

		return kerning_mu(mu_skip);
	}

	unsigned int math_text_renderer_t::
	math_family(const math_text_t::math_symbol_t &math_symbol) const
	{
		// Use the text font for Latin, Greek, Cyrillic and the minus
		// sign, and STIX for everything else.
		if(math_symbol._glyph <= L'\u017e' ||
		   (math_symbol._glyph >= L'\u0384' &&
			math_symbol._glyph <= L'\u03ce') ||
		   (math_symbol._glyph >= L'\u0400' &&
			math_symbol._glyph <= L'\u052f') ||
		   math_symbol._glyph == L'\u2212') {
			return math_symbol._family;
		}
		else {
			switch(math_symbol._family) {
			case FAMILY_REGULAR:
				return FAMILY_STIX_REGULAR;
			case FAMILY_ITALIC:
				return FAMILY_STIX_ITALIC;
			case FAMILY_BOLD:
				return FAMILY_STIX_BOLD;
			case FAMILY_BOLD_ITALIC:
				return FAMILY_STIX_BOLD_ITALIC;
			case FAMILY_STIX_REGULAR:
			case FAMILY_STIX_ITALIC:
			case FAMILY_STIX_BOLD:
			case FAMILY_STIX_BOLD_ITALIC:
			case FAMILY_STIX_SIZE_1_REGULAR:
			case FAMILY_STIX_SIZE_1_BOLD:
			case FAMILY_STIX_SIZE_2_REGULAR:
			case FAMILY_STIX_SIZE_2_BOLD:
			case FAMILY_STIX_SIZE_3_REGULAR:
			case FAMILY_STIX_SIZE_3_BOLD:
			case FAMILY_STIX_SIZE_4_REGULAR:
			case FAMILY_STIX_SIZE_4_BOLD:
			case FAMILY_STIX_SIZE_5_REGULAR:
				return math_symbol._family;
			default:
				return FAMILY_STIX_REGULAR;
			}
		}
	}

	void math_text_renderer_t::
	large_family(unsigned long &nfamily, const unsigned int *&family,
				 const math_text_t::math_symbol_t &math_symbol) const
	{
		static const unsigned long nlarge_family = 5;
		static const unsigned int
			large_family_regular[nlarge_family] = {
			FAMILY_STIX_REGULAR,
			FAMILY_STIX_SIZE_1_REGULAR,
			FAMILY_STIX_SIZE_2_REGULAR,
			FAMILY_STIX_SIZE_3_REGULAR,
			FAMILY_STIX_SIZE_4_REGULAR,
		};
		static const unsigned int
			large_family_bold[nlarge_family] = {
			FAMILY_STIX_BOLD,
			FAMILY_STIX_SIZE_1_BOLD,
			FAMILY_STIX_SIZE_2_BOLD,
			FAMILY_STIX_SIZE_3_BOLD,
			FAMILY_STIX_SIZE_4_BOLD,
		};

		nfamily = nlarge_family;
		family = math_symbol.bold() ?
			large_family_bold : large_family_regular;
	}

	void math_text_renderer_t::
	extensible_glyph(wchar_t glyph[4], unsigned long &nrepeat,
					 const math_text_t::math_symbol_t &math_symbol,
					 const unsigned int style, const float height)
	{
		// See Knuth, The METAFONTbook (1986), p. 318
		enum {
			GLYPH_TOP = 0,
			GLYPH_MIDDLE,
			GLYPH_BOTTOM,
			GLYPH_REPEATABLE,
			NGLYPH
		};

		switch(math_symbol._glyph) {
		case L'(':
			glyph[GLYPH_TOP] = L'\u239b';
			glyph[GLYPH_MIDDLE] = L'\0';
			glyph[GLYPH_BOTTOM] = L'\u239d';
			glyph[GLYPH_REPEATABLE] = L'\u239c';
			break;
		case L')':
			glyph[GLYPH_TOP] = L'\u239e';
			glyph[GLYPH_MIDDLE] = L'\0';
			glyph[GLYPH_BOTTOM] = L'\u23a0';
			glyph[GLYPH_REPEATABLE] = L'\u239f';
			break;
		case L'[':
			glyph[GLYPH_TOP] = L'\u23a1';
			glyph[GLYPH_MIDDLE] = L'\0';
			glyph[GLYPH_BOTTOM] = L'\u23a3';
			glyph[GLYPH_REPEATABLE] = L'\u23a2';
			break;
		case L']':
			glyph[GLYPH_TOP] = L'\u23a4';
			glyph[GLYPH_MIDDLE] = L'\0';
			glyph[GLYPH_BOTTOM] = L'\u23a6';
			glyph[GLYPH_REPEATABLE] = L'\u23a5';
			break;
		case L'{':
			glyph[GLYPH_TOP] = L'\u23a7';
			glyph[GLYPH_MIDDLE] = L'\u23a8';
			glyph[GLYPH_BOTTOM] = L'\u23a9';
			glyph[GLYPH_REPEATABLE] = L'\u23aa';
			break;
		case L'|':
			glyph[GLYPH_TOP] = math_symbol._glyph;
			glyph[GLYPH_MIDDLE] = L'\0';
			glyph[GLYPH_BOTTOM] = math_symbol._glyph;
			glyph[GLYPH_REPEATABLE] = math_symbol._glyph;
			break;
		case L'}':
			glyph[GLYPH_TOP] = L'\u23ab';
			glyph[GLYPH_MIDDLE] = L'\u23ac';
			glyph[GLYPH_BOTTOM] = L'\u23ad';
			glyph[GLYPH_REPEATABLE] = L'\u23aa';
			break;
#if 0
		// FIXME: \lmoustache, \rmoustache, \radical require
		// horizontal offsets
		case L'\u211a':		// \lmoustache
			glyph[GLYPH_TOP] = L'\0';
			glyph[GLYPH_MIDDLE] = L'\0';
			glyph[GLYPH_BOTTOM] = L'\u23b7';
			glyph[GLYPH_REPEATABLE] = L'\u23b9';
			break;
		case L'\u23b0':		// \rmoustache
			glyph[GLYPH_TOP] = L'\u239b';
			glyph[GLYPH_MIDDLE] = L'\0';
			glyph[GLYPH_BOTTOM] = L'\u23a0';
			glyph[GLYPH_REPEATABLE] = L'\u239c';
			break;
		case L'\u23b1':		// \radical
			glyph[GLYPH_TOP] = L'\u239e';
			glyph[GLYPH_MIDDLE] = L'\0';
			glyph[GLYPH_BOTTOM] = L'\u239d';
			glyph[GLYPH_REPEATABLE] = L'\u239f';
			break;
#endif
		default:
			glyph[GLYPH_TOP] = L'\0';
			glyph[GLYPH_MIDDLE] = L'\0';
			glyph[GLYPH_BOTTOM] = L'\0';
			glyph[GLYPH_REPEATABLE] = L'\0';
		}

		const unsigned int family = math_symbol._glyph == L'|' ?
			FAMILY_STIX_REGULAR : FAMILY_STIX_SIZE_1_REGULAR;
		const float size = style_size(style);

		if(glyph[GLYPH_REPEATABLE] != L'\0') {
			bounding_box_t bounding_box_sum(0, 0, 0, 0, 0, 0);
			float current_y = 0;

			for(unsigned long i = GLYPH_TOP; i <= GLYPH_BOTTOM;
				i++) {
				if(glyph[i] != L'\0') {
					bounding_box_t glyph_bounding_box =
						math_bounding_box(glyph[i], family, size);

					current_y += glyph_bounding_box.descent();
					bounding_box_sum = bounding_box_sum.
						merge(point_t(0, current_y) +
							  glyph_bounding_box);
					current_y += glyph_bounding_box.ascent();
				}
			}

			const bounding_box_t bounding_box_repeatable =
				math_bounding_box(glyph[GLYPH_REPEATABLE], family,
								  size);
			const float remaining_height =
				height - bounding_box_sum.height();
			const unsigned long repeat_ratio =
				(unsigned long)ceil(
					remaining_height /
					bounding_box_repeatable.height());

			nrepeat = glyph[GLYPH_MIDDLE] == L'\0' ?
				repeat_ratio : ((repeat_ratio + 1UL) >> 1);
		}
		else
			nrepeat = 0;
	}

	// Font parameters
#include "table/mathfontparam.h"

}
