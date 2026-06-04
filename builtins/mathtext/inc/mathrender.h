// -*- mode: c++; -*-

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

#ifndef MATHRENDER_H_
#define MATHRENDER_H_

#include <string>
#include <iostream>
#include <stdint.h>
#include <mathtext/geometry.h>
#include <mathtext/mathtext.h>

namespace mathtext {

#ifdef __INTEL_COMPILER
#pragma warning(push)
#pragma warning(disable: 869)
#endif // __INTEL_COMPILER
	/**
	 * Mathematical layout engine for formulae represented by
	 * math_text_t
	 *
	 * The class math_text_renderer_t is a layout engine based on
	 * TeX's conversion algorithm from a math list to a horizontal
	 * list.
	 *
	 * @see ISO/IEC JTC1/SC2/WG2, ISO/IEC 10646:2003/Amd.2:2006 (ISO,
	 * Geneva, 2006).
	 * @see D. E. Knuth, The TeXbook (Addision-Wesley, Cambridge, MA,
	 * 1986).
	 * @see D. E. Knuth, The METAFONTbook (Addision-Wesley, Cambridge,
	 * MA, 1986).
	 * @see W. Schmidt, The macro package lucimatx (2005),
	 * unpublished.
	 * @see W. Schmidt, Using the MathTime Professional II fonts with
	 * LaTeX (2006), unpublished.
	 * @see B. Beeton, A. Freytag, M. Sargent III, Unicode support for
	 * mathematics, Unicode Technical Report #25
	 * @author Yue Shi Lai <ylai@phys.columbia.edu>
	 * @version 1.0
	 */
	class math_text_renderer_t {
	public:
		enum {
			DIRECTION_LEFT_TO_RIGHT = 0,
			DIRECTION_RIGHT_TO_LEFT,
			DIRECTION_TOP_TO_BOTTOM
		};
		enum {
			MATH_STYLE_LATIN = 0,
			MATH_STYLE_MAGHREB
		};
		enum {
			FAMILY_PLAIN = 0,
			FAMILY_REGULAR,
			FAMILY_ITALIC,
			FAMILY_BOLD,
			FAMILY_BOLD_ITALIC,
			FAMILY_STIX_REGULAR,
			FAMILY_STIX_ITALIC,
			FAMILY_STIX_BOLD,
			FAMILY_STIX_BOLD_ITALIC,
			FAMILY_STIX_SIZE_1_REGULAR,
			FAMILY_STIX_SIZE_1_BOLD,
			FAMILY_STIX_SIZE_2_REGULAR,
			FAMILY_STIX_SIZE_2_BOLD,
			FAMILY_STIX_SIZE_3_REGULAR,
			FAMILY_STIX_SIZE_3_BOLD,
			FAMILY_STIX_SIZE_4_REGULAR,
			FAMILY_STIX_SIZE_4_BOLD,
			FAMILY_STIX_SIZE_5_REGULAR,
			NFAMILY
		};
	private:
		/////////////////////////////////////////////////////////////
		// Font parameter
		static const float script_ratio;
		static const float script_script_ratio;
		static const float thin_mu_skip;
		static const float med_mu_skip;
		static const float thick_mu_skip;
		static const float delimiter_factor;
		static const float delimiter_shortfall;
		/////////////////////////////////////////////////////////////
		static const float num_1;
		static const float num_2;
		static const float num_3;
		static const float denom_1;
		static const float denom_2;
		static const float sup_1;
		static const float sup_2;
		static const float sup_3;
		static const float sub_1;
		static const float sub_2;
		static const float sup_drop;
		static const float sub_drop;
		static const float delim_1;
		static const float delim_2;
		static const float axis_height;
		static const float default_rule_thickness;
		static const float big_op_spacing_1;
		static const float big_op_spacing_2;
		static const float big_op_spacing_3;
		static const float big_op_spacing_4;
		static const float big_op_spacing_5;
		/////////////////////////////////////////////////////////////
		static const float radical_rule_thickness;
		static const float large_operator_display_scale;
		/////////////////////////////////////////////////////////////
		static const float baselineskip_factor;
		/////////////////////////////////////////////////////////////
		// Token
		class math_token_t {
		public:
			point_t _offset;
			bounding_box_t _bounding_box;
			union {
				unsigned int _style;
				struct {
					wchar_t _glyph;
					unsigned int _family;
					float _size;
				} _extensible;
			};
			float _delimiter_height;
			inline math_token_t(
				const bounding_box_t bounding_box,
				const unsigned int style,
				const float delimiter_height = 0.0F)
				: _offset(0, 0), _bounding_box(bounding_box),
				  _style(style), _delimiter_height(delimiter_height)
			{
			}
			inline math_token_t(
				const point_t offset,
				const bounding_box_t bounding_box,
				const unsigned int style,
				const float delimiter_height = 0.0F)
				: _offset(offset), _bounding_box(bounding_box),
				  _style(style), _delimiter_height(delimiter_height)
			{
			}
			inline math_token_t(
				const bounding_box_t bounding_box,
				const wchar_t glyph, const unsigned int family,
				const float size)
				: _offset(0, 0), _bounding_box(bounding_box),
				  _delimiter_height(0.0F)
			{
				_extensible._glyph = glyph;
				_extensible._family = family;
				_extensible._size = size;
			}
			inline math_token_t(
				const point_t offset,
				const bounding_box_t bounding_box,
				const wchar_t glyph, const unsigned int family,
				const float size)
				: _offset(offset), _bounding_box(bounding_box),
				  _delimiter_height(0.0F)
			{
				_extensible._glyph = glyph;
				_extensible._family = family;
				_extensible._size = size;
			}
		};
		/////////////////////////////////////////////////////////////
		// Style test and change
		float style_size(const unsigned int style) const;
		bool is_display_style(const unsigned int style) const;
		bool is_script_style(const unsigned int style) const;
		unsigned int prime_style(const unsigned int style) const;
		bool is_prime_style(const unsigned int style) const;
		template<typename value_t>
		inline value_t
		if_else_display(const unsigned int style,
						const value_t display_value,
						const value_t otherwise_value) const
		{
			switch (style) {
			case math_text_t::item_t::STYLE_DISPLAY:
			case math_text_t::item_t::STYLE_DISPLAY_PRIME:
				return display_value;
			default:
				return otherwise_value;
			}
		}
		unsigned int next_superscript_style(const unsigned int style)
			const;
		unsigned int next_subscript_style(const unsigned int style)
			const;
		unsigned int next_numerator_style(const unsigned int style)
			const;
		unsigned int next_denominator_style(const unsigned int style)
			const;
		/////////////////////////////////////////////////////////////
		float x_height(const unsigned int style);
		float quad(const unsigned int style) const;
		unsigned int
		math_family(const math_text_t::math_symbol_t &math_symbol)
			const;
		void post_process_atom_type_initial(unsigned int &atom_type)
			const;
		void post_process_atom_type_interior(
			unsigned int &previous_atom_type,
			unsigned int &atom_type)
			const;
		bool valid_accent(
			bool &vertical_alignment,
			const std::vector<math_text_t::item_t>::const_iterator &
			iterator, 
			const std::vector<math_text_t::item_t>::const_iterator &
			math_list_end) const;
		float kerning_mu(float amount) const;
		float math_spacing(
			unsigned int left_type, unsigned int right_type,
			unsigned int style) const;
	protected:
		virtual affine_transform_t
		transform_logical_to_pixel(void) const = 0;
		virtual affine_transform_t
		transform_pixel_to_logical(void) const = 0;
		/////////////////////////////////////////////////////////////
		// Box rendering
		bounding_box_t math_bounding_box(
			const math_text_t::box_t &box, const unsigned int style);
		void math_text(
			const point_t origin, const math_text_t::box_t &box,
			const unsigned int style, const bool render_structure);
		/////////////////////////////////////////////////////////////
		// Symbol rendering
		static bool is_wgl_4(const wchar_t c);
		static bool is_left_to_right(const wchar_t c);
		static bool is_right_to_left(const wchar_t c);
		static bool is_top_to_bottom(const wchar_t c);
		static bool is_cyrillic(const wchar_t c);
		static bool is_cjk(const wchar_t c);
		static bool is_cjk_punctuation_open(const wchar_t c);
		static bool is_cjk_punctuation_closed(const wchar_t c);
		bounding_box_t math_bounding_box(
			const wchar_t &glyph, const unsigned int family,
			const float size);
		void math_text(
			const point_t origin, const wchar_t &glyph,
			const unsigned int family, const float size,
			const bool render_structure);
		bounding_box_t math_bounding_box(
			const math_text_t::math_symbol_t &math_symbol,
			const unsigned int style);
		void math_text(
			const point_t origin,
			const math_text_t::math_symbol_t &math_symbol,
			const unsigned int style, const bool render_structure);
		/////////////////////////////////////////////////////////////
		// Extensible glyph rendering
		void large_family(
			unsigned long &nfamily, const unsigned int *&family,
			const math_text_t::math_symbol_t &math_symbol) const;
		void extensible_glyph(
			wchar_t glyph[4], unsigned long &nrepeat,
			const math_text_t::math_symbol_t &math_symbol,
			const unsigned int style, const float height);
		std::vector<math_token_t> math_tokenize(
			const math_text_t::math_symbol_t &math_symbol,
			const unsigned int style, const float height);
		bounding_box_t math_bounding_box(
			const math_text_t::math_symbol_t &math_symbol,
			const unsigned int style, const float height);
		void math_text(
			const point_t origin,
			const math_text_t::math_symbol_t &math_symbol,
			const unsigned int style, const float height,
			const bool render_structure);
		/////////////////////////////////////////////////////////////
		// Math list rendering
		std::vector<math_token_t> math_tokenize(
			const std::vector<math_text_t::item_t>::const_iterator &
			math_list_begin,
			const std::vector<math_text_t::item_t>::const_iterator &
			math_list_end,
			unsigned int style);
		bounding_box_t math_bounding_box(
			const std::vector<math_text_t::item_t>::const_iterator &
			math_list_begin,
			const std::vector<math_text_t::item_t>::const_iterator &
			math_list_end,
			unsigned int style);
		void math_text(
			const point_t origin,
			const std::vector<math_text_t::item_t>::const_iterator &
			math_list_begin,
			const std::vector<math_text_t::item_t>::const_iterator &
			math_list_end,
			const unsigned int style, const bool render_structure);
		/////////////////////////////////////////////////////////////
		// Field rendering
		bounding_box_t math_bounding_box(
			const math_text_t::field_t &field,
			const unsigned int style);
		void math_text(
			const point_t origin, const math_text_t::field_t &field,
			const unsigned int style, const bool render_structure);
		/////////////////////////////////////////////////////////////
		// Atom rendering
		std::vector<math_token_t> math_tokenize(
			const math_text_t::atom_t &atom, unsigned int style);
		bounding_box_t math_bounding_box(
			const math_text_t::atom_t &atom,
			const unsigned int style);
		void math_text(
			const point_t origin, const math_text_t::atom_t &atom,
			const unsigned int style, const bool render_structure);
		/////////////////////////////////////////////////////////////
	public:
		/////////////////////////////////////////////////////////////
		// Constructor and destructor
		inline math_text_renderer_t(void)
		{
		}
		inline virtual ~math_text_renderer_t(void)
		{
		}
		/////////////////////////////////////////////////////////////
		// Virtual functions
		virtual float font_size(
			const unsigned int family = FAMILY_PLAIN) const = 0;
		virtual void set_font_size(
			const float size, const unsigned int family) = 0;
		virtual void set_font_size(const float size) = 0;
		virtual void reset_font_size(
			const unsigned int family) = 0;
		virtual void point(const float x, const float y) = 0;
		virtual void filled_rectangle(
			const bounding_box_t &bounding_box) = 0;
		virtual void rectangle(
			const bounding_box_t &bounding_box) = 0;
		virtual bounding_box_t bounding_box(
			const std::wstring string,
			const unsigned int family = FAMILY_PLAIN) = 0;
		virtual void text_raw(
			const float x, const float y, const std::wstring string,
			const unsigned int family = FAMILY_PLAIN) = 0;
		virtual void text_with_bounding_box(
			const float x, const float y, const std::wstring string,
			const unsigned int family = FAMILY_PLAIN) = 0;
		/////////////////////////////////////////////////////////////
		// Interface
		bounding_box_t bounding_box(
			const math_text_t &math_text,
			const bool display_style = false);
		void text(
			const float x, const float y,
			const math_text_t &math_text,
			const bool display_style = false);
		/////////////////////////////////////////////////////////////
		inline float default_axis_height(
			const bool display_style = false) const
		{
			return axis_height * style_size(
				math_text_t::item_t::STYLE_TEXT);
		}
		/////////////////////////////////////////////////////////////
	};
#ifdef __INTEL_COMPILER
#pragma warning(pop)
#endif // __INTEL_COMPILER

}

#endif // MATHRENDER_H_
