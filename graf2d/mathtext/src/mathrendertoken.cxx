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

#include <algorithm>
#include "../inc/mathrender.h"

/////////////////////////////////////////////////////////////////////

namespace mathtext {

   /////////////////////////////////////////////////////////////////
   // Math List Tokenization

   std::vector<math_text_renderer_t::math_token_t>
   math_text_renderer_t::
   math_tokenize(const math_text_t::math_symbol_t &math_symbol,
                 const unsigned int style, const float height)
   {
      if(math_symbol._glyph == L'.')
         return std::vector<math_text_renderer_t::math_token_t>();

      enum {
         GLYPH_TOP = 0,
         GLYPH_MIDDLE,
         GLYPH_BOTTOM,
         GLYPH_REPEATABLE,
         NGLYPH
      };
      const unsigned int family = math_family(math_symbol);
      const float size = style_size(style);
      const bounding_box_t normal_bounding_box =
      math_bounding_box(math_symbol._glyph, family, size);

      if(normal_bounding_box.height() >= height) {
         const math_token_t token(normal_bounding_box,
                                  math_symbol._glyph, family,
                                  size);

         return std::vector<math_token_t>(1, token);
      }

      unsigned long nmath_symbol_large_family;
      const unsigned int *math_symbol_large_family;

      large_family(nmath_symbol_large_family,
                   math_symbol_large_family, math_symbol);

      for(unsigned int i = 0; i < nmath_symbol_large_family; i++) {
         const bounding_box_t large_bounding_box =
         math_bounding_box(math_symbol._glyph,
                           math_symbol_large_family[i], size);

         if(large_bounding_box.height() >= height) {
            const math_token_t token(large_bounding_box,
                                     math_symbol._glyph,
                                     math_symbol_large_family[i],
                                     size);

            return std::vector<math_token_t>(1, token);
         }
      }

      // Extensible glyph after the TFM charlist mechanism, see
      // Knuth, The METAFONTbook (1986), p. 317f.

      wchar_t glyph[NGLYPH];
      unsigned long nrepeat;

      extensible_glyph(glyph, nrepeat, math_symbol, style, height);

      if(glyph[GLYPH_BOTTOM] != L'\0' &&
         glyph[GLYPH_REPEATABLE] != L'\0') {
         static const unsigned int extensible_family =
         math_symbol._glyph == L'|' ? FAMILY_STIX_REGULAR :
         FAMILY_STIX_SIZE_1_REGULAR;
         const bounding_box_t bounding_box_bottom =
         math_bounding_box(glyph[GLYPH_BOTTOM],
                           extensible_family, size);
         std::vector<math_token_t> token_list;

         token_list.push_back(math_token_t(
                                           bounding_box_bottom, glyph[GLYPH_BOTTOM],
                                           extensible_family, size));
         float current_y = 0;

         current_y += bounding_box_bottom.ascent();
         for(unsigned long i = 0; i < nrepeat; i++) {
            const bounding_box_t bounding_box_repeatable =
            math_bounding_box(glyph[GLYPH_REPEATABLE],
                              extensible_family, size);

            current_y += bounding_box_repeatable.descent();
            token_list.push_back(math_token_t(
                                              point_t(0, current_y), bounding_box_repeatable,
                                              glyph[GLYPH_REPEATABLE], extensible_family,
                                              size));
            current_y += bounding_box_repeatable.ascent();
         }
         if(glyph[GLYPH_MIDDLE] != L'\0') {
            const bounding_box_t bounding_box_middle =
            math_bounding_box(glyph[GLYPH_MIDDLE],
                              extensible_family, size);

            current_y += bounding_box_middle.descent();
            token_list.push_back(math_token_t(
                                              point_t(0, current_y), bounding_box_middle,
                                              glyph[GLYPH_MIDDLE], extensible_family, size));
            current_y += bounding_box_middle.ascent();
            for(unsigned long i = 0; i < nrepeat; i++) {
               const bounding_box_t bounding_box_repeatable =
               math_bounding_box(glyph[GLYPH_REPEATABLE],
                                 extensible_family, size);

               current_y += bounding_box_repeatable.descent();
               token_list.push_back(math_token_t(
                                                 point_t(0, current_y),
                                                 bounding_box_repeatable,
                                                 glyph[GLYPH_REPEATABLE], extensible_family,
                                                 size));
               current_y += bounding_box_repeatable.ascent();
            }
         }

         const bounding_box_t bounding_box_top =
         math_bounding_box(glyph[GLYPH_TOP],
                           extensible_family, size);

         current_y += bounding_box_top.descent();
         token_list.push_back(math_token_t(
                                           point_t(0, current_y), bounding_box_top,
                                           glyph[GLYPH_TOP], extensible_family, size));

         return token_list;
      }

      const math_token_t token(normal_bounding_box,
                               math_symbol._glyph, family, size);

      return std::vector<math_token_t>(1, token);
   }

   std::vector<math_text_renderer_t::math_token_t>
   math_text_renderer_t::
   math_tokenize(const std::vector<math_text_t::item_t>::
                 const_iterator &math_list_begin,
                 const std::vector<math_text_t::item_t>::
                 const_iterator &math_list_end,
                 const unsigned int style)
   {
      const float size = style_size(style);
      const float style_axis_height = axis_height * size;
      unsigned int previous_atom_type =
      math_text_t::atom_t::TYPE_UNKNOWN;
      float current_x = 0;
      bool has_accent = false;

      // Rule 19
      std::vector<math_text_t::item_t>::const_iterator
      math_list_begin_interior = math_list_begin;
      std::vector<math_text_t::item_t>::const_iterator
      math_list_end_interior = math_list_end;
      static const math_text_t::item_t fraction_item =
      math_text_t::item_t::TYPE_GENERALIZED_FRACTION;
      const std::vector<math_text_t::item_t>::const_iterator
      fraction_iterator =
      std::find(math_list_begin_interior,
                math_list_end_interior, fraction_item);
      const bool generalized_fraction =
      fraction_iterator != math_list_end_interior;
      bool delimiter = false;
      float delimiter_height = 0.f;
      bounding_box_t bounding_box_delimiter_left(0, 0, 0, 0, 0, 0);
      bounding_box_t bounding_box_delimiter_right(0, 0, 0, 0, 0, 0);

      if(math_list_begin->_type ==
         math_text_t::item_t::TYPE_BOUNDARY &&
         (math_list_end - 1)->_type ==
         math_text_t::item_t::TYPE_BOUNDARY) {
         ++math_list_begin_interior;
         --math_list_end_interior;
         delimiter = true;

         const bounding_box_t bounding_box_interior =
         math_bounding_box(math_list_begin_interior,
                           math_list_end_interior, style);
         const float extension =
         std::max(bounding_box_interior.ascent() -
                  style_axis_height,
                  bounding_box_interior.descent() +
                  style_axis_height);

         delimiter_height = extension * 0.002F * delimiter_factor;
         if(generalized_fraction)
            // Rule 15e
            delimiter_height = std::max(delimiter_height,
                                        if_else_display(style, delim_1, delim_2) * size);

         bounding_box_delimiter_left =
         math_bounding_box(math_list_begin->_atom.
                           _nucleus._math_symbol,
                           style, delimiter_height);
         bounding_box_delimiter_right =
         math_bounding_box((math_list_end - 1)->_atom.
                           _nucleus._math_symbol,
                           style, delimiter_height);
      }

      std::vector<math_token_t> token_list;

      if(delimiter) {
         // Standard advance mode

         // post_process_atom_type_initial(atom_type) is not
         // necessary, since the current item is guaranteed not to
         // be of type Bin.

         // FIXME: Must be in starting style.

         const float shift_delimiter_left = style_axis_height -
         bounding_box_delimiter_left.vertical_center();

         token_list.push_back(math_token_t(
                                           point_t(current_x, shift_delimiter_left),
                                           bounding_box_delimiter_left, style,
                                           delimiter_height));

         current_x += bounding_box_delimiter_left.advance();
         previous_atom_type = math_list_begin->_atom._type;
      }

      if(generalized_fraction) {
         // Rule 15a, 15b
         const float thickness = fraction_iterator->_length *
         default_rule_thickness * size;
         const bounding_box_t numerator_bounding_box =
         math_bounding_box(math_list_begin_interior,
                           fraction_iterator,
                           next_numerator_style(style));
         const bounding_box_t denominator_bounding_box =
         math_bounding_box(fraction_iterator + 1,
                           math_list_end_interior,
                           next_denominator_style(style));
         const float min_shift_up =
         if_else_display(style, num_1,
                         thickness > 0 ? num_2 : num_3) *
         size;
         const float min_shift_down =
         if_else_display(style, denom_1, denom_2) * size;
         float shift_up;
         float shift_down;

         if(thickness <= 0) {
            // Rule 15c (\atop)
            const float min_clearance =
            if_else_display(style, 7.0F, 3.0F) *
            default_rule_thickness * size;
            const float actual_clearance =
            (min_shift_up -
             numerator_bounding_box.descent()) -
            (denominator_bounding_box.ascent() -
             min_shift_down);

            if(actual_clearance < min_clearance) {
               const float difference = 0.5F *
               (min_clearance - actual_clearance);

               shift_up = min_shift_up + difference;
               shift_down = min_shift_down + difference;
            }
            else {
               shift_up = min_shift_up;
               shift_down = min_shift_down;
            }
         }
         else {
            // Rule 15d (\over)
            const float min_bar_clearance =
            if_else_display(style, 3.0F, 1.0F) * thickness;
            const float actual_numerator_clearance =
            (min_shift_up -
             numerator_bounding_box.descent()) -
            (style_axis_height + 0.5F * thickness);
            const float actual_denominator_clearance =
            (style_axis_height - 0.5F * thickness) -
            (denominator_bounding_box.ascent() -
             min_shift_down);

            if(actual_numerator_clearance <
               min_bar_clearance) {
               const float difference =
               (min_bar_clearance -
                actual_numerator_clearance);

               shift_up = min_shift_up + difference;
            }
            else
               shift_up = min_shift_up;
            if(actual_denominator_clearance <
               min_bar_clearance) {
               const float difference =
               (min_bar_clearance -
                actual_denominator_clearance);

               shift_down = min_shift_down + difference;
            }
            else
               shift_down = min_shift_down;
         }

         const float horizontal_center_difference =
         numerator_bounding_box.horizontal_center() -
         denominator_bounding_box.horizontal_center();
         float horizontal_shift_numerator;
         float horizontal_shift_denominator;

         if(horizontal_center_difference > 0) {
            horizontal_shift_numerator = 0;
            horizontal_shift_denominator =
            horizontal_center_difference;
         }
         else {
            horizontal_shift_numerator =
            -horizontal_center_difference;
            horizontal_shift_denominator = 0;
         }

         token_list.push_back(math_token_t(
                                           point_t(current_x + horizontal_shift_denominator,
                                                   -shift_down),
                                           denominator_bounding_box,
                                           next_denominator_style(style)));
         if(thickness > 0) {
#if 0
            const float constrained_thickness =
            std::max(1.0F, thickness);
#endif
            const float left =
            std::min(numerator_bounding_box.left(),
                     denominator_bounding_box.left());
            const float right =
            std::max(numerator_bounding_box.right(),
                     denominator_bounding_box.right());

            token_list.push_back(math_token_t(
                                              point_t(current_x + left,
                                                      style_axis_height - 0.5F * thickness),
                                              bounding_box_t(0, 0, right - left, thickness,
                                                             0, 0),
                                              style));
         }
         token_list.push_back(math_token_t(
                                           point_t(current_x + horizontal_shift_numerator,
                                                   shift_up),
                                           numerator_bounding_box,
                                           next_numerator_style(style)));

         const float advance =
         std::max(horizontal_shift_numerator +
                  numerator_bounding_box.left() +
                  numerator_bounding_box.right(),
                  horizontal_shift_denominator +
                  denominator_bounding_box.left() +
                  denominator_bounding_box.right());

         current_x += advance;
      }
      else
         // Incrementally process a math list
         for (std::vector<math_text_t::item_t>::const_iterator iterator = math_list_begin_interior;
              iterator != math_list_end_interior; ++iterator) {
            unsigned int atom_type;
            bounding_box_t item_bounding_box;
            unsigned int current_style = has_accent ?
            prime_style(style) : style;
            bool accent;
            bool vertical_alignment;

            switch(iterator->_type) {
               case math_text_t::item_t::TYPE_ATOM:
                  atom_type = iterator->_atom._type;
                  item_bounding_box =
                  math_bounding_box(iterator->_atom,
                                    current_style);
                  accent = valid_accent(vertical_alignment,
                                        iterator, math_list_end);
                  if(accent) {
                     // Accent advance mode
                     const std::vector<math_text_t::item_t>::
                     const_iterator iterator_next =
                     iterator + 1;
                     const bounding_box_t next_item_bounding_box =
                     math_bounding_box(iterator_next->_atom,
                                       style);
                     const float horizontal_shift =
                     (iterator == math_list_begin ? 0.0F :
                      math_spacing(previous_atom_type,
                                   atom_type, style)) +
                     next_item_bounding_box.
                     horizontal_center() +
                     0.5F * next_item_bounding_box.
                     italic_correction() -
                     item_bounding_box.horizontal_center();
                     const float vertical_shift =
                     vertical_alignment ?
                     std::max(0.0F,
                              next_item_bounding_box.ascent() -
                              x_height(style)) : 0.0F;

                     token_list.push_back(math_token_t(
                                                       point_t(current_x + horizontal_shift,
                                                               vertical_shift),
                                                       item_bounding_box, style));
                     has_accent = true;
                  }
                  else {
                     // Standard advance mode
                     if(iterator == math_list_begin)
                        post_process_atom_type_initial(atom_type);
                     else {
                        post_process_atom_type_interior(
                                                        previous_atom_type, atom_type);

                        const float horizontal_shift =
                        math_spacing(previous_atom_type,
                                     atom_type,
                                     current_style);

                        current_x +=
                        horizontal_shift;
                     }

                     token_list.push_back(math_token_t(
                                                       point_t(current_x, 0), item_bounding_box,
                                                       current_style));
                     current_x += item_bounding_box.advance();
                     has_accent = false;
                     previous_atom_type = atom_type;
                  }
                  break;
               case math_text_t::item_t::TYPE_KERN:
                  // Rule 2
                  current_x += kerning_mu(iterator->_length);
                  break;
            }
         }

      // Rule 19 (again)
      if(delimiter) {
         unsigned int atom_type =
         (math_list_end - 1)->_atom._type;
         // Standard advance mode

         // FIXME: Must be in starting style.

         post_process_atom_type_interior(previous_atom_type,
                                         atom_type);

         const float horizontal_shift =
         math_spacing(previous_atom_type, atom_type, style);

         current_x += horizontal_shift;

         const float shift_delimiter_right = style_axis_height -
         bounding_box_delimiter_right.vertical_center();

         token_list.push_back(math_token_t(
                                           point_t(current_x, shift_delimiter_right),
                                           bounding_box_delimiter_right, style,
                                           delimiter_height));
      }

      return token_list;
   }

   std::vector<math_text_renderer_t::math_token_t>
   math_text_renderer_t::
   math_tokenize(const math_text_t::atom_t &atom,
                 const unsigned int style)
   {
      const float size = style_size(style);
      std::vector<math_text_renderer_t::math_token_t> token_list;
      bounding_box_t nucleus_bounding_box;
      float current_x = 0;
      float nucleus_shift_up = 0;

      // Rule 11
      if(atom._type == math_text_t::atom_t::TYPE_RAD) {
         float style_radical_rule_thickness =
         default_rule_thickness * size;
         const float min_clearance =
         style_radical_rule_thickness + 0.25F *
         if_else_display(style, x_height(style),
                         style_radical_rule_thickness);
         const bounding_box_t bounding_box_radicand =
         math_bounding_box(atom._nucleus,
                           prime_style(style));
         const float min_delimiter_height =
         bounding_box_radicand.height() + min_clearance +
         style_radical_rule_thickness;
         const math_text_t::math_symbol_t
         symbol_surd("\\surd", style);
         const bounding_box_t bounding_box_surd =
         math_bounding_box(symbol_surd, style,
                           min_delimiter_height);
         const float surd_intrinsic_shift_down =
         bounding_box_surd.ascent() -
         style_radical_rule_thickness;
         const float modified_descent =
         bounding_box_surd.descent() +
         surd_intrinsic_shift_down;
         const float clearance = modified_descent >
         bounding_box_radicand.height() + min_clearance ?
         0.5F * (min_clearance + modified_descent -
                 bounding_box_radicand.height()) :
         min_clearance;

         if(!atom._index.empty()) {
            // The positive space of 5 mu to the index is hard
            // wired in plain.tex
            current_x += (5.0F / 18.0F) * size;

            const bounding_box_t bounding_box_index =
            math_bounding_box(atom._index,
                              math_text_t::item_t::STYLE_SCRIPT_SCRIPT);
            const float radical_height =
            (std::max(math_bounding_box(
                                        atom._index, prime_style(style)).ascent(),
                      bounding_box_radicand.ascent() +
                      clearance +
                      2.0F * style_radical_rule_thickness) -
             std::max(math_bounding_box(
                                        atom._index, prime_style(style)).descent(),
                      bounding_box_radicand.descent()));

            nucleus_bounding_box =
            point_t(current_x, 0.6F * radical_height) +
            math_bounding_box(
                              atom._index,
                              math_text_t::item_t::STYLE_SCRIPT_SCRIPT);
            token_list.push_back(math_token_t(
                                              point_t(current_x, 0.6F * radical_height),
                                              bounding_box_index,
                                              math_text_t::item_t::STYLE_SCRIPT_SCRIPT));
            // The negative space of 10 mu to the surd is hard
            // wired in plain.tex
            current_x += bounding_box_index.advance() -
            (10.0F / 18.0F) * size;
         }

         const float radicand_ascent_clearance =
         bounding_box_radicand.ascent() + clearance;

         nucleus_bounding_box = nucleus_bounding_box.merge(
                                                           point_t(current_x,
                                                                   radicand_ascent_clearance -
                                                                   surd_intrinsic_shift_down) +
                                                           bounding_box_surd);
         token_list.push_back(math_token_t(
                                           point_t(current_x,
                                                   radicand_ascent_clearance -
                                                   surd_intrinsic_shift_down),
                                           bounding_box_surd, style, min_delimiter_height));
         current_x += bounding_box_surd.advance();

         const float constrained_thickness =
#if 0
         // Pixel height constraint
         std::max(1.0F, style_radical_rule_thickness)
#else
         style_radical_rule_thickness
#endif
         ;
         static const float surd_rule_correction_x = -1.5F;
         static const float surd_rule_correction_y = -0.5F;

         const point_t origin_rule =
         point_t(current_x, radicand_ascent_clearance);
         const bounding_box_t bounding_box_rule =
         bounding_box_t(surd_rule_correction_x *
                        constrained_thickness,
                        surd_rule_correction_y *
                        constrained_thickness,
                        bounding_box_radicand.advance(),
                        (surd_rule_correction_y + 1.0F) *
                        constrained_thickness,
                        0, 0);

         nucleus_bounding_box = nucleus_bounding_box.merge(
                                                           origin_rule + bounding_box_rule);
         token_list.push_back(math_token_t(
                                           origin_rule, bounding_box_rule, style));

         const bounding_box_t bounding_box_clearance =
         bounding_box_t(0, -2.0F * constrained_thickness,
                        bounding_box_radicand.advance(),
                        constrained_thickness,
                        0, 0);

         nucleus_bounding_box = nucleus_bounding_box.merge(
                                                           origin_rule + bounding_box_clearance);
         token_list.push_back(math_token_t(
                                           origin_rule, bounding_box_clearance, style));

         const point_t origin_radicand = point_t(current_x, 0);

         nucleus_bounding_box = nucleus_bounding_box.merge(
                                                           origin_radicand + bounding_box_radicand);
         token_list.push_back(math_token_t(
                                           origin_radicand, bounding_box_radicand,
                                           prime_style(style)));
         current_x += bounding_box_radicand.advance();
      }
      else if(atom._type == math_text_t::atom_t::TYPE_OP) {
         const bool limits =
         atom._limits == math_text_t::atom_t::LIMITS_LIMITS ||
         (atom._limits ==
          math_text_t::atom_t::LIMITS_DISPLAYLIMITS &&
          is_display_style(style));

         if(atom._nucleus._type ==
            math_text_t::field_t::TYPE_MATH_SYMBOL) {
            // Rule 13
            nucleus_bounding_box = math_bounding_box(
                                                     atom._nucleus._math_symbol._glyph,
                                                     FAMILY_STIX_REGULAR,
                                                     size * if_else_display(style,
                                                                            large_operator_display_scale, 1.0F));
            nucleus_shift_up =
            axis_height * size -
            nucleus_bounding_box.vertical_center();
            if(limits && atom._subscript.empty())
               nucleus_bounding_box.advance() +=
               nucleus_bounding_box.italic_correction();
         }
         else
            nucleus_bounding_box =
            math_bounding_box(atom._nucleus, style);

         if(limits && !(atom._superscript.empty() &&
                        atom._subscript.empty())) {
            // Rule 13a
            const unsigned int superscript_style =
            next_superscript_style(style);
            const unsigned int subscript_style =
            next_subscript_style(style);

            if(atom._superscript.empty()) {
               const bounding_box_t subscript_bounding_box =
               math_bounding_box(atom._subscript,
                                 subscript_style);
               const float shift_right =
               nucleus_bounding_box.horizontal_center() -
               subscript_bounding_box.horizontal_center() -
               0.5F *
               nucleus_bounding_box.italic_correction();
               const float clearance =
               std::max(big_op_spacing_2 * size,
                        big_op_spacing_4 * size -
                        subscript_bounding_box.ascent());
               const float shift_down =
               nucleus_bounding_box.descent() + clearance +
               subscript_bounding_box.ascent() -
               nucleus_shift_up;

               if(shift_right >= 0) {
                  token_list.push_back(math_token_t(
                                                    point_t(0, nucleus_shift_up),
                                                    nucleus_bounding_box, style));
                  token_list.push_back(math_token_t(
                                                    point_t(shift_right, -shift_down),
                                                    subscript_bounding_box,
                                                    subscript_style));
               }
               else {
                  token_list.push_back(math_token_t(
                                                    point_t(-shift_right,
                                                            nucleus_shift_up),
                                                    nucleus_bounding_box, style));
                  token_list.push_back(math_token_t(
                                                    point_t(0, -shift_down),
                                                    subscript_bounding_box,
                                                    subscript_style));
               }
               return token_list;
            }
            if(atom._subscript.empty()) {
               const bounding_box_t superscript_bounding_box =
               math_bounding_box(atom._superscript,
                                 superscript_style);
               const float shift_right =
               nucleus_bounding_box.horizontal_center() -
               superscript_bounding_box.
               horizontal_center() + 0.5F *
               nucleus_bounding_box.italic_correction();
               const float clearance =
               std::max(big_op_spacing_1 * size,
                        big_op_spacing_3 * size -
                        superscript_bounding_box.descent());
               const float shift_up =
               nucleus_bounding_box.ascent() + clearance +
               superscript_bounding_box.descent() +
               nucleus_shift_up;

               if(shift_right >= 0) {
                  token_list.push_back(math_token_t(
                                                    point_t(0, nucleus_shift_up),
                                                    nucleus_bounding_box, style));
                  token_list.push_back(math_token_t(
                                                    point_t(shift_right, shift_up),
                                                    superscript_bounding_box,
                                                    superscript_style));
               }
               else {
                  token_list.push_back(math_token_t(
                                                    point_t(-shift_right,
                                                            nucleus_shift_up),
                                                    nucleus_bounding_box, style));
                  token_list.push_back(math_token_t(
                                                    point_t(0, shift_up),
                                                    superscript_bounding_box,
                                                    superscript_style));
               }
               return token_list;
            }
            const bounding_box_t superscript_bounding_box =
            math_bounding_box(atom._superscript,
                              superscript_style);
            const bounding_box_t subscript_bounding_box =
            math_bounding_box(atom._subscript,
                              subscript_style);
            const float shift_right_superscript =
            nucleus_bounding_box.horizontal_center() -
            superscript_bounding_box.horizontal_center() +
            0.5F * nucleus_bounding_box.italic_correction();
            const float shift_right_subscript =
            nucleus_bounding_box.horizontal_center() -
            subscript_bounding_box.horizontal_center() -
            0.5F * nucleus_bounding_box.italic_correction();
            const float clearance_superscript =
            std::max(big_op_spacing_1 * size,
                     big_op_spacing_3 * size -
                     superscript_bounding_box.descent());
            const float clearance_subscript =
            std::max(big_op_spacing_2 * size,
                     big_op_spacing_4 * size -
                     subscript_bounding_box.ascent());
            const float shift_up =
            nucleus_bounding_box.ascent() +
            clearance_superscript +
            superscript_bounding_box.descent() +
            nucleus_shift_up;
            const float shift_down =
            nucleus_bounding_box.descent() +
            clearance_subscript +
            subscript_bounding_box.ascent() -
            nucleus_shift_up;
            const float min_shift_right =
            std::min(0.0F, std::min(shift_right_superscript,
                                    shift_right_subscript));

            token_list.push_back(math_token_t(
                                              point_t(-min_shift_right,
                                                      nucleus_shift_up),
                                              nucleus_bounding_box, style));
            token_list.push_back(math_token_t(
                                              point_t(shift_right_superscript -
                                                      min_shift_right,
                                                      shift_up),
                                              superscript_bounding_box,
                                              superscript_style));
            token_list.push_back(math_token_t(
                                              point_t(shift_right_subscript -
                                                      min_shift_right,
                                                      -shift_down),
                                              subscript_bounding_box,
                                              subscript_style));
            return token_list;
         }

         // \nolimits or nucleus only
         token_list.push_back(math_token_t(
                                           point_t(0, nucleus_shift_up),
                                           nucleus_bounding_box, style));
         current_x += nucleus_bounding_box.advance();
      }
      else {   // Neither Rad nor Op
         nucleus_bounding_box =
         math_bounding_box(atom._nucleus, style);
         token_list.push_back(math_token_t(
                                           nucleus_bounding_box, style));
         current_x += nucleus_bounding_box.advance();
      }

      if(atom._superscript.empty() && atom._subscript.empty())
         return token_list;

      const float current_x_italic_corrected = current_x +
      nucleus_bounding_box.italic_correction();
      const float nucleus_size = size;
      const unsigned int superscript_style =
      next_superscript_style(style);
      const unsigned int subscript_style =
      next_subscript_style(style);
      const float superscript_size = style_size(superscript_style);
      const float subscript_size = style_size(subscript_style);
      // Rule 18a
      const float min_shift_up = nucleus_bounding_box.ascent() -
      superscript_size * sup_drop;
      const float min_shift_down = nucleus_bounding_box.descent() +
      subscript_size * sub_drop;

      // Rule 18b
      if(atom._superscript.empty()) {
         const bounding_box_t subscript_bounding_box =
         math_bounding_box(atom._subscript, subscript_style);
         const float shift_down =
         std::max(std::max(min_shift_down,
                           nucleus_size * sub_1),
                  subscript_bounding_box.ascent() -
                  0.8F * x_height(subscript_style));
         token_list.push_back(math_token_t(
                                           point_t(current_x, nucleus_shift_up - shift_down),
                                           subscript_bounding_box, subscript_style));
         return token_list;
      }
      // Rule 18c
      const bounding_box_t superscript_bounding_box =
      math_bounding_box(atom._superscript, superscript_style);
      const float min_shift_up_2 = nucleus_size *
      (style == math_text_t::item_t::STYLE_DISPLAY ?
       sup_1 : is_prime_style(style) ? sup_3 : sup_2);
      float shift_up =
      std::max(std::max(min_shift_up, min_shift_up_2),
               superscript_bounding_box.descent() +
               0.2F * x_height(superscript_style));
      // Rule 18d
      if(atom._subscript.empty()) {
         token_list.push_back(math_token_t(
                                           point_t(current_x_italic_corrected,
                                                   nucleus_shift_up + shift_up),
                                           superscript_bounding_box, superscript_style));
         return token_list;
      }

      // Still rule 18d
      float shift_down =
      std::max(min_shift_down, nucleus_size * sub_2);
      // Rule 18e
      const bounding_box_t subscript_bounding_box =
      math_bounding_box(atom._subscript, subscript_style);

      if((shift_up - superscript_bounding_box.descent()) -
         (subscript_bounding_box.ascent() - shift_down) <
         4.0F * default_rule_thickness *
         nucleus_size) {
         shift_down = 4.0F * default_rule_thickness *
         nucleus_size + subscript_bounding_box.ascent() +
         superscript_bounding_box.descent() - shift_up;

         const float superscript_adjustment =
         0.8F * x_height(superscript_style) -
         (shift_up - superscript_bounding_box.descent());

         if(superscript_adjustment > 0) {
            shift_up += superscript_adjustment;
            shift_down -= superscript_adjustment;
         }
      }
      // Rule 18f
      token_list.push_back(math_token_t(
                                        point_t(current_x_italic_corrected,
                                                nucleus_shift_up + shift_up),
                                        superscript_bounding_box, superscript_style));
      token_list.push_back(math_token_t(
                                        point_t(current_x,
                                                nucleus_shift_up - shift_down),
                                        subscript_bounding_box, subscript_style));

      return token_list;
   }

}
