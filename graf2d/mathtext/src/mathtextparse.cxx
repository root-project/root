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

#include <cmath>
#include <iostream>
#include <algorithm>
#include <string>
#include "../inc/mathtext.h"

/////////////////////////////////////////////////////////////////////

namespace mathtext {

   /////////////////////////////////////////////////////////////////

   void math_text_t::field_t::
   parse_math_list(const std::vector<std::string> &str_split,
                   const unsigned int default_family)
   {
#if 0
      std::cerr << "parsing [";
      for(std::vector<std::string>::const_iterator iterator =
          str_split.begin();
          iterator != str_split.end(); iterator++)
         std::cerr << '"' << *iterator << "\", ";
      std::cerr << ']' << std::endl;
#endif

      // State of radical parsing
      enum {
         RADICAL_STATE_NONE = 0,
         RADICAL_STATE_RADICAND,
         RADICAL_STATE_OF,
         RADICAL_STATE_INDEX
      };

      unsigned int family = default_family;
      int level = 0;
      int delimiter_level = 0;
      std::vector<std::string> buffer;
#if 1
      bool superscript = false;
      bool subscript = false;
      bool delimiter_right = false;
      unsigned int radical_state = RADICAL_STATE_NONE;
      std::vector<std::string> radical_index;
      bool horizontal_box = false;

      for(std::vector<std::string>::const_iterator iterator =
          str_split.begin();
          iterator != str_split.end(); iterator++) {
         // ONLY LEVEL 0 superscript and subscript are interpreted,
         // and they are ignored afterwards.
         if(level == 0 && delimiter_level == 0) {
            if((*iterator)[0] == '^') {
               superscript = true;
               continue;
            }
            else if((*iterator)[0] == '_') {
               subscript = true;
               continue;
            }
            else if(*iterator == "\\sqrt") {
               radical_state = RADICAL_STATE_RADICAND;
               radical_index = std::vector<std::string>();
               continue;
            }
            else if(iterator->substr(0, 5) == "\\root") {
               radical_state = RADICAL_STATE_INDEX;
               continue;
            }
            else if(radical_state == RADICAL_STATE_INDEX &&
                    *iterator == "\\of") {
               radical_index = buffer;
               buffer.clear();
               radical_state = RADICAL_STATE_RADICAND;
               continue;
            }
            else if(iterator->substr(0, 5) == "\\hbox" ||
                    iterator->substr(0, 5) == "\\text") {
               horizontal_box = true;
               continue;
            }

            const char **lower;

#include "table/mathfontch.h"
            lower = std::lower_bound(
                                     font_change_control_sequence,
                                     font_change_control_sequence + nfont_change,
                                     *iterator);

            if(lower <
               font_change_control_sequence + nfont_change &&
               *lower == *iterator) {
               const unsigned long index =
               lower - font_change_control_sequence;

               family = font_change_family[index];
               continue;
            }
#include "table/mathopstd.h"
            lower = std::lower_bound(
                                     operator_control_sequence,
                                     operator_control_sequence + noperator,
                                     *iterator);
            if(lower <
               operator_control_sequence + noperator &&
               *lower == *iterator) {
               const unsigned long index =
               lower - operator_control_sequence;

               if(operator_code_point[index] == L'\0') {
                  // Operator defined with \mathop
                  const field_t operator_math_list(field_t(
                                                           tex_split(operator_content[index]),
                                                           math_symbol_t::FAMILY_REGULAR));
                  atom_t atom(atom_t::TYPE_OP, field_t(
                                                       operator_math_list));

                  atom._limits = operator_nolimits[index] ?
                  atom_t::LIMITS_NOLIMITS :
                  atom_t::LIMITS_DISPLAYLIMITS;
                  append(atom);
               }
               else {
                  // Operator defined with \mathchardef
                  const field_t operator_math_symbol(
                                                     math_symbol_t(
                                                                   *iterator,
                                                                   operator_code_point[index],
                                                                   math_symbol_t::FAMILY_REGULAR));
                  atom_t atom(atom_t::TYPE_OP, field_t(
                                                       operator_math_symbol));

                  atom._limits = operator_nolimits[index] ?
                  atom_t::LIMITS_NOLIMITS :
                  atom_t::LIMITS_DISPLAYLIMITS;
                  append(atom);
               }
               continue;
            }
         }

         if((*iterator)[0] == '}') {
            level--;
            // When the level decreases to 0 here, the compound
            // expression is complete, create a subfield and
            // append it to the math list.
            if(level == 0 && delimiter_level == 0) {
               // Create subfields recursively
               const field_t subfield(buffer, family);

               if(radical_state == RADICAL_STATE_RADICAND) {
                  atom_t atom(atom_t::TYPE_RAD, subfield);

                  atom._index = field_t(radical_index, family);
                  append(item_t(atom));
                  radical_state = RADICAL_STATE_NONE;
               }
               else
                  append(subfield, superscript, subscript);
               buffer.clear();
            }
         }
         else if(*iterator == "\\right") {
            delimiter_level--;
            // When the delimtier level decreases to 0 here, the
            // compound expression is complete, create a subfield
            // and append it with the appropriate delimiter atoms
            // to the math list.
            if(level == 0 && delimiter_level == 0) {
               delimiter_right = true;
               continue;
            }
         }

         /////////////////////////////////////////////////////////

#if 0
         std::cerr << __FILE__ << ':' << __LINE__
         << ": L" << level << ", DL"
         << delimiter_level << ", hbox = "
         << horizontal_box << ", *iterator = "
         << *iterator << std::endl;
#endif

         if(level > 0 || delimiter_level > 0 ||
            radical_state == RADICAL_STATE_INDEX) {
            buffer.push_back(*iterator);
         }
         else if(delimiter_right) {
            const std::string left = buffer.front();

            buffer.erase(buffer.begin());

            const field_t subfield(left, buffer, *iterator,
                                   family);

            if(radical_state == RADICAL_STATE_RADICAND) {
               atom_t atom(atom_t::TYPE_RAD, subfield);

               atom._index = field_t(radical_index, family);
               append(item_t(atom));
               radical_state = RADICAL_STATE_NONE;
            }
            else
               append(subfield, superscript, subscript);
            buffer.clear();
         }
         else if(horizontal_box) {
            box_t box(math_text_t::utf8_cast(*iterator));

            append(field_t(box), superscript, subscript);
         }
         else if((*iterator)[0] != '{' &&
                 (*iterator)[0] != '}' &&
                 *iterator != "\\left" &&
                 *iterator != "\\right") {
            if(radical_state == RADICAL_STATE_RADICAND) {
               const field_t subfield(
                                      std::vector<std::string>(1, *iterator),
                                      family);
               atom_t atom(atom_t::TYPE_RAD, subfield);

               atom._index = field_t(radical_index, family);
               append(item_t(atom));
            }
            // FIXME: This should be a true table
            else if(*iterator == "\\over") {
               append(item_t(item_t::TYPE_GENERALIZED_FRACTION,
                             1.0F));
            }
            else if(*iterator == "\\atop") {
               append(item_t(item_t::TYPE_GENERALIZED_FRACTION,
                             0.0F));
            }
            // FIXME: This should be a true table
            else if(*iterator == "\\!") {
               append(item_t(item_t::TYPE_KERN, -3.0F));
            }
            else if(*iterator == "\\,") {
               append(item_t(item_t::TYPE_KERN, 3.0F));
            }
            else if(*iterator == "\\:") {
               append(item_t(item_t::TYPE_KERN, 4.0F));
            }
            else if(*iterator == "\\;") {
               append(item_t(item_t::TYPE_KERN, 5.0F));
            }
            else if(*iterator == "\\quad") {
               append(item_t(item_t::TYPE_KERN, 18.0F));
            }
            else if(*iterator == "\\qquad") {
               append(item_t(item_t::TYPE_KERN, 36.0F));
            }
            else {
               const math_symbol_t
               math_symbol(*iterator, family);

               append(item_t::TYPE_ATOM, math_symbol,
                      superscript, subscript);
            }
         }

         if((*iterator)[0] == '{') {
            level++;
         }
         else if(*iterator == "\\left") {
            // Since the actual delimiter follows, it is going to
            // be appended to the buffer "automatically".
            delimiter_level++;
         }
         // Reset superscript and subscript flags only for level 0
         if(level == 0 && delimiter_level == 0 &&
            (radical_state == RADICAL_STATE_RADICAND ||
             radical_state == RADICAL_STATE_NONE)) {
               superscript = false;
               subscript = false;
               delimiter_right = false;
               radical_state = RADICAL_STATE_NONE;
               family = default_family;
               horizontal_box = false;
            }
      }
#else
      for(std::vector<std::string>::const_iterator iterator =
          str_split.begin();
          iterator != str_split.end(); iterator++) {
         if((*iterator)[0] == '}') {
            level--;

         }
         else if(*iterator == "\\right") {
            delimiter_level--;
         }
         if((*iterator)[0] == '{') {
            level++;
         }
         else if(*iterator == "\\left") {
            // Since the actual delimiter follows, it is going to
            // be appended to the buffer "automatically".
            delimiter_level++;
         }
         else if(level == 0 && delimiter_level == 0) {
#if 1
            std::cerr << __FILE__ << ':' << __LINE__
            << ": L" << level << ", DL"
            << delimiter_level << ", *iterator = "
            << *iterator << ", buffer = { ";
            for(std::vector<std::string>::const_iterator
                buffer_iterator = buffer.begin();
                buffer_iterator != buffer.end(); buffer_iterator++) {
               std::cerr << '"' << *buffer_iterator << "\" ";
            }
            std::cerr << '}' << std::endl;
#endif
            buffer.clear();
         }
         else {
            buffer.push_back(*iterator);
         }
      }
#endif
   }

   std::vector<std::string> math_text_t::
   tex_split(const std::string &raw_code, const char escape_character)
   {
      std::string code = raw_code;

      for(std::string::iterator iterator = code.begin();
          iterator != code.end(); iterator++) {
         if(*iterator == escape_character) {
            *iterator = '\\';
         }
      }

      std::vector<std::string> ret;

      if(code.size() <= 0) {
         return ret;
      }

      size_t begin = 0;
      size_t end = 1;
      bool box = false;

      while (code[begin] == ' ') {
         begin++;
      }
      while (begin < code.size()) {
         end = begin + 1;
         if(code[begin] == '\\') {
            if(isalpha(code[end])) {
               while(end < code.size() && isalpha(code[end])) {
                  end++;
               }
            }
            else if(end < code.size()) {
               end++;
            }

#include "table/mathbracketcs.h"
            const char **lower =
            std::lower_bound(bracket_control_sequence,
                             bracket_control_sequence +
                             nbracket_control_sequence,
                             code.substr(begin,
                                         end - begin));

            if(lower < bracket_control_sequence +
               nbracket_control_sequence &&
               *lower == code.substr(begin, end - begin) &&
               end + 1 < code.size() && code[end] == '[') {
               while(end < code.size() && code[end] != ']') {
                  end++;
               }
               if(end < code.size()) {
                  end++;
               }
            }
         }

         std::string code_substr =
         code.substr(begin, end - begin);

#if 1
         if(code_substr == "\\hbox" || code_substr == "\\text") {
            box = true;
         }
         else if(box) {
            if(code[begin] == '{') {
               for(unsigned int level = 1;
                   end < code.size() && level > 0; end++) {
                  if(code[end - 1] != '\\') {
                     switch(code[end]) {
                        case '{':   level++; break;
                        case '}':   level--; break;
                     }
                  }
               }
               code_substr =
               code.substr(begin + 1, end - begin - 2);
            }
            else if(code[begin] == '\\' &&
                    begin + 1 < code.size()) {
               code_substr = code.substr(begin, 2);
            }
            else {
               code_substr = code.substr(begin, 1);
            }
            // FIXME: Proper interpretation of escaped characters
            box = false;
         }
#endif

         ret.push_back(code_substr);
         begin = end;
         while( begin<code.size() && code[begin] == ' ' ) {
            begin++;
         }
      }

      return ret;
   }

   std::vector<std::string> math_text_t::
   tex_replace(const std::vector<std::string> &/*code*/)
   {
#if 0
      static const size_t ncontrol_max = 256;
      static const char *table[][ncontrol_max] = {
         { "\\%", "\0", "\\root", "\1", "\\of", "\2", NULL },
         { "\\sqrt", "[]", "\1", "\\root", "\1", "\\of", "\2", NULL },
         { "\\frac", "\2", "{", "\1", "\\over", "\2", "}", NULL }
      };
#endif

      return std::vector<std::string>();
   }

}
