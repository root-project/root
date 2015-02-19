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

#ifdef WIN32
// On Windows, Disable the warning:
// "characters beyond first in wide-character constant ignored"
#pragma warning( push )
#pragma warning( disable : 4066)
#endif

#include <iostream>
#include <algorithm>
#include <string>
#include "../inc/mathtext.h"

/////////////////////////////////////////////////////////////////////

namespace mathtext {

   /////////////////////////////////////////////////////////////////

   // For the symbol classification, see Knuth, The TeXbook (1986),
   // pp. 434ff.

   void math_text_t::math_symbol_t::math_italic_is_upright(void)
   {
      if(_family == FAMILY_MATH_ITALIC) {
         _family = FAMILY_REGULAR;
      }
      else if(_family == FAMILY_MATH_BOLD_ITALIC) {
         _family = FAMILY_BOLD;
      }
   }

   void math_text_t::math_symbol_t::math_italic_is_italic(void)
   {
      if(_family == FAMILY_MATH_ITALIC) {
         _family = FAMILY_ITALIC;
      }
      else if(_family == FAMILY_MATH_BOLD_ITALIC) {
         _family = FAMILY_BOLD_ITALIC;
      }
   }

   void math_text_t::math_symbol_t::encode_character(void)
   {
      if(_code.size() != 1)
         return;

      // Character encoding for plain characters (not a TeX control
      // sequence)
      switch(_code[0]) {
         case '|':
         case '/':
         case '.':
            math_italic_is_upright();
            _glyph = _code[0];
            _type = atom_t::TYPE_ORD;
            break;
         case '+':
            math_italic_is_upright();
            _glyph = _code[0];
            _type = atom_t::TYPE_BIN;
            break;
         case '-':
            math_italic_is_upright();
            _glyph = L'\u2212';
            _type = atom_t::TYPE_BIN;
            break;
         case '*':
            math_italic_is_upright();
            _glyph = L'\u2217';
            _type = atom_t::TYPE_BIN;
            break;
         case '<':
         case '=':
         case '>':
         case ':':
            math_italic_is_upright();
            _glyph = _code[0];
            _type = atom_t::TYPE_REL;
            break;
         case '(':
         case '[':
            math_italic_is_upright();
            _glyph = _code[0];
            _type = atom_t::TYPE_OPEN;
            break;
         case '!':
         case '?':
         case ')':
         case ']':
            math_italic_is_upright();
            _glyph = _code[0];
            _type = atom_t::TYPE_CLOSE;
            break;
         case ',':
         case ';':
            math_italic_is_upright();
            _glyph = _code[0];
            _type = atom_t::TYPE_PUNCT;
            break;
         default:
            if((_code[0] >= 'A' && _code[0] <= 'Z') ||
               (_code[0] >= 'a' && _code[0] <= 'z')) {
               _glyph = _code[0];
               _type = atom_t::TYPE_ORD;
            }
            else if(_code[0] >= '0' && _code[0] <= '9') {
               math_italic_is_upright();
               _glyph = _code[0];
               _type = atom_t::TYPE_ORD;
            }
      }
   }

   void math_text_t::math_symbol_t::encode_control_sequence(void)
   {
      // Character encoding for TeX control sequences
#include "table/mathglyphstd.h"
      const char **lower =
      std::lower_bound(glyph_control_sequence,
                       glyph_control_sequence + nglyph,
                       _code);

      if(lower < glyph_control_sequence + nglyph &&
#ifdef WIN32
         strncmp(*lower, _code.c_str(), _code.size()) == 0
#else // WIN32
         *lower == _code
#endif // WIN32
         ) {
         const unsigned long index =
         lower - glyph_control_sequence;

         if(glyph_upright[index])
            math_italic_is_upright();
         _glyph = glyph_code_point[index];
         _type = glyph_type[index];
      }
   }

   void math_text_t::math_symbol_t::encode_math_blackboard_bold(void)
   {
      if(_code.size() != 1) {
         return;
      }

      if(_code[0] >= 'A' && _code[0] <= 'Z') {
         _family = FAMILY_STIX_REGULAR;
         switch(_code[0]) {
            case 'C':   _glyph = L'\u2102'; break;
            case 'H':   _glyph = L'\u210d'; break;
            case 'N':   _glyph = L'\u2115'; break;
            case 'P':   _glyph = L'\u2119'; break;
            case 'Q':   _glyph = L'\u211a'; break;
            case 'R':   _glyph = L'\u211d'; break;
            case 'Z':   _glyph = L'\u2124'; break;
            default:   _glyph = L'\U0001d538' + (_code[0] - 'A');
         }
         _type = atom_t::TYPE_ORD;
      }
      else if(_code[0] >= 'a' && _code[0] <= 'z') {
         _family = FAMILY_STIX_REGULAR;
         _glyph = L'\U0001d552' + (_code[0] - 'a');
         _type = atom_t::TYPE_ORD;
      }
      else if(_code[0] >= '0' && _code[0] <= '9') {
         _family = FAMILY_STIX_REGULAR;
         _glyph = L'\U0001d7d8' + (_code[0] - '0');
         _type = atom_t::TYPE_ORD;
      }
   }

   void math_text_t::math_symbol_t::encode_math_script_italic(void)
   {
      if(_code.size() != 1) {
         return;
      }

      if(_code[0] >= 'A' && _code[0] <= 'Z') {
         _family = FAMILY_STIX_ITALIC;
         switch(_code[0]) {
            case 'H':   _glyph = L'\u210b'; break;
            case 'I':   _glyph = L'\u2110'; break;
            case 'L':   _glyph = L'\u2112'; break;
            case 'P':   _glyph = L'\u2118'; break;
            case 'R':   _glyph = L'\u211b'; break;
            case 'B':   _glyph = L'\u212c'; break;
            case 'E':   _glyph = L'\u2130'; break;
            case 'F':   _glyph = L'\u2131'; break;
            case 'M':   _glyph = L'\u2133'; break;
            default:   _glyph = L'\U0001d49c' + (_code[0] - 'A');
         }
         _type = atom_t::TYPE_ORD;
      }
      else if(_code[0] >= 'a' && _code[0] <= 'z') {
         _family = FAMILY_STIX_ITALIC;
         switch(_code[0]) {
            case 'g':   _glyph = L'\u210a'; break;
            case 'l':   _glyph = L'\u2113'; break;
            case 'e':   _glyph = L'\u212f'; break;
            case 'o':   _glyph = L'\u2134'; break;
            default:   _glyph = L'\U0001d4b6' + (_code[0] - 'a');
         }
         _type = atom_t::TYPE_ORD;
      }
   }

   void math_text_t::math_symbol_t::encode_math_script_bold_italic(void)
   {
      if(_code.size() != 1)
         return;

      if(_code[0] >= 'A' && _code[0] <= 'Z') {
         _family = FAMILY_STIX_BOLD_ITALIC;
         _glyph = L'\U0001d49c' + (_code[0] - 'A');
         _type = atom_t::TYPE_ORD;
      }
      else if(_code[0] >= 'a' && _code[0] <= 'z') {
         _family = FAMILY_STIX_BOLD_ITALIC;
         _glyph = L'\U0001d4b6' + (_code[0] - 'a');
         _type = atom_t::TYPE_ORD;
      }
   }

   void math_text_t::math_symbol_t::encode_math_fraktur_regular(void)
   {
      if(_code.size() != 1)
         return;

      if(_code[0] >= 'A' && _code[0] <= 'Z') {
         _family = FAMILY_STIX_REGULAR;
         switch(_code[0]) {
            case 'H':   _glyph = L'\u210c'; break;
            case 'I':   _glyph = L'\u2111'; break;
            case 'R':   _glyph = L'\u211c'; break;
            case 'Z':   _glyph = L'\u2128'; break;
            case 'C':   _glyph = L'\u212d'; break;
            default:   _glyph = L'\U0001d504' + (_code[0] - 'A');
         }
         _type = atom_t::TYPE_ORD;
      }
      else if(_code[0] >= 'a' && _code[0] <= 'z') {
         _family = FAMILY_STIX_REGULAR;
         _glyph = L'\U0001d51e' + (_code[0] - 'a');
         _type = atom_t::TYPE_ORD;
      }
   }

   void math_text_t::math_symbol_t::encode_math_fraktur_bold(void)
   {
      if(_code.size() != 1)
         return;

      if(_code[0] >= 'A' && _code[0] <= 'Z') {
         _family = FAMILY_STIX_BOLD;
         _glyph = L'\U0001d56c' + (_code[0] - 'A');
         _type = atom_t::TYPE_ORD;
      }
      else if(_code[0] >= 'a' && _code[0] <= 'z') {
         _family = FAMILY_STIX_BOLD;
         _glyph = L'\U0001d586' + (_code[0] - 'a');
         _type = atom_t::TYPE_ORD;
      }
   }

   void math_text_t::math_symbol_t::
   encode_math_sans_serif_regular(void)
   {
      if(_code.size() != 1)
         return;

      if(_code[0] >= 'A' && _code[0] <= 'Z') {
         _family = FAMILY_STIX_REGULAR;
         _glyph = L'\U0001d5a0' + (_code[0] - 'A');
         _type = atom_t::TYPE_ORD;
      }
      else if(_code[0] >= 'a' && _code[0] <= 'z') {
         _family = FAMILY_STIX_REGULAR;
         _glyph = L'\U0001d5ba' + (_code[0] - 'a');
         _type = atom_t::TYPE_ORD;
      }
      else if(_code[0] >= '0' && _code[0] <= '9') {
         _family = FAMILY_STIX_REGULAR;
         _glyph = L'\U0001d7e2' + (_code[0] - '0');
         _type = atom_t::TYPE_ORD;
      }
   }

   void math_text_t::math_symbol_t::
   encode_math_sans_serif_italic(void)
   {
      if(_code.size() != 1)
         return;

      if(_code[0] >= 'A' && _code[0] <= 'Z') {
         _family = FAMILY_STIX_ITALIC;
         _glyph = L'\U0001d608' + (_code[0] - 'A');
         _type = atom_t::TYPE_ORD;
      }
      else if(_code[0] >= 'a' && _code[0] <= 'z') {
         _family = FAMILY_STIX_ITALIC;
         _glyph = L'\U0001d622' + (_code[0] - 'a');
         _type = atom_t::TYPE_ORD;
      }
   }

   void math_text_t::math_symbol_t::
   encode_math_sans_serif_bold(void)
   {
      if(_code.size() != 1)
         return;

      if(_code[0] >= 'A' && _code[0] <= 'Z') {
         _family = FAMILY_STIX_BOLD;
         _glyph = L'\U0001d5d4' + (_code[0] - 'A');
         _type = atom_t::TYPE_ORD;
      }
      else if(_code[0] >= 'a' && _code[0] <= 'z') {
         _family = FAMILY_STIX_BOLD;
         _glyph = L'\U0001d5ee' + (_code[0] - 'a');
         _type = atom_t::TYPE_ORD;
      }
      else if(_code[0] >= '0' && _code[0] <= '9') {
         _family = FAMILY_STIX_BOLD;
         _glyph = L'\U0001d7ec' + (_code[0] - '0');
         _type = atom_t::TYPE_ORD;
      }
   }

   void math_text_t::math_symbol_t::
   encode_math_sans_serif_bold_italic(void)
   {
      if(_code.size() != 1)
         return;

      if(_code[0] >= 'A' && _code[0] <= 'Z') {
         _family = FAMILY_STIX_BOLD_ITALIC;
         _glyph = L'\U0001d63c' + (_code[0] - 'A');
         _type = atom_t::TYPE_ORD;
      }
      else if(_code[0] >= 'a' && _code[0] <= 'z') {
         _family = FAMILY_STIX_BOLD_ITALIC;
         _glyph = L'\U0001d656' + (_code[0] - 'a');
         _type = atom_t::TYPE_ORD;
      }
   }

   void math_text_t::math_symbol_t::encode_math_alpha(void)
   {
      switch(_family) {
         case FAMILY_MATH_BLACKBOARD_BOLD:
            encode_math_blackboard_bold();
            break;
         case FAMILY_MATH_SCRIPT_ITALIC:
            encode_math_script_italic();
            break;
         case FAMILY_MATH_SCRIPT_BOLD_ITALIC:
            encode_math_script_bold_italic();
            break;
         case FAMILY_MATH_FRAKTUR_REGULAR:
            encode_math_fraktur_regular();
            break;
         case FAMILY_MATH_FRAKTUR_BOLD:
            encode_math_fraktur_bold();
            break;
         case FAMILY_MATH_SANS_SERIF_REGULAR:
            encode_math_sans_serif_regular();
            break;
         case FAMILY_MATH_SANS_SERIF_ITALIC:
            encode_math_sans_serif_italic();
            break;
         case FAMILY_MATH_SANS_SERIF_BOLD:
            encode_math_sans_serif_bold();
            break;
         case FAMILY_MATH_SANS_SERIF_BOLD_ITALIC:
            encode_math_sans_serif_bold_italic();
            break;
      }
   }

   void math_text_t::math_symbol_t::encode(void)
   {
      encode_character();
      encode_control_sequence();
      encode_math_alpha();
      math_italic_is_italic();
      if(_family > FAMILY_STIX_SIZE_5_REGULAR) {
         std::cerr << __FILE__ << ':' << __LINE__
         << ": error: encoding results in a "
         "nonphysical font family" << std::endl;
      }
   }

   bool math_text_t::math_symbol_t::bold(void) const
   {
      switch(_family) {
         case FAMILY_BOLD:
         case FAMILY_BOLD_ITALIC:
         case FAMILY_STIX_BOLD:
         case FAMILY_STIX_BOLD_ITALIC:
         case FAMILY_STIX_SIZE_1_BOLD:
         case FAMILY_STIX_SIZE_2_BOLD:
         case FAMILY_STIX_SIZE_3_BOLD:
         case FAMILY_STIX_SIZE_4_BOLD:
         case FAMILY_MATH_BOLD_ITALIC:
         case FAMILY_MATH_SCRIPT_BOLD_ITALIC:
         case FAMILY_MATH_FRAKTUR_BOLD:
         case FAMILY_MATH_BLACKBOARD_BOLD:
         case FAMILY_MATH_SANS_SERIF_BOLD:
         case FAMILY_MATH_SANS_SERIF_BOLD_ITALIC:
            return true;
         default:
            return false;
      }
   }

}
#ifdef WIN32
#pragma warning( pop )
#endif
