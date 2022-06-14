// @(#)root/base
// Author: Philippe Canal 12/2015

/*************************************************************************
 * Copyright (C) 1995-2015, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_StringConv
#define ROOT_StringConv


#include "ROOT/RStringView.hxx"
#include "Rtypes.h"
#include "RConfigure.h"
#include <cmath>

namespace ROOT {

   // Adapted from http://stackoverflow.com/questions/3758606/
   // how-to-convert-byte-size-into-human-readable-format-in-java
   // and http://agentzlerich.blogspot.com/2011/01/converting-to-and-from-human-readable.html
   // However those sources use the 'conventional' 'legacy' nomenclature,
   // rather than the official Standard Units.  See
   // http://physics.nist.gov/cuu/Units/binary.html
   // and http://www.dr-lex.be/info-stuff/bytecalc.html for example.

///////////////////////////////////////////////////////////////////////////////
/// Return the size expressed in 'human readable' format.
/// \param bytes the size in bytes to be converted
/// \param si whether to use the SI units or not.
/// \param coeff return the size expressed in the new unit.
/// \param units return a pointer to the string representation of the new unit
template <typename value_type>
void ToHumanReadableSize(value_type bytes,
                         Bool_t si,
                         Double_t *coeff,
                         const char **units)
{
   // Static lookup table of byte-based SI units
   static const char *const suffix[][2] =
   { { "B",  "B"   },
     { "KB", "KiB" },
     { "MB", "MiB" },
     { "GB", "GiB" },
     { "TB", "TiB" },
     { "EB", "EiB" },
     { "ZB", "ZiB" },
     { "YB", "YiB" } };
   value_type unit = si ? 1000 : 1024;
   int exp = 0;
   if (bytes == unit) {
      // On some 32bit platforms, the result of
      //    (int) (std::log(bytes) / std::log(unit)
      // in the case of bytes==unit ends up surprisingly to be zero
      // rather than one, so 'hard code' the result
      exp = 1;
   } else if (bytes > 0) {
      exp = std::min( (int) (std::log(bytes) / std::log(unit)),
                     (int) (sizeof(suffix) / sizeof(suffix[0]) - 1));
   }
   *coeff = bytes / std::pow(unit, exp);
   *units  = suffix[exp][!si];
}

enum class EFromHumanReadableSize {
   kSuccess,
   kParseFail,
   kOverflow
};

///////////////////////////////////////////////////////////////////////////////
/// Convert strings like the following into byte counts
///    5MB, 5 MB, 5M, 3.7GB, 123b, 456kB, 3.7GiB, 5MiB
/// with some amount of forgiveness baked into the parsing.
/// For this routine we use the official SI unit where the [i] is reserved
/// for the 'legacy' power of two units.  1KB = 1000 bytes, 1KiB = 1024 bytes.
/// \param str the string to be parsed
/// \param value will be updated with the result if and only if the parse is successful and does not overflow for the type of value.
/// \return return a EFromHumanReadableSize enum value indicating the success or failure of the parse.
///
template <typename T>
EFromHumanReadableSize FromHumanReadableSize(std::string_view str, T &value)
{
   try {
      size_t cur, size = str.size();
      // Parse leading numeric factor
      const double coeff = stod(std::string(str.data(), str.size()), &cur);

      // Skip any intermediate white space
      while (cur<size && isspace(str[cur])) ++cur;

      // Read off first character which should be an SI prefix
      int exp = 0, unit = 1000;

      auto result = [coeff,&exp,&unit,&value]() {
         double v = exp ? coeff * std::pow(unit, exp / 3) : coeff;
         if (v < (double) std::numeric_limits<T>::max()) {
            value = (T)v;
            return EFromHumanReadableSize::kSuccess;
         } else {
            return EFromHumanReadableSize::kOverflow;
         }
      };
      if (cur==size) return result();

      switch (toupper(str[cur])) {
         case 'B':  exp =  0; break;
         case 'K':  exp =  3; break;
         case 'M':  exp =  6; break;
         case 'G':  exp =  9; break;
         case 'T':  exp = 12; break;
         case 'E':  exp = 15; break;
         case 'Z':  exp = 18; break;
         case 'Y':  exp = 21; break;

         default:   return EFromHumanReadableSize::kParseFail;
      }
      ++cur;

      // If an 'i' or 'I' is present use non-SI factor-of-1024 units
      if (cur<size && toupper(str[cur]) == 'I') {
         ++cur;
         unit = 1024;
      }

      if (cur==size) return result();

      // Next character must be one of B/empty/whitespace
      switch (toupper(str[cur])) {
         case 'B':
         case ' ':
         case '\t': ++cur;  break;

         case '\0': return result();

         default:   return EFromHumanReadableSize::kParseFail;
      }

      // Skip any remaining white space
      // while (cur<size && isspace(str[cur])) ++cur;

      // Do not:
      // Parse error on anything but a null terminator
      // if (cur<size) return -1;

      return result();
   } catch (...) {
      return EFromHumanReadableSize::kParseFail;
   }

}

template <typename T>
EFromHumanReadableSize FromHumanReadableSize(ROOT::Internal::TStringView str, T &value)
{
   return FromHumanReadableSize(std::string_view(str),value);
}

} // namespace ROOT.

#endif // ROOT_StringConv
