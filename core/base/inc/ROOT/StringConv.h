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


#include "RStringView.h"
#include "Rtypes.h"
#include "RConfigure.h"
#include <cmath>

namespace ROOT {

   // Adapted from http://stackoverflow.com/questions/3758606/
   // how-to-convert-byte-size-into-human-readable-format-in-java
   // and http://agentzlerich.blogspot.com/2011/01/converting-to-and-from-human-readable.html


////////////////////////////////////////////////////////////////////////////////
/// Return the size expressed in 'human readeable' format.
/// \param bytes the size in bytes to be converted
/// \param si whether to use the SI units or not.
/// \param coeff return the size expressed in the new unit.
/// \param units return a point to the string representation of the new unit
void ToHumanReadableSize(Long64_t bytes,
                         Bool_t si,
                         double *coeff,
                         const char **units)
{
   // Static lookup table of byte-based SI units
   static const char *const suffix[][2] =
     { { "B",  "B"   },
       { "kB", "KiB" },
       { "MB", "MiB" },
       { "GB", "GiB" },
       { "TB", "TiB" },
       { "EB", "EiB" },
       { "ZB", "ZiB" },
       { "YB", "YiB" } };
   int unit = si ? 1000 : 1024;
   int exp = 0;
   if (bytes > 0) {
      exp = std::min( (int) (std::log(bytes) / std::log(unit)),
                      (int) (sizeof(suffix) / sizeof(suffix[0]) - 1));
   }
   *coeff = bytes / std::pow(unit, exp);
   *units  = suffix[exp][!!si];
}

// Convert strings like the following into byte counts
//    5MB, 5 MB, 5M, 3.7GB, 123b, 456kB
// with some amount of forgiveness baked into the parsing.
Long64_t FromHumanReadableSize(std::string_view str)
{
   try {
      size_t size = str.size();
      size_t cur;
      // Parse leading numeric factor
      const double coeff = stod(str, &cur);

      // Skip any intermediate white space
      while (cur<size && isspace(str[cur])) ++cur;

      // Read off first character which should be an SI prefix
      int exp  = 0;
      int unit = 1024;

      auto result = [coeff,&exp,&unit]() {
         return exp ? coeff * std::pow(unit, exp / 3) : coeff;
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

         default:   return -1;
      }
      ++cur;

      // If an 'i' or 'I' is present use SI factor-of-1000 units
      if (cur<size && toupper(str[cur]) == 'I') {
         ++cur;
         unit = 1000;
      }

      if (cur==size) return result();

      // Next character must be one of B/empty/whitespace
      switch (toupper(str[cur])) {
         case 'B':
         case ' ':
         case '\t': ++cur;  break;

         case '\0': return result();

         default:   return -1;
      }

      // Skip any remaining white space
      // while (cur<size && isspace(str[cur])) ++cur;

      // Do not:
      // Parse error on anything but a null terminator
      // if (cur<size) return -1;

      return result();
      //done:
      //   return exp ? coeff * pow(unit, exp / 3) : coeff;
   } catch (...) {
      return -1;
   }

}

} // namespace ROOT.

#endif // ROOT_StringConv