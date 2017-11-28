/// \file ROOT/TDrawingOptsReader.hxx
/// \ingroup Gpad ROOT7
/// \author Axel Naumann <axel@cern.ch>
/// \date 2017-09-29
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2017, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_TDrawingOptsReader
#define ROOT7_TDrawingOptsReader

#include <ROOT/TColor.hxx>

#include <RStringView.h>

#include <unordered_map>
#include <vector>

namespace ROOT {
namespace Experimental {
namespace Internal {
/** \class ROOT::Experimental::TFrame
 Reads the attribute config values from `.rootstylerc`. If the style entry is not found there, tries `~/.rootstylerc`
 and finally `$ROOTSYS/system.rootstylerc`.
  */
class TDrawingOptsReader {
public:
   using Attrs_t = std::unordered_map<std::string, std::string>;

private:
   /// Attributes to operate on.
   Attrs_t &fAttrs;

public:
   TDrawingOptsReader(Attrs_t &attrs): fAttrs(attrs) {}

   ///  Reads the attribute config values from `.rootstylerc`. If the style entry is not found there, tries
   ///  `~/.rootstylerc` and finally `$ROOTSYS/system.rootstylerc`.

   static Attrs_t ReadDefaults();

   /// Parse a TColor from attr's value.
   /// Colors can be specified as RGBA (red green blue alpha) or RRGGBBAA:
   ///     #fa7f #ffa07bff
   /// For all predefined colors in TColor, colors can be specified as name without leading 'k', e.g. `red` for
   /// `TColor::kRed`.
   /// Prints an error and returns `TColor::kBlack` if the attribute string cannot be parsed or if the attribute has no
   /// entry in `fAttrs`.
   TColor ParseColor(std::string_view attr, const TColor& deflt);

   /// Parse an integer attribute, or if `opts` is given, return the index of the string from the options file in
   /// `opts`. Returns `0` (and prints an error) if the string cannot be found in opts, or if the integer cannot be
   /// parsed or if the attribute has no entry in `fAttrs`.
   long long ParseInt(std::string_view attr, long long deflt, std::vector<std::string_view> opts = {});

   /// Parse a floating point attribute.
   /// Returns `0.` and prints an error if the attribute string cannot be parsed as a floating point number.
   /// Prints an error if the attribute has no entry in `fAttrs`.
   double ParseFP(std::string_view attr, double deflt);

   /// Convenience overloads:
   TColor Parse(std::string_view attr, const TColor &deflt, std::vector<std::string_view> = {}) { return ParseColor(attr, deflt); }
   long long Parse(std::string_view attr, long long deflt, std::vector<std::string_view> opts = {})
   {
      return ParseInt(attr, deflt, opts);
   }
   double Parse(std::string_view attr, double deflt, std::vector<std::string_view> = {}) { return ParseFP(attr, deflt); }

   /// Adds attributes specified in `filename` to those already existing in `fAttrs`.
   /// Overwrites values for attributes that already exist in `attrs`!
   /// \returns `true` on success, `false` if the file cannot be found or the syntax is wrong.
   /// Prints an error if the syntax is wrong (but not if the file does not exist).
   bool AddFromStyleFile(std::string_view filename);
};
} // namespace Internal
} // namespace Experimental
} // namespace ROOT

#endif
