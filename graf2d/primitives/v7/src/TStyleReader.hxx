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
 and finally `$ROOTSYS/etc/system.rootstylerc`.
  */
class TDrawingOptsReader {
public:
   using Attrs_t = std::unordered_map<std::string, std::string>;

private:
   /// Attributes to read into.
   Attrs_t &fAttrs;

public:
   TDrawingOptsReader(Attrs_t &attrs): fAttrs(attrs) {}

   ///  Reads the attribute config values from `.rootstylerc`. If the style entry is not found there, tries
   ///  `~/.rootstylerc` and finally `$ROOTSYS/etc/system.rootstylerc`.
   static Attrs_t ReadDefaults();

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
