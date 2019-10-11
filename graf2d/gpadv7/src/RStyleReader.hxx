/*************************************************************************
 * Copyright (C) 1995-2017, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RStyleReader
#define ROOT7_RStyleReader

#include <ROOT/RStringView.hxx>
#include <ROOT/RStyle.hxx>

#include <string>
#include <unordered_map>
#include <vector>

namespace ROOT {
namespace Experimental {
namespace Internal {
/** \class RStyleReader
\ingroup GpadROOT7
\brief Reads the attribute config values from `.rootstylerc`.
If the style entry is not found there, tries `~/.rootstylerc` and finally `$ROOTSYS/etc/system.rootstylerc`.
\author Axel Naumann <axel@cern.ch>
\date 2017-09-29
\warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
*/

/*
class RStyleReader {
public:
   /// Key is the style name.
   using AllStyles_t = std::unordered_map<std::string, RStyle>;

private:
   /// Collection of attributes to read into.
   AllStyles_t &fAttrs;

public:
   RStyleReader(AllStyles_t &attrs): fAttrs(attrs) {}

   ///  Reads the attribute config values from `.rootstylerc`. If the style entry is not found there, tries
   ///  `~/.rootstylerc` and finally `$ROOTSYS/etc/system.rootstylerc`.
   ///
   ///\param[out] target - collection to read into.
   void ReadDefaults();

   /// Adds attributes specified in `filename` to those already existing in `fAttrs`.
   /// Overwrites values for attributes that already exist in `attrs`!
   /// \returns `true` on success, `false` if the file cannot be found or the syntax is wrong.
   /// Prints an error if the syntax is wrong (but not if the file does not exist).
   bool AddFromStyleFile(const std::string &filename);
};

*/
} // namespace Internal
} // namespace Experimental
} // namespace ROOT

#endif
