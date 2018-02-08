/// \file ROOT/TStringAttr.hxx
/// \ingroup Gpad ROOT7
/// \author Axel Naumann <axel@cern.ch>
/// \date 2018-02-08
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_TStringAttr
#define ROOT7_TStringAttr

#include <string>
#include <vector>

namespace ROOT {
namespace Experimental {

/** \class ROOT::Experimental::TStringAttrSet
 Graphics attribute that consists of a string, selected from a set of options.
 This is the set of options. It's expected to be of static storage duration.
 */
class TStringAttrSet {
   std::vector<std::string> fOptSet; ///< The set of options.

public:
   TStringAttrSet(std::vector<std::string> &&optionsSet): fOptSet(std::move(optionsSet)) {}
   const std::vector<std::string>& GetSet() const { return fOptSet; }

   std::size_t Find(const std::string &opt) const {
      auto iter = std::find(fOptSet.begin(), fOptSet.end(), opt);
      if (iter != fOptSet.end())
         return (std::size_t)(iter - fOptSet.begin());
      return (std::size_t) -1;
   }

   const std::string &operator[](std::size_t idx) const { return fOptSet[idx]; }
};

/** \class ROOT::Experimental::TStringAttr
 Graphics attribute that consists of a string, selected from a set of options.
 */
class TStringAttr {
   std::size_t fIdx; ///< Selected option from fStringSet.
   const TStringAttrSet &fStringSet; ///< Reference to the set of options.

public:
   TStringAttr(std::size_t idx, const TStringAttrSet &strSet): fIdx(idx), fStringSet(strSet) {}
   const std::string& GetAsString() const { return fStringSet[fIdx]; }

};

} // namespace Experimental
} // namespace ROOT

#endif
