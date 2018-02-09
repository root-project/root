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

#ifndef ROOT7_TStringEnumAttr
#define ROOT7_TStringEnumAttr

#include <algorithm>
#include <initializer_list>
#include <string>
#include <vector>

namespace ROOT {
namespace Experimental {

/** \class ROOT::Experimental::TStringAttrSet
 Graphics attribute that consists of a string, selected from a set of options.
 This is the set of options. It's expected to be of static storage duration.
 */
class TStringEnumAttrSet {
   std::vector<std::string> fOptSet; ///< The set of options.

public:
   TStringEnumAttrSet(std::initializer_list<const char*> il)
   {
      fOptSet.insert(fOptSet.end(), il.begin(), il.end());
   }
   const std::vector<std::string>& GetSet() const { return fOptSet; }

   std::size_t Find(const std::string &opt) const {
      auto iter = std::find(fOptSet.begin(), fOptSet.end(), opt);
      if (iter != fOptSet.end())
         return (std::size_t)(iter - fOptSet.begin());
      return (std::size_t) -1;
   }

   const std::string &operator[](std::size_t idx) const { return fOptSet[idx]; }
};

/** \class ROOT::Experimental::TStringEnumAttr
 Graphics attribute that consists of a string, selected from a set of options.
 */
class TStringEnumAttr {
   std::size_t fIdx; ///< Selected option from fStringSet.
   const TStringEnumAttrSet &fStringSet; ///< Reference to the set of options.

public:
   /// Construct the option from the set of strings and the selected option index.
   TStringEnumAttr(std::size_t idx, const TStringEnumAttrSet &strSet): fIdx(idx), fStringSet(strSet) {}

   /// Set the index of the selected option.
   void SetIndex(int idx) { fIdx = idx; }

   /// Get the string representing the selected option.
   const std::string& GetAsString() const { return fStringSet[fIdx]; }

   /// Get the index of the selected option.
   std::size_t GetIndex() const { return fIdx; }
};

/// Initialize an attribute `val` from a string value.
///
///\param[in] name - the attribute name, for diagnostic purposes.
///\param[in] strval - the attribute value as a string.
///\param[out] val - the value to be initialized.
void InitializeAttrFromString(const std::string &name, const std::string &strval, TStringEnumAttr& val);

} // namespace Experimental
} // namespace ROOT

#endif
