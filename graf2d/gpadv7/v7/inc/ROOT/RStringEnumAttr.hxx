/// \file ROOT/RStringEnumAttr.hxx
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

#ifndef ROOT7_RStringEnumAttr
#define ROOT7_RStringEnumAttr

#include <algorithm>
#include <initializer_list>
#include <string>
#include <vector>

namespace ROOT {
namespace Experimental {

/** \class ROOT::Experimental::RStringAttrSet
 Graphics attribute that consists of a string, selected from a set of options.
 This is the set of options. It's expected to be of static storage duration.
 */
class RStringEnumAttrSet {
   std::vector<std::string> fOptSet; ///< The set of options.

public:
   RStringEnumAttrSet(std::initializer_list<const char*> il)
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

/** \class ROOT::Experimental::RStringEnumAttrBase
 Base class for template RStringEnumAttr, erasing the underlying integral type.
 */
class RStringEnumAttrBase {
protected:
   /// Selected option from fStringSet.
   std::size_t fIdx;

   /// Reference to the set of options.
   const RStringEnumAttrSet *fStringSet; //!

public:
   RStringEnumAttrBase(std::size_t idx, const RStringEnumAttrSet &strSet): fIdx(idx), fStringSet(&strSet) {}
};

/** \class ROOT::Experimental::RStringEnumAttr
 Graphics attribute that consists of a string, selected from a set of options.
 \tparam ENUM - underlying enum type (or `int` etc).
 */
template <class ENUM>
class RStringEnumAttr: public RStringEnumAttrBase {
public:
   /// Construct the option from the set of strings and the selected option index.
   RStringEnumAttr(ENUM idx, const RStringEnumAttrSet &strSet): RStringEnumAttrBase((std::size_t)idx, strSet) {}

   /// Set the index of the selected option.
   void SetIndex(ENUM idx) { fIdx = (std::size_t)idx; }

   /// Get the string representing the selected option.
   const std::string& GetAsString() const { return (*fStringSet)[(std::size_t)fIdx]; }

   /// Get the index of the selected option.
   ENUM GetIndex() const { return static_cast<ENUM>(fIdx); }
};

/// Initialize an attribute `val` from a string value.
///
///\param[in] name - the attribute name, for diagnostic purposes.
///\param[in] strval - the attribute value as a string.
///\param[out] val - the value to be initialized.
void InitializeAttrFromString(const std::string &name, const std::string &strval, RStringEnumAttrBase& val);

} // namespace Experimental
} // namespace ROOT

#endif
