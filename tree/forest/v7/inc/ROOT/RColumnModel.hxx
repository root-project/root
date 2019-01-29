/// \file ROOT/RColumnModel.hxx
/// \ingroup Forest ROOT7
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2018-10-09
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2015, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RColumnModel
#define ROOT7_RColumnModel

#include <ROOT/RStringView.hxx>

#include <string>

namespace ROOT {
namespace Experimental {

// clang-format off
/**
\class ROOT::Experimental::EColumnType
\ingroup Forest
\brief The available trivial, native content types of a column

More complex types, such as classes, get translated into columns of such simple types by the RTreeField.
*/
// clang-format on
enum class EColumnType {
   kUnknown,
   kIndex, // type for root columns of nested collections
   kByte,
   kReal64,
   kReal32,
   kReal16,
   kReal8,
   kInt64,
   kInt32,
   kInt16,
   //...
};

// clang-format off
/**
\class ROOT::Experimental::RColumnModel
\ingroup Forest
\brief Holds the static meta-data of a column in a tree
*/
// clang-format on
class RColumnModel {
private:
   std::string fName;
   EColumnType fType;
   bool fIsSorted;

public:
   RColumnModel(std::string_view name, EColumnType type, bool isSorted)
      : fName(name), fType(type), fIsSorted(isSorted) {}

   std::string GetName() const { return fName; }
   EColumnType GetType() const { return fType; }
   bool GetIsSorted() const { return fIsSorted; }
};

} // namespace Experimental
} // namespace ROOT

#endif
