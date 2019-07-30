/// \file ROOT/RColumnModel.hxx
/// \ingroup NTuple ROOT7
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2018-10-09
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
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

namespace Detail {
class RColumnElementBase;
}

// clang-format off
/**
\class ROOT::Experimental::EColumnType
\ingroup NTuple
\brief The available trivial, native content types of a column

More complex types, such as classes, get translated into columns of such simple types by the RField.
*/
// clang-format on
enum class EColumnType {
   kUnknown = 0,
   // type for root columns of (nested) collections; 32bit integers that count
   // relative to the current cluster
   kIndex,
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

/**
 * Lookup table for the element size in bytes for column types. The array has to correspond to EColumnTypes.
 */
constexpr std::size_t kColumnElementSizes[] =
  {0 /* kUnknown */, 4 /* kIndex */, 1 /* kByte */, 8 /* kReal64 */, 4 /* kReal32 */, 2 /* kReal16 */,
   1 /* kReal8 */, 8 /* kInt64 */, 4 /* kInt32 */, 2 /* kInt16 */};

// clang-format off
/**
\class ROOT::Experimental::RColumnModel
\ingroup NTuple
\brief Holds the static meta-data of a column in a tree
*/
// clang-format on
class RColumnModel {
private:
   EColumnType fType;
   bool fIsSorted;

public:
   RColumnModel() : fType(EColumnType::kUnknown), fIsSorted(false) {}
   RColumnModel(EColumnType type, bool isSorted) : fType(type), fIsSorted(isSorted) {}

   std::size_t GetElementSize() const { return kColumnElementSizes[static_cast<int>(fType)]; }
   EColumnType GetType() const { return fType; }
   bool GetIsSorted() const { return fIsSorted; }

   Detail::RColumnElementBase *GenerateElement();
   bool operator ==(const RColumnModel &other) const {
      return (fType == other.fType) && (fIsSorted == other.fIsSorted);
   }
};

} // namespace Experimental
} // namespace ROOT

#endif
