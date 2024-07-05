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

#include <string_view>

#include <string>

namespace ROOT {
namespace Experimental {

// clang-format off
/**
\class ROOT::Experimental::EColumnType
\ingroup NTuple
\brief The available trivial, native content types of a column

More complex types, such as classes, get translated into columns of such simple types by the RField.
New types need to be accounted for in RColumnElementBase::Generate() and RColumnElementBase::GetBitsOnStorage(), too.
When changed, remember to update
  - RColumnElement::Generate()
  - RColumnElement::GetBitsOnStorage()
  - RColumnElement::GetTypeName()
  - RColumnElement template specializations / packing & unpacking
  - If necessary, endianess handling for the packing + unit test in ntuple_endian
  - RNTupleSerializer::[Des|S]erializeColumnType
*/
// clang-format on
enum class EColumnType {
   kUnknown = 0,
   // type for root columns of (nested) collections; offsets are relative to the current cluster
   kIndex64,
   kIndex32,
   // 96 bit column that is a pair of a kIndex64 and a 32bit dispatch tag to a column ID;
   // used to serialize std::variant.
   kSwitch,
   kByte,
   kChar,
   kBit,
   kReal64,
   kReal32,
   kReal16,
   kInt64,
   kUInt64,
   kInt32,
   kUInt32,
   kInt16,
   kUInt16,
   kInt8,
   kUInt8,
   kSplitIndex64,
   kSplitIndex32,
   kSplitReal64,
   kSplitReal32,
   kSplitInt64,
   kSplitUInt64,
   kSplitInt32,
   kSplitUInt32,
   kSplitInt16,
   kSplitUInt16,
   kMax,
};

} // namespace Experimental
} // namespace ROOT

#endif
