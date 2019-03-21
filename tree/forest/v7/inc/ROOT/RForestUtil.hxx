/// \file ROOT/RForestUtil.hxx
/// \ingroup Forest ROOT7
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2018-10-04
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2015, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RForestUtil
#define ROOT7_RForestUtil

#include <cstdint>

#include <string>
#include <vector>

namespace ROOT {
namespace Experimental {

/**
 * Used in unit tests to serialize and deserialize classes with TClass
 */
struct RForestTest {
  float a = 0.0;
  std::vector<float> v1;
  std::vector<std::vector<float>> v2;
  std::string s;
};

/// Integer type long enough to hold the maximum number of entries in a column
using ForestIndex_t = std::uint64_t;
constexpr ForestIndex_t kInvalidForestIndex = std::uint64_t(-1);
/// Wrap the 32bit integer in a struct in order to avoid template specialization clash with std::uint32_t
struct RClusterIndex {
   RClusterIndex() : fValue(0) {}
   explicit constexpr RClusterIndex(std::uint32_t value) : fValue(value) {}
   RClusterIndex& operator =(const std::uint32_t value) { fValue = value; return *this; }
   RClusterIndex& operator +=(const std::uint32_t value) { fValue += value; return *this; }
   operator std::uint32_t() const { return fValue; }

   std::uint32_t fValue;
};
using ClusterIndex_t = RClusterIndex;
constexpr ClusterIndex_t kInvalidClusterIndex(std::uint32_t(-1));

/// Uniquely identifies a physical column within the scope of the current process, used to tag pages
using ColumnId_t = std::int64_t;
constexpr ColumnId_t kInvalidColumnId = -1;

} // namespace Experimental
} // namespace ROOT

#endif
