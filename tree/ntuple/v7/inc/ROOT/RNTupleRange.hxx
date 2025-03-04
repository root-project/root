/// \file ROOT/RNTupleRange.hxx
/// \ingroup NTuple ROOT7
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2018-10-05
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RNTupleRange
#define ROOT7_RNTupleRange

#include <ROOT/RNTupleUtil.hxx>

namespace ROOT {

// clang-format off
/**
\class ROOT::RNTupleGlobalRange
\ingroup NTuple
\brief Used to loop over indexes (entries or collections) between start and end
*/
// clang-format on
class RNTupleGlobalRange {
private:
   ROOT::NTupleSize_t fStart;
   ROOT::NTupleSize_t fEnd;

public:
   class RIterator {
   private:
      ROOT::NTupleSize_t fIndex = ROOT::kInvalidNTupleIndex;

   public:
      using iterator = RIterator;
      using iterator_category = std::forward_iterator_tag;
      using value_type = ROOT::NTupleSize_t;
      using difference_type = ROOT::NTupleSize_t;
      using pointer = ROOT::NTupleSize_t *;
      using reference = ROOT::NTupleSize_t &;

      RIterator() = default;
      explicit RIterator(ROOT::NTupleSize_t index) : fIndex(index) {}
      ~RIterator() = default;

      iterator operator++(int) /* postfix */
      {
         auto r = *this;
         fIndex++;
         return r;
      }
      iterator &operator++() /* prefix */
      {
         ++fIndex;
         return *this;
      }
      reference operator*() { return fIndex; }
      pointer operator->() { return &fIndex; }
      bool operator==(const iterator &rh) const { return fIndex == rh.fIndex; }
      bool operator!=(const iterator &rh) const { return fIndex != rh.fIndex; }
   };

   RNTupleGlobalRange(ROOT::NTupleSize_t start, ROOT::NTupleSize_t end) : fStart(start), fEnd(end) {}
   RIterator begin() const { return RIterator(fStart); }
   RIterator end() const { return RIterator(fEnd); }
   ROOT::NTupleSize_t size() const { return fEnd - fStart; }
   bool IsValid() const { return (fStart != ROOT::kInvalidNTupleIndex) && (fEnd != ROOT::kInvalidNTupleIndex); }
};

// clang-format off
/**
\class ROOT::RNTupleLocalRange
\ingroup NTuple
\brief Used to loop over entries of collections in a single cluster
*/
// clang-format on
class RNTupleLocalRange {
private:
   const ROOT::DescriptorId_t fClusterId;
   const ROOT::NTupleSize_t fStart;
   const ROOT::NTupleSize_t fEnd;

public:
   class RIterator {
   private:
      RNTupleLocalIndex fLocalIndex;

   public:
      using iterator = RIterator;
      using iterator_category = std::forward_iterator_tag;
      using value_type = RNTupleLocalIndex;
      using difference_type = RNTupleLocalIndex;
      using pointer = RNTupleLocalIndex *;
      using reference = RNTupleLocalIndex &;

      RIterator() = default;
      explicit RIterator(RNTupleLocalIndex localIndex) : fLocalIndex(localIndex) {}
      ~RIterator() = default;

      iterator operator++(int) /* postfix */
      {
         auto r = *this;
         fLocalIndex++;
         return r;
      }
      iterator &operator++() /* prefix */
      {
         fLocalIndex++;
         return *this;
      }
      reference operator*() { return fLocalIndex; }
      pointer operator->() { return &fLocalIndex; }
      bool operator==(const iterator &rh) const { return fLocalIndex == rh.fLocalIndex; }
      bool operator!=(const iterator &rh) const { return fLocalIndex != rh.fLocalIndex; }
   };

   RNTupleLocalRange(ROOT::DescriptorId_t clusterId, ROOT::NTupleSize_t start, ROOT::NTupleSize_t end)
      : fClusterId(clusterId), fStart(start), fEnd(end)
   {
   }
   RIterator begin() const { return RIterator(RNTupleLocalIndex(fClusterId, fStart)); }
   RIterator end() const { return RIterator(RNTupleLocalIndex(fClusterId, fEnd)); }
   ROOT::NTupleSize_t size() const { return fEnd - fStart; }
};

namespace Experimental {
// TODO(gparolini): remove before branching ROOT v6.36
using RNTupleGlobalRange [[deprecated("ROOT::Experimental::RNTupleGlobalRange moved to ROOT::RNTupleGlobalRange")]] =
   ROOT::RNTupleGlobalRange;
using RNTupleClusterRange [[deprecated("ROOT::Experimental::RNTupleClusterRange moved to ROOT::RNTupleLocalRange")]] =
   ROOT::RNTupleLocalRange;
} // namespace Experimental

} // namespace ROOT

#endif
