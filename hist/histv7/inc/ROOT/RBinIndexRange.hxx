/// \file
/// \warning This is part of the %ROOT 7 prototype! It will change without notice. It might trigger earthquakes.
/// Feedback is welcome!

#ifndef ROOT_RBinIndexRange
#define ROOT_RBinIndexRange

#include "RBinIndex.hxx"

#include <cassert>
#include <cstddef> // for std::ptrdiff_t
#include <cstdint>
#include <iterator>

namespace ROOT {
namespace Experimental {

// forward declarations for friend declaration
class RBinIndexRange;
namespace Internal {
static RBinIndexRange CreateBinIndexRange(RBinIndex begin, RBinIndex end, std::uint64_t nNormalBins);
} // namespace Internal

/**
A range of bin indices \f$[fBegin, fEnd)\f$.

The interface allows convenient iteration over RBinIndex. If included, RBinIndex::Underflow() is encountered before the
normal bins and RBinIndex::Overflow() is the last value.

\code
ROOT::Experimental::RRegularAxis axis(10, 0, 1);
for (auto index : axis.GetNormalRange(2, 5)) {
   // Will iterate over [2, 3, 4]
}
for (auto index : axis.GetFullRange()) {
   // Will iterate over all bins, starting with the underflow and ending with the overflow bin
}
\endcode

\warning This is part of the %ROOT 7 prototype! It will change without notice. It might trigger earthquakes.
Feedback is welcome!
*/
class RBinIndexRange final {
   friend RBinIndexRange Internal::CreateBinIndexRange(RBinIndex, RBinIndex, std::uint64_t);

   /// The begin of the range (inclusive)
   RBinIndex fBegin;
   /// The end of the range (exclusive)
   RBinIndex fEnd;
   /// The number of normal bins, after which iteration advances to RBinIndex::Overflow()
   std::uint64_t fNNormalBins = 0;

public:
   /// Construct an invalid bin index range.
   RBinIndexRange() = default;

   RBinIndex GetBegin() const { return fBegin; }
   RBinIndex GetEnd() const { return fEnd; }
   // fNNormalBins is not exposed because it might be confusing for partial ranges.

   friend bool operator==(const RBinIndexRange &lhs, const RBinIndexRange &rhs)
   {
      return lhs.fBegin == rhs.fBegin && lhs.fEnd == rhs.fEnd && lhs.fNNormalBins == rhs.fNNormalBins;
   }

   friend bool operator!=(const RBinIndexRange &lhs, const RBinIndexRange &rhs) { return !(lhs == rhs); }

   /// Iterator over RBinIndex.
   class RIterator final {
      /// The current bin index
      RBinIndex fIndex;
      /// The number of normal bins, after which iteration advances to RBinIndex::Overflow()
      std::uint64_t fNNormalBins = 0;

   public:
      using difference_type = std::ptrdiff_t;
      using value_type = RBinIndex;
      using pointer = const RBinIndex *;
      using reference = RBinIndex;
      using iterator_category = std::input_iterator_tag;

      RIterator() = default;
      RIterator(RBinIndex index, std::uint64_t nNormalBins) : fIndex(index), fNNormalBins(nNormalBins) {}

      RIterator &operator++()
      {
         if (fIndex.IsUnderflow()) {
            fIndex = 0;
         } else if (fIndex.IsOverflow()) {
            fIndex = RBinIndex();
         } else if (fIndex.IsInvalid()) {
            // This should never happen! In the worst case, when built with NDEBUG, the iterator stays at Invalid.
            assert(0); // GCOVR_EXCL_LINE
         } else {
            fIndex++;
            if (fIndex.GetIndex() == fNNormalBins) {
               fIndex = RBinIndex::Overflow();
            }
         }
         return *this;
      }
      RIterator operator++(int)
      {
         RIterator old = *this;
         operator++();
         return old;
      }

      RBinIndex operator*() const { return fIndex; }
      const RBinIndex *operator->() const { return &fIndex; }

      friend bool operator==(const RIterator &lhs, const RIterator &rhs)
      {
         return lhs.fIndex == rhs.fIndex && lhs.fNNormalBins == rhs.fNNormalBins;
      }
      friend bool operator!=(const RIterator &lhs, const RIterator &rhs) { return !(lhs == rhs); }
   };

   RIterator begin() const { return RIterator(fBegin, fNNormalBins); }
   RIterator end() const { return RIterator(fEnd, fNNormalBins); }
};

namespace Internal {

/// %Internal function to create RBinIndexRange.
///
/// Users are strongly advised to create bin index ranges via the respective axis types, for example with
/// \ref RRegularAxis::GetNormalRange(RBinIndex, RBinIndex) const "RRegularAxis::GetNormalRange(RBinIndex, RBinIndex)"
/// or RRegularAxis::GetFullRange().
///
/// \param[in] begin the begin of the bin index range (inclusive)
/// \param[in] end the end of the bin index range (exclusive)
/// \param[in] nNormalBins the number of normal bins, after which iteration advances to RBinIndex::Overflow()
static RBinIndexRange CreateBinIndexRange(RBinIndex begin, RBinIndex end, std::uint64_t nNormalBins)
{
   RBinIndexRange range;
   range.fBegin = begin;
   range.fEnd = end;
   range.fNNormalBins = nNormalBins;
   return range;
}

} // namespace Internal

} // namespace Experimental
} // namespace ROOT

#endif
