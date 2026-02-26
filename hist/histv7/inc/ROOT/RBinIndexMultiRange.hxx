/// \file
/// \warning This is part of the %ROOT 7 prototype! It will change without notice. It might trigger earthquakes.
/// Feedback is welcome!

#ifndef ROOT_RBinIndexMultiRange
#define ROOT_RBinIndexMultiRange

#include "RBinIndex.hxx"
#include "RBinIndexRange.hxx"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <utility>
#include <vector>

namespace ROOT {
namespace Experimental {

/**
A multidimensional range of bin indices.

The interface allows convenient iteration over multiple RBinIndexRange. The result is available as vector of RBinIndex.

\warning This is part of the %ROOT 7 prototype! It will change without notice. It might trigger earthquakes.
Feedback is welcome!
*/
class RBinIndexMultiRange final {
   /// The original ranges
   std::vector<RBinIndexRange> fRanges;
   /// Whether there is an empty range
   bool fHasEmptyRange = false;

public:
   /// Construct an invalid bin index range.
   RBinIndexMultiRange() = default;
   /// Construct a multidimensional range of bin indices.
   RBinIndexMultiRange(std::vector<RBinIndexRange> ranges) : fRanges(std::move(ranges))
   {
      for (auto &&range : fRanges) {
         if (range.GetBegin() == range.GetEnd()) {
            fHasEmptyRange = true;
         }
      }
   }

   const std::vector<RBinIndexRange> &GetRanges() const { return fRanges; }

   friend bool operator==(const RBinIndexMultiRange &lhs, const RBinIndexMultiRange &rhs)
   {
      return lhs.fRanges == rhs.fRanges;
   }

   friend bool operator!=(const RBinIndexMultiRange &lhs, const RBinIndexMultiRange &rhs) { return !(lhs == rhs); }

   /// Iterator over RBinIndexMultiRange.
   class RIterator final {
      friend class RBinIndexMultiRange;

      /// The current iterators
      std::vector<RBinIndexRange::RIterator> fIterators;
      /// The current bin indices
      std::vector<RBinIndex> fIndices;
      /// Pointer to the original RBinIndexMultiRange
      const RBinIndexMultiRange *fMultiRange = nullptr;

      RIterator(const RBinIndexMultiRange &multiRange) : fMultiRange(&multiRange) {}

   public:
      using difference_type = std::ptrdiff_t;
      using value_type = std::vector<RBinIndex>;
      using pointer = const std::vector<RBinIndex> *;
      using reference = const std::vector<RBinIndex> &;
      using iterator_category = std::input_iterator_tag;

      RIterator() = default;

      RIterator &operator++()
      {
         const std::size_t N = fIterators.size();
         std::size_t j = 0;
         for (; j < N; j++) {
            // Reverse iteration order to advance the innermost index first.
            const std::size_t i = N - 1 - j;

            // Advance this iterator and get the index by dereferencing.
            // NB: We dereference even if reaching the end. This is fine because we know the implementation of
            // RBinIndexRange, in the worst case the returned RBinIndex will be invalid.
            fIterators[i]++;
            fIndices[i] = *fIterators[i];
            // If we have not reached the end, we are done (with this loop).
            if (fIndices[i] != fMultiRange->fRanges[i].GetEnd()) {
               break;
            }
         }
         // If we iterated until j = N, all fIterators and fIndices are at the end.
         if (j == N) {
            // Clear fIterators to compare equal to the empty iterator returned by end().
            fIterators.clear();
         } else {
            // Otherwise we need to wrap around the innermost dimensions.
            for (std::size_t k = 0; k < j; k++) {
               // Reverse the iteration order as above.
               const std::size_t i = N - 1 - k;
               fIterators[i] = fMultiRange->fRanges[i].begin();
               fIndices[i] = *fIterators[i];
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

      const std::vector<RBinIndex> &operator*() const { return fIndices; }
      const std::vector<RBinIndex> *operator->() const { return &fIndices; }

      friend bool operator==(const RIterator &lhs, const RIterator &rhs)
      {
         return lhs.fIterators == rhs.fIterators && lhs.fMultiRange == rhs.fMultiRange;
      }
      friend bool operator!=(const RIterator &lhs, const RIterator &rhs) { return !(lhs == rhs); }
   };

   RIterator begin() const
   {
      RIterator it(*this);
      // If there is an empty range, return an empty iterator.
      if (fHasEmptyRange) {
         return it;
      }

      for (auto &&range : fRanges) {
         it.fIterators.push_back(range.begin());
         it.fIndices.push_back(*it.fIterators.back());
      }
      return it;
   }
   RIterator end() const { return RIterator(*this); }
};

} // namespace Experimental
} // namespace ROOT

#endif
