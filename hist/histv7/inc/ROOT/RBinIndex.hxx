/// \file
/// \warning This is part of the %ROOT 7 prototype! It will change without notice. It might trigger earthquakes.
/// Feedback is welcome!

#ifndef ROOT_RBinIndex
#define ROOT_RBinIndex

#include <cassert>
#include <cstddef>
#include <cstdint>

namespace ROOT {
namespace Experimental {

/**
A bin index with special values for underflow and overflow bins.

Objects of this type should be passed by value.

\warning This is part of the %ROOT 7 prototype! It will change without notice. It might trigger earthquakes.
Feedback is welcome!
*/
class RBinIndex final {
   static constexpr auto UnderflowIndex = static_cast<std::uint64_t>(-3);
   static constexpr auto OverflowIndex = static_cast<std::uint64_t>(-2);
   static constexpr auto InvalidIndex = static_cast<std::uint64_t>(-1);

   // We use std::uint64_t instead of std::size_t for the index because for sparse histograms, not all bins have to be
   // allocated in memory. However, we require that the index has at least that size.
   static_assert(sizeof(std::uint64_t) >= sizeof(std::size_t), "index type not large enough to address all bins");

   std::uint64_t fIndex = InvalidIndex;

public:
   /// Construct an invalid bin index.
   RBinIndex() = default;

   /// Construct a bin index for a normal bin.
   RBinIndex(std::uint64_t index) : fIndex(index) { assert(IsNormal()); }

   /// Return the index for a normal bin.
   std::uint64_t GetIndex() const
   {
      assert(IsNormal());
      return fIndex;
   }

   /// A bin index is normal iff it is not one of the special values.
   ///
   /// Note that a normal bin index may not actually be valid for a given axis if it is outside its range.
   bool IsNormal() const { return fIndex < UnderflowIndex; }
   bool IsUnderflow() const { return fIndex == UnderflowIndex; }
   bool IsOverflow() const { return fIndex == OverflowIndex; }
   bool IsInvalid() const { return fIndex == InvalidIndex; }

   RBinIndex &operator+=(std::uint64_t a)
   {
      if (!IsNormal()) {
         // Arithmetic operations on special values go to InvalidIndex.
         fIndex = InvalidIndex;
      } else {
         std::uint64_t old = fIndex;
         fIndex += a;
         if (fIndex < old || !IsNormal()) {
            // The addition wrapped around, go to InvalidIndex.
            fIndex = InvalidIndex;
         }
      }
      return *this;
   }

   RBinIndex operator+(std::uint64_t a) const
   {
      RBinIndex ret = *this;
      ret += a;
      return ret;
   }

   RBinIndex &operator++()
   {
      operator+=(1);
      return *this;
   }

   RBinIndex operator++(int)
   {
      RBinIndex old = *this;
      operator++();
      return old;
   }

   RBinIndex &operator-=(std::uint64_t a)
   {
      if (!IsNormal()) {
         // Arithmetic operations on special values go to InvalidIndex.
         fIndex = InvalidIndex;
      } else if (fIndex >= a) {
         fIndex -= a;
      } else {
         // The operation would wrap around, go to InvalidIndex.
         fIndex = InvalidIndex;
      }
      return *this;
   }

   RBinIndex operator-(std::uint64_t a) const
   {
      RBinIndex ret = *this;
      ret -= a;
      return ret;
   }

   RBinIndex &operator--()
   {
      operator-=(1);
      return *this;
   }

   RBinIndex operator--(int)
   {
      RBinIndex old = *this;
      operator--();
      return old;
   }

   friend bool operator==(RBinIndex lhs, RBinIndex rhs) { return lhs.fIndex == rhs.fIndex; }
   friend bool operator!=(RBinIndex lhs, RBinIndex rhs) { return !(lhs == rhs); }

   friend bool operator<(RBinIndex lhs, RBinIndex rhs)
   {
      if (lhs.IsNormal() && rhs.IsNormal()) {
         return lhs.fIndex < rhs.fIndex;
      }
      return false;
   }
   friend bool operator<=(RBinIndex lhs, RBinIndex rhs) { return lhs == rhs || lhs < rhs; }

   friend bool operator>(RBinIndex lhs, RBinIndex rhs)
   {
      if (lhs.IsNormal() && rhs.IsNormal()) {
         return lhs.fIndex > rhs.fIndex;
      }
      return false;
   }
   friend bool operator>=(RBinIndex lhs, RBinIndex rhs) { return lhs == rhs || lhs > rhs; }

   static RBinIndex Underflow()
   {
      RBinIndex underflow;
      underflow.fIndex = UnderflowIndex;
      return underflow;
   }

   static RBinIndex Overflow()
   {
      RBinIndex overflow;
      overflow.fIndex = OverflowIndex;
      return overflow;
   }
};

} // namespace Experimental
} // namespace ROOT

#endif
