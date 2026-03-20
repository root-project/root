/// \file
/// \warning This is part of the %ROOT 7 prototype! It will change without notice. It might trigger earthquakes.
/// Feedback is welcome!

#ifndef ROOT_RBinWithError
#define ROOT_RBinWithError

#include "RHistUtils.hxx"

#include <cmath>
#include <cstring>

namespace ROOT {
namespace Experimental {

/**
A special bin content type to compute the bin error in weighted filling.

\warning This is part of the %ROOT 7 prototype! It will change without notice. It might trigger earthquakes.
Feedback is welcome!
*/
struct RBinWithError final {
   double fSum = 0;
   double fSum2 = 0;

   explicit operator float() const { return fSum; }
   explicit operator double() const { return fSum; }

   RBinWithError &operator++()
   {
      fSum++;
      fSum2++;
      return *this;
   }

   RBinWithError operator++(int)
   {
      RBinWithError old = *this;
      operator++();
      return old;
   }

   RBinWithError &operator+=(double w)
   {
      fSum += w;
      fSum2 += w * w;
      return *this;
   }

   RBinWithError &operator+=(const RBinWithError &rhs)
   {
      fSum += rhs.fSum;
      fSum2 += rhs.fSum2;
      return *this;
   }

   RBinWithError &operator*=(double factor)
   {
      fSum *= factor;
      fSum2 *= factor * factor;
      return *this;
   }

private:
   void AtomicAdd(double a, double a2)
   {
      // The sum of squares is always non-negative. Use the sign bit to implement a cheap spin lock.
      double origSum2;
      Internal::AtomicLoad(&fSum2, &origSum2);

      while (true) {
         // Repeat loads from memory until we see a non-negative value.
         // NB: do not use origSum2 < 0, it does not work for -0.0!
         while (std::signbit(origSum2)) {
            Internal::AtomicLoad(&fSum2, &origSum2);
         }

         // The variable appears to be unlocked, confirm with a compare-exchange. It uses acquire memory order in case
         // of success, so the release store synchronizes with this load. This ensures that the previous write of fSum
         // becomes a visible side-effect in this thread and the access below will see it.
         double negated = std::copysign(origSum2, -1.0);
         if (Internal::AtomicCompareExchangeAcquire(&fSum2, &origSum2, &negated)) {
            break;
         }
      }

      // By using a spin lock, we do not need atomic operations for fSum.
      fSum += a;

      double sum2 = origSum2 + a2;
      // This store synchronizes with the compare-exchange, see above.
      Internal::AtomicStoreRelease(&fSum2, &sum2);
   }

public:
   void AtomicInc() { AtomicAdd(1.0, 1.0); }

   void AtomicAdd(double w) { AtomicAdd(w, w * w); }

   /// Add another bin content using atomic instructions.
   ///
   /// \param[in] rhs another bin content that must not be modified during the operation
   void AtomicAdd(const RBinWithError &rhs) { AtomicAdd(rhs.fSum, rhs.fSum2); }

   void AtomicLoad(RBinWithError *ret) const
   {
      double origSum2;
      Internal::AtomicLoad(&fSum2, &origSum2);

      while (true) {
         // Repeat loads from memory until we see a non-negative value.
         // NB: do not use origSum2 < 0, it does not work for -0.0!
         while (std::signbit(origSum2)) {
            Internal::AtomicLoad(&fSum2, &origSum2);
         }

         Internal::AtomicLoad(&fSum, &ret->fSum);
         Internal::AtomicLoad(&fSum2, &ret->fSum2);

         // Check if the value of fSum2 is still identical to the first load.
         // NB: do not use origSum2 == ret->fSum2, it is problematic because 0.0 == -0.0!
         if (!std::memcmp(&origSum2, &ret->fSum2, sizeof(origSum2))) {
            return;
         }

         // The comparison failed, an update happened or is in progress.
         origSum2 = ret->fSum2;
      }
   }
};

} // namespace Experimental
} // namespace ROOT

#endif
