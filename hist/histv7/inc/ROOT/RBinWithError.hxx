/// \file
/// \warning This is part of the %ROOT 7 prototype! It will change without notice. It might trigger earthquakes.
/// Feedback is welcome!

#ifndef ROOT_RBinWithError
#define ROOT_RBinWithError

#include "RHistUtils.hxx"

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

   void AtomicInc()
   {
      Internal::AtomicInc(&fSum);
      Internal::AtomicInc(&fSum2);
   }

   void AtomicAdd(double w)
   {
      Internal::AtomicAdd(&fSum, w);
      Internal::AtomicAdd(&fSum2, w * w);
   }

   /// Add another bin content using atomic instructions.
   ///
   /// \param[in] rhs another bin content that must not be modified during the operation
   void AtomicAdd(const RBinWithError &rhs)
   {
      Internal::AtomicAdd(&fSum, rhs.fSum);
      Internal::AtomicAdd(&fSum2, rhs.fSum2);
   }
};

} // namespace Experimental
} // namespace ROOT

#endif
