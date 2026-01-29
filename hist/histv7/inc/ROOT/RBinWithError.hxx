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
      Internal::AtomicAdd(&fSum, a);
      Internal::AtomicAdd(&fSum2, a2);
   }

public:
   void AtomicInc() { AtomicAdd(1.0, 1.0); }

   void AtomicAdd(double w) { AtomicAdd(w, w * w); }

   /// Add another bin content using atomic instructions.
   ///
   /// \param[in] rhs another bin content that must not be modified during the operation
   void AtomicAdd(const RBinWithError &rhs) { AtomicAdd(rhs.fSum, rhs.fSum2); }
};

} // namespace Experimental
} // namespace ROOT

#endif
