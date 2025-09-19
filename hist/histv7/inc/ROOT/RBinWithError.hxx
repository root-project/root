/// \file
/// \warning This is part of the %ROOT 7 prototype! It will change without notice. It might trigger earthquakes.
/// Feedback is welcome!

#ifndef ROOT_RBinWithError
#define ROOT_RBinWithError

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
};

} // namespace Experimental
} // namespace ROOT

#endif
