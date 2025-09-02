/// \file
/// \warning This is part of the %ROOT 7 prototype! It will change without notice. It might trigger earthquakes.
/// Feedback is welcome!

#ifndef ROOT_RWeight
#define ROOT_RWeight

namespace ROOT {
namespace Experimental {

/**
A weight for filling histograms.

\warning This is part of the %ROOT 7 prototype! It will change without notice. It might trigger earthquakes.
Feedback is welcome!
*/
struct RWeight final {
   double fValue;

   explicit RWeight(double value) : fValue(value) {}
};

} // namespace Experimental
} // namespace ROOT

#endif
