/// \file
/// \warning This is part of the %ROOT 7 prototype! It will change without notice. It might trigger earthquakes.
/// Feedback is welcome!

#ifndef ROOT_Hist_ConversionUtils
#define ROOT_Hist_ConversionUtils

class TAxis;

namespace ROOT {
namespace Experimental {

class RAxisVariant;

namespace Hist {
namespace Internal {

/// Convert a single axis object to TAxis.
///
/// \param[out] dst the target TAxis object
/// \param[in] src the input axis to convert
void ConvertAxis(TAxis &dst, const RAxisVariant &src);

} // namespace Internal
} // namespace Hist
} // namespace Experimental
} // namespace ROOT

#endif
