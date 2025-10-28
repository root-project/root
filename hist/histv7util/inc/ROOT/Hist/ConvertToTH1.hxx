/// \file
/// \warning This is part of the %ROOT 7 prototype! It will change without notice. It might trigger earthquakes.
/// Feedback is welcome!

#ifndef ROOT_Hist_ConvertToTH1
#define ROOT_Hist_ConvertToTH1

#include <ROOT/RHist.hxx>
#include <ROOT/RHistEngine.hxx>

class TH1I;

#include <memory>

namespace ROOT {
namespace Experimental {
namespace Hist {

/// Convert a one-dimensional histogram to TH1I.
///
/// As RHistEngine does not have global statistics, the number of entries and the total sum of weights will be unset.
///
/// Throws an exception if the histogram has more than one dimension.
///
/// \param[in] engine the RHistEngine to convert
/// \return the converted TH1I
std::unique_ptr<TH1I> ConvertToTH1I(const RHistEngine<int> &engine);

/// Convert a one-dimensional histogram to TH1I.
///
/// Throws an exception if the histogram has more than one dimension.
///
/// \param[in] hist the RHist to convert
/// \return the converted TH1I
std::unique_ptr<TH1I> ConvertToTH1I(const RHist<int> &hist);

} // namespace Hist
} // namespace Experimental
} // namespace ROOT

#endif
