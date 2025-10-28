/// \file
/// \warning This is part of the %ROOT 7 prototype! It will change without notice. It might trigger earthquakes.
/// Feedback is welcome!

#ifndef ROOT_Hist_ConvertToTH1
#define ROOT_Hist_ConvertToTH1

#include <ROOT/RBinWithError.hxx>
#include <ROOT/RHist.hxx>
#include <ROOT/RHistEngine.hxx>

class TH1C;
class TH1S;
class TH1I;
class TH1L;
class TH1F;
class TH1D;

#include <memory>

namespace ROOT {
namespace Experimental {
namespace Hist {

/// Convert a one-dimensional histogram to TH1C.
///
/// \copydetails ConvertToTH1I(const RHistEngine<int> &engine)
std::unique_ptr<TH1C> ConvertToTH1C(const RHistEngine<char> &engine);

/// Convert a one-dimensional histogram to TH1S.
///
/// \copydetails ConvertToTH1I(const RHistEngine<int> &engine)
std::unique_ptr<TH1S> ConvertToTH1S(const RHistEngine<short> &engine);

/// Convert a one-dimensional histogram to TH1I.
///
/// As RHistEngine does not have global statistics, the number of entries and the total sum of weights will be unset.
///
/// Throws an exception if the histogram has more than one dimension.
///
/// \param[in] engine the RHistEngine to convert
/// \return the converted TH1
std::unique_ptr<TH1I> ConvertToTH1I(const RHistEngine<int> &engine);

/// Convert a one-dimensional histogram to TH1L.
///
/// \copydetails ConvertToTH1I(const RHistEngine<int> &engine)
std::unique_ptr<TH1L> ConvertToTH1L(const RHistEngine<long> &engine);

/// Convert a one-dimensional histogram to TH1L.
///
/// \copydetails ConvertToTH1I(const RHistEngine<int> &engine)
std::unique_ptr<TH1L> ConvertToTH1L(const RHistEngine<long long> &engine);

/// Convert a one-dimensional histogram to TH1F.
///
/// \copydetails ConvertToTH1I(const RHistEngine<int> &engine)
std::unique_ptr<TH1F> ConvertToTH1F(const RHistEngine<float> &engine);

/// Convert a one-dimensional histogram to TH1D.
///
/// \copydetails ConvertToTH1I(const RHistEngine<int> &engine)
std::unique_ptr<TH1D> ConvertToTH1D(const RHistEngine<double> &engine);

/// Convert a one-dimensional histogram to TH1D.
///
/// \copydetails ConvertToTH1I(const RHistEngine<int> &engine)
std::unique_ptr<TH1D> ConvertToTH1D(const RHistEngine<RBinWithError> &engine);

/// Convert a one-dimensional histogram to TH1C.
///
/// \copydetails ConvertToTH1I(const RHist<int> &hist)
std::unique_ptr<TH1C> ConvertToTH1C(const RHist<char> &hist);

/// Convert a one-dimensional histogram to TH1S.
///
/// \copydetails ConvertToTH1I(const RHist<int> &hist)
std::unique_ptr<TH1S> ConvertToTH1S(const RHist<short> &hist);

/// Convert a one-dimensional histogram to TH1I.
///
/// Throws an exception if the histogram has more than one dimension.
///
/// \param[in] hist the RHist to convert
/// \return the converted TH1
std::unique_ptr<TH1I> ConvertToTH1I(const RHist<int> &hist);

/// Convert a one-dimensional histogram to TH1L.
///
/// \copydetails ConvertToTH1I(const RHist<int> &hist)
std::unique_ptr<TH1L> ConvertToTH1L(const RHist<long> &hist);

/// Convert a one-dimensional histogram to TH1L.
///
/// \copydetails ConvertToTH1I(const RHist<int> &hist)
std::unique_ptr<TH1L> ConvertToTH1L(const RHist<long long> &hist);

/// Convert a one-dimensional histogram to TH1F.
///
/// \copydetails ConvertToTH1I(const RHist<int> &hist)
std::unique_ptr<TH1F> ConvertToTH1F(const RHist<float> &hist);

/// Convert a one-dimensional histogram to TH1D.
///
/// \copydetails ConvertToTH1I(const RHist<int> &hist)
std::unique_ptr<TH1D> ConvertToTH1D(const RHist<double> &hist);

/// Convert a one-dimensional histogram to TH1D.
///
/// \copydetails ConvertToTH1I(const RHist<int> &hist)
std::unique_ptr<TH1D> ConvertToTH1D(const RHist<RBinWithError> &hist);

} // namespace Hist
} // namespace Experimental
} // namespace ROOT

#endif
