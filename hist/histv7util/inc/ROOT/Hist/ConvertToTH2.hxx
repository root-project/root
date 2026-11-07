/// \file
/// \warning This is part of the %ROOT 7 prototype! It will change without notice. It might trigger earthquakes.
/// Feedback is welcome!

#ifndef ROOT_Hist_ConvertToTH2
#define ROOT_Hist_ConvertToTH2

#include <ROOT/RBinWithError.hxx>
#include <ROOT/RHist.hxx>
#include <ROOT/RHistEngine.hxx>

class TH2C;
class TH2S;
class TH2I;
class TH2L;
class TH2F;
class TH2D;

#include <memory>

namespace ROOT {
namespace Experimental {
namespace Hist {

/// Convert a two-dimensional histogram to TH2C.
///
/// \copydetails ConvertToTH2I(const RHistEngine<int> &engine)
std::unique_ptr<TH2C> ConvertToTH2C(const RHistEngine<char> &engine);

/// Convert a two-dimensional histogram to TH2S.
///
/// \copydetails ConvertToTH2I(const RHistEngine<int> &engine)
std::unique_ptr<TH2S> ConvertToTH2S(const RHistEngine<short> &engine);

/// Convert a two-dimensional histogram to TH2I.
///
/// As RHistEngine does not have global statistics, the number of entries and the total sum of weights will be unset.
///
/// Throws an exception if the histogram has more than one dimension.
///
/// \param[in] engine the RHistEngine to convert
/// \return the converted TH2
std::unique_ptr<TH2I> ConvertToTH2I(const RHistEngine<int> &engine);

/// Convert a two-dimensional histogram to TH2L.
///
/// \copydetails ConvertToTH2I(const RHistEngine<int> &engine)
std::unique_ptr<TH2L> ConvertToTH2L(const RHistEngine<long> &engine);

/// Convert a two-dimensional histogram to TH2L.
///
/// \copydetails ConvertToTH2I(const RHistEngine<int> &engine)
std::unique_ptr<TH2L> ConvertToTH2L(const RHistEngine<long long> &engine);

/// Convert a two-dimensional histogram to TH2F.
///
/// \copydetails ConvertToTH2I(const RHistEngine<int> &engine)
std::unique_ptr<TH2F> ConvertToTH2F(const RHistEngine<float> &engine);

/// Convert a two-dimensional histogram to TH2D.
///
/// \copydetails ConvertToTH2I(const RHistEngine<int> &engine)
std::unique_ptr<TH2D> ConvertToTH2D(const RHistEngine<double> &engine);

/// Convert a two-dimensional histogram to TH2D.
///
/// \copydetails ConvertToTH2I(const RHistEngine<int> &engine)
std::unique_ptr<TH2D> ConvertToTH2D(const RHistEngine<RBinWithError> &engine);

/// Convert a two-dimensional histogram to TH2C.
///
/// \copydetails ConvertToTH2I(const RHist<int> &hist)
std::unique_ptr<TH2C> ConvertToTH2C(const RHist<char> &hist);

/// Convert a two-dimensional histogram to TH2S.
///
/// \copydetails ConvertToTH2I(const RHist<int> &hist)
std::unique_ptr<TH2S> ConvertToTH2S(const RHist<short> &hist);

/// Convert a two-dimensional histogram to TH2I.
///
/// If the RHistStats are tainted, for example after setting bin contents, the number of entries and the total sum of
/// weights will be unset.
///
/// Throws an exception if the histogram has more than one dimension.
///
/// \param[in] hist the RHist to convert
/// \return the converted TH2
std::unique_ptr<TH2I> ConvertToTH2I(const RHist<int> &hist);

/// Convert a two-dimensional histogram to TH2L.
///
/// \copydetails ConvertToTH2I(const RHist<int> &hist)
std::unique_ptr<TH2L> ConvertToTH2L(const RHist<long> &hist);

/// Convert a two-dimensional histogram to TH2L.
///
/// \copydetails ConvertToTH2I(const RHist<int> &hist)
std::unique_ptr<TH2L> ConvertToTH2L(const RHist<long long> &hist);

/// Convert a two-dimensional histogram to TH2F.
///
/// \copydetails ConvertToTH2I(const RHist<int> &hist)
std::unique_ptr<TH2F> ConvertToTH2F(const RHist<float> &hist);

/// Convert a two-dimensional histogram to TH2D.
///
/// \copydetails ConvertToTH2I(const RHist<int> &hist)
std::unique_ptr<TH2D> ConvertToTH2D(const RHist<double> &hist);

/// Convert a two-dimensional histogram to TH2D.
///
/// \copydetails ConvertToTH2I(const RHist<int> &hist)
std::unique_ptr<TH2D> ConvertToTH2D(const RHist<RBinWithError> &hist);

} // namespace Hist
} // namespace Experimental
} // namespace ROOT

#endif
