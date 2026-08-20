/// \file
/// \warning This is part of the %ROOT 7 prototype! It will change without notice. It might trigger earthquakes.
/// Feedback is welcome!

#ifndef ROOT_Hist_ConvertToTH3
#define ROOT_Hist_ConvertToTH3

#include <ROOT/RBinWithError.hxx>
#include <ROOT/RHist.hxx>
#include <ROOT/RHistEngine.hxx>

class TH3C;
class TH3S;
class TH3I;
class TH3L;
class TH3F;
class TH3D;

#include <memory>

namespace ROOT {
namespace Experimental {
namespace Hist {

/// Convert a three-dimensional histogram to TH3C.
///
/// \copydetails ConvertToTH3I(const RHistEngine<int> &engine)
std::unique_ptr<TH3C> ConvertToTH3C(const RHistEngine<char> &engine);

/// Convert a three-dimensional histogram to TH3S.
///
/// \copydetails ConvertToTH3I(const RHistEngine<int> &engine)
std::unique_ptr<TH3S> ConvertToTH3S(const RHistEngine<short> &engine);

/// Convert a three-dimensional histogram to TH3I.
///
/// As RHistEngine does not have global statistics, the number of entries and the total sum of weights will be unset.
///
/// Throws an exception if the histogram does not have three dimensions.
///
/// \param[in] engine the RHistEngine to convert
/// \return the converted TH3
std::unique_ptr<TH3I> ConvertToTH3I(const RHistEngine<int> &engine);

/// Convert a three-dimensional histogram to TH3L.
///
/// \copydetails ConvertToTH3I(const RHistEngine<int> &engine)
std::unique_ptr<TH3L> ConvertToTH3L(const RHistEngine<long> &engine);

/// Convert a three-dimensional histogram to TH3L.
///
/// \copydetails ConvertToTH3I(const RHistEngine<int> &engine)
std::unique_ptr<TH3L> ConvertToTH3L(const RHistEngine<long long> &engine);

/// Convert a three-dimensional histogram to TH3F.
///
/// \copydetails ConvertToTH3I(const RHistEngine<int> &engine)
std::unique_ptr<TH3F> ConvertToTH3F(const RHistEngine<float> &engine);

/// Convert a three-dimensional histogram to TH3D.
///
/// \copydetails ConvertToTH3I(const RHistEngine<int> &engine)
std::unique_ptr<TH3D> ConvertToTH3D(const RHistEngine<double> &engine);

/// Convert a three-dimensional histogram to TH3D.
///
/// \copydetails ConvertToTH3I(const RHistEngine<int> &engine)
std::unique_ptr<TH3D> ConvertToTH3D(const RHistEngine<RBinWithError> &engine);

/// Convert a three-dimensional histogram to TH3C.
///
/// \copydetails ConvertToTH3I(const RHist<int> &hist)
std::unique_ptr<TH3C> ConvertToTH3C(const RHist<char> &hist);

/// Convert a three-dimensional histogram to TH3S.
///
/// \copydetails ConvertToTH3I(const RHist<int> &hist)
std::unique_ptr<TH3S> ConvertToTH3S(const RHist<short> &hist);

/// Convert a three-dimensional histogram to TH3I.
///
/// If the RHistStats are tainted, for example after setting bin contents, the number of entries and the total sum of
/// weights will be unset.
///
/// Throws an exception if the histogram does not have three dimensions.
///
/// \param[in] hist the RHist to convert
/// \return the converted TH3
std::unique_ptr<TH3I> ConvertToTH3I(const RHist<int> &hist);

/// Convert a three-dimensional histogram to TH3L.
///
/// \copydetails ConvertToTH3I(const RHist<int> &hist)
std::unique_ptr<TH3L> ConvertToTH3L(const RHist<long> &hist);

/// Convert a three-dimensional histogram to TH3L.
///
/// \copydetails ConvertToTH3I(const RHist<int> &hist)
std::unique_ptr<TH3L> ConvertToTH3L(const RHist<long long> &hist);

/// Convert a three-dimensional histogram to TH3F.
///
/// \copydetails ConvertToTH3I(const RHist<int> &hist)
std::unique_ptr<TH3F> ConvertToTH3F(const RHist<float> &hist);

/// Convert a three-dimensional histogram to TH3D.
///
/// \copydetails ConvertToTH3I(const RHist<int> &hist)
std::unique_ptr<TH3D> ConvertToTH3D(const RHist<double> &hist);

/// Convert a three-dimensional histogram to TH3D.
///
/// \copydetails ConvertToTH3I(const RHist<int> &hist)
std::unique_ptr<TH3D> ConvertToTH3D(const RHist<RBinWithError> &hist);

} // namespace Hist
} // namespace Experimental
} // namespace ROOT

#endif
