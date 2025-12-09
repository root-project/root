/// \file
/// \warning This is part of the %ROOT 7 prototype! It will change without notice. It might trigger earthquakes.
/// Feedback is welcome!

#ifndef ROOT_RHistAutoAxisFiller
#define ROOT_RHistAutoAxisFiller

#include "RHist.hxx"
#include "RHistEngine.hxx"
#include "RWeight.hxx"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <type_traits> // for std::conditional_t
#include <utility>
#include <vector>

namespace ROOT {
namespace Experimental {

/**
A histogram filler that automatically determines the axis interval.

This class allows filling a regular one-dimensional histogram without specifying an axis interval during construction.
After a configurable number of buffered entries, or upon request, a RRegularAxis is constructed using the minimum and
maximum values until that point. This ensures all initial entries are filled into normal bins. Note that this cannot be
guaranteed for further calls to Fill.

\code
ROOT::Experimental::RHistAutoAxisFiller<int> filler(20);
filler.Fill(1.0);
filler.Fill(1.5);
filler.Fill(2.0);

// The following will implicitly trigger the histogram creation
auto &hist = filler.GetHist();
// hist.GetNEntries() will return 3
\endcode

\warning This is part of the %ROOT 7 prototype! It will change without notice. It might trigger earthquakes.
Feedback is welcome!
*/
template <typename BinContentType>
class RHistAutoAxisFiller final {
public:
   static constexpr bool SupportsWeightedFilling = RHistEngine<BinContentType>::SupportsWeightedFilling;

private:
   /// The filled histogram, after it has been constructed
   std::optional<RHist<BinContentType>> fHist;

   /// The number of normal bins
   std::uint64_t fNNormalBins;
   /// The maximum buffer size until Flush() is automatically called
   std::size_t fMaxBufferSize;
   /// The fraction of the axis interval to use as margin
   double fMarginFraction;

   using BufferElement = std::conditional_t<SupportsWeightedFilling, std::pair<double, RWeight>, double>;

   /// The buffer of filled entries
   std::vector<BufferElement> fBuffer;
   /// The minimum of the filled entries
   double fMinimum = std::numeric_limits<double>::infinity();
   /// The maximum of the filled entries
   double fMaximum = -std::numeric_limits<double>::infinity();

public:
   /// Create a filler object.
   ///
   /// \param[in] nNormalBins the number of normal bins, must be > 0
   /// \param[in] maxBufferSize the maximum buffer size, must be > 0
   /// \param[in] marginFraction the fraction of the axis interval to use as margin, must be > 0
   explicit RHistAutoAxisFiller(std::uint64_t nNormalBins, std::size_t maxBufferSize = 1024,
                                double marginFraction = 0.05)
      : fNNormalBins(nNormalBins), fMaxBufferSize(maxBufferSize), fMarginFraction(marginFraction)
   {
      if (nNormalBins == 0) {
         throw std::invalid_argument("nNormalBins must be > 0");
      }
      if (maxBufferSize == 0) {
         throw std::invalid_argument("maxBufferSize must be > 0");
      }
      if (marginFraction <= 0) {
         throw std::invalid_argument("marginFraction must be > 0");
      }
   }

   std::uint64_t GetNNormalBins() const { return fNNormalBins; }
   std::size_t GetMaxBufferSize() const { return fMaxBufferSize; }
   double GetMarginFraction() const { return fMarginFraction; }

private:
   void BufferImpl(double x, RWeight weight)
   {
      if constexpr (SupportsWeightedFilling) {
         fBuffer.emplace_back(x, weight);
      } else {
         assert(weight.fValue == 1.0);
         // Silence compiler warning about unused parameter
         (void)weight;
         fBuffer.push_back(x);
      }
      fMinimum = std::min(fMinimum, x);
      fMaximum = std::max(fMaximum, x);

      if (fBuffer.size() >= fMaxBufferSize) {
         Flush();
      }
   }

public:
   /// Fill an entry into the histogram.
   ///
   /// \param[in] x the argument
   /// \par See also
   /// the \ref Fill(double x, RWeight weight) "overload for weighted filling"
   void Fill(double x)
   {
      // If the histogram exists, forward the Fill call.
      if (fHist) {
         fHist->Fill(x);
         return;
      }
      BufferImpl(x, RWeight(1.0));
   }

   /// Fill an entry into the histogram with a weight.
   ///
   /// This overload is only available for floating-point bin content types (see
   /// \ref RHistEngine::SupportsWeightedFilling).
   ///
   /// \param[in] x the argument
   /// \param[in] weight the weight for this entry
   /// \par See also
   /// the \ref Fill(double x) "overload for unweighted filling"
   void Fill(double x, RWeight weight)
   {
      // If the histogram exists, forward the Fill call.
      if (fHist) {
         fHist->Fill(x, weight);
         return;
      }
      BufferImpl(x, weight);
   }

   /// Flush the buffer of entries and construct the histogram.
   ///
   /// Throws an exception if the buffer is empty, the axis interval cannot be determined, or if it would be empty
   /// because the minimum equals the maximum.
   void Flush()
   {
      if (fHist) {
         assert(fBuffer.empty() && "buffer should have been emptied");
         return;
      }

      if (fBuffer.empty()) {
         throw std::runtime_error("buffer is empty, cannot create histogram");
      }
      if (!std::isfinite(fMinimum) || !std::isfinite(fMaximum)) {
         throw std::runtime_error("could not determine axis interval");
      }
      if (fMinimum == fMaximum) {
         throw std::runtime_error("axis interval is empty");
      }

      // Add some margin to the axis interval to make sure the maximum is included in the last bin, but also to
      // accommodate closeby values.
      const auto margin = fMarginFraction * (fMaximum - fMinimum);
      const auto high = fMaximum + margin;
      const auto low = fMinimum - margin;
      assert(high > low);
      fHist.emplace(fNNormalBins, std::make_pair(low, high));

      for (auto &&x : fBuffer) {
         if constexpr (SupportsWeightedFilling) {
            fHist->Fill(x.first, x.second);
         } else {
            fHist->Fill(x);
         }
      }
      fBuffer.clear();
   }

   /// Return the constructed histogram.
   ///
   /// \see Flush()
   RHist<BinContentType> &GetHist()
   {
      Flush();
      assert(fHist.has_value());
      return *fHist;
   }
};

} // namespace Experimental
} // namespace ROOT

#endif
