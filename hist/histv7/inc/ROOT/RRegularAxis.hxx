/// \file
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

#ifndef ROOT_RRegularAxis
#define ROOT_RRegularAxis

#include "RLinearizedIndex.hxx"

#include <cstddef>
#include <stdexcept>

class TBuffer;

namespace ROOT {
namespace Experimental {

/**
A regular axis with equidistant bins in the interval \f$[fLow, fHigh)\f$.

For example, the following creates a regular axis with 10 normal bins between 5 and 15:
~~~ {.cxx}
ROOT::Experimental::RRegularAxis axis(10, 5, 15);
~~~

It is possible to disable underflow and overflow bins by passing `enableFlowBins = false`. In that case, arguments
outside the axis will be silently discarded.

\warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is
welcome!
*/
class RRegularAxis final {
   /// The number of normal bins
   std::size_t fNumNormalBins;
   /// The lower end of the axis interval
   double fLow;
   /// The upper end of the axis interval
   double fHigh;
   /// The cached inverse of the bin width to speed up ComputeLinearizedIndex
   double fInvBinWidth; //!
   /// Whether underflow and overflow bins are enabled
   bool fEnableFlowBins;

public:
   /// Construct a regular axis object.
   ///
   /// \param[in] numNormalBins the number of normal bins
   /// \param[in] low the lower end of the axis interval (inclusive)
   /// \param[in] high the upper end of the axis interval (exclusive)
   /// \param[in] enableFlowBins whether to enable underflow and overflow bins
   RRegularAxis(std::size_t numNormalBins, double low, double high, bool enableFlowBins = true)
      : fNumNormalBins(numNormalBins), fLow(low), fHigh(high), fEnableFlowBins(enableFlowBins)
   {
      // FIXME: should validate numNormalBins > 0 and low < high
      fInvBinWidth = numNormalBins / (high - low);
   }

   std::size_t GetNumNormalBins() const { return fNumNormalBins; }
   std::size_t GetTotalNumBins() const { return fEnableFlowBins ? fNumNormalBins + 2 : fNumNormalBins; }
   double GetLow() const { return fLow; }
   double GetHigh() const { return fHigh; }
   bool HasFlowBins() const { return fEnableFlowBins; }

   friend bool operator==(const RRegularAxis &lhs, const RRegularAxis &rhs)
   {
      return lhs.fNumNormalBins == rhs.fNumNormalBins && lhs.fLow == rhs.fLow && lhs.fHigh == rhs.fHigh &&
             lhs.fEnableFlowBins == rhs.fEnableFlowBins;
   }

   /// Compute the linarized index for a single argument.
   ///
   /// The normal bins have indices \f$0\f$ to \f$fNumNormalBins - 1\f$, the underflow bin has index
   /// \f$fNumNormalBins\f$, and the overflow bin has index \f$fNumNormalBins + 1\f$. If the argument is outside the
   /// interval \f$[fLow, fHigh)\f$ and the flow bins are disabled, the return value is invalid.
   ///
   /// \param[in] x the argument
   /// \return the linearized index that may be invalid
   RLinearizedIndex ComputeLinearizedIndex(double x) const
   {
      bool underflow = x < fLow;
      // Put NaNs into overflow bin.
      bool overflow = !(x < fHigh);
      if (underflow) {
         return {fNumNormalBins, fEnableFlowBins};
      } else if (overflow) {
         return {fNumNormalBins + 1, fEnableFlowBins};
      }

      std::size_t bin = (x - fLow) * fInvBinWidth;
      return {bin, true};
   }

   /// ROOT Streamer function to throw when trying to store an object of this class.
   void Streamer(TBuffer &) { throw std::runtime_error("unable to store RRegularAxis"); }
};

} // namespace Experimental
} // namespace ROOT

#endif
