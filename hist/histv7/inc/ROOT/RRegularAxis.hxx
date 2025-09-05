/// \file
/// \warning This is part of the %ROOT 7 prototype! It will change without notice. It might trigger earthquakes.
/// Feedback is welcome!

#ifndef ROOT_RRegularAxis
#define ROOT_RRegularAxis

#include "RBinIndex.hxx"
#include "RBinIndexRange.hxx"
#include "RLinearizedIndex.hxx"

#include <cassert>
#include <cstddef>
#include <stdexcept>
#include <string>

class TBuffer;

namespace ROOT {
namespace Experimental {

/**
A regular axis with equidistant bins in the interval \f$[fLow, fHigh)\f$.

For example, the following creates a regular axis with 10 normal bins between 5 and 15:
\code
ROOT::Experimental::RRegularAxis axis(10, 5, 15);
\endcode

It is possible to disable underflow and overflow bins by passing `enableFlowBins = false`. In that case, arguments
outside the axis will be silently discarded.

\warning This is part of the %ROOT 7 prototype! It will change without notice. It might trigger earthquakes.
Feedback is welcome!
*/
class RRegularAxis final {
   /// The number of normal bins
   std::size_t fNNormalBins;
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
   /// \param[in] nNormalBins the number of normal bins, must be > 0
   /// \param[in] low the lower end of the axis interval (inclusive)
   /// \param[in] high the upper end of the axis interval (exclusive), must be > low
   /// \param[in] enableFlowBins whether to enable underflow and overflow bins
   RRegularAxis(std::size_t nNormalBins, double low, double high, bool enableFlowBins = true)
      : fNNormalBins(nNormalBins), fLow(low), fHigh(high), fEnableFlowBins(enableFlowBins)
   {
      if (nNormalBins == 0) {
         throw std::invalid_argument("nNormalBins must be > 0");
      }
      if (low >= high) {
         std::string msg = "high must be > low, but " + std::to_string(low) + " >= " + std::to_string(high);
         throw std::invalid_argument(msg);
      }
      fInvBinWidth = nNormalBins / (high - low);
   }

   std::size_t GetNNormalBins() const { return fNNormalBins; }
   std::size_t GetTotalNBins() const { return fEnableFlowBins ? fNNormalBins + 2 : fNNormalBins; }
   double GetLow() const { return fLow; }
   double GetHigh() const { return fHigh; }
   bool HasFlowBins() const { return fEnableFlowBins; }

   friend bool operator==(const RRegularAxis &lhs, const RRegularAxis &rhs)
   {
      return lhs.fNNormalBins == rhs.fNNormalBins && lhs.fLow == rhs.fLow && lhs.fHigh == rhs.fHigh &&
             lhs.fEnableFlowBins == rhs.fEnableFlowBins;
   }

   /// Compute the linarized index for a single argument.
   ///
   /// The normal bins have indices \f$0\f$ to \f$fNNormalBins - 1\f$, the underflow bin has index
   /// \f$fNNormalBins\f$, and the overflow bin has index \f$fNNormalBins + 1\f$. If the argument is outside the
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
         return {fNNormalBins, fEnableFlowBins};
      } else if (overflow) {
         return {fNNormalBins + 1, fEnableFlowBins};
      }

      std::size_t bin = (x - fLow) * fInvBinWidth;
      return {bin, true};
   }

   /// Get the linearized index for an RBinIndex.
   ///
   /// The normal bins have indices \f$0\f$ to \f$fNNormalBins - 1\f$, the underflow bin has index
   /// \f$fNNormalBins\f$, and the overflow bin has index \f$fNNormalBins + 1\f$.
   ///
   /// \param[in] index the RBinIndex
   /// \return the linearized index that may be invalid
   RLinearizedIndex GetLinearizedIndex(RBinIndex index) const
   {
      if (index.IsUnderflow()) {
         return {fNNormalBins, fEnableFlowBins};
      } else if (index.IsOverflow()) {
         return {fNNormalBins + 1, fEnableFlowBins};
      } else if (index.IsInvalid()) {
         return {0, false};
      }
      assert(index.IsNormal());
      std::size_t bin = index.GetIndex();
      return {bin, bin < fNNormalBins};
   }

   /// Get the range of all normal bins.
   ///
   /// \return the bin index range from the first to the last normal bin, inclusive
   RBinIndexRange GetNormalRange() const
   {
      return Internal::CreateBinIndexRange(RBinIndex(0), RBinIndex(fNNormalBins), 0);
   }

   /// Get a range of normal bins.
   ///
   /// \param[in] begin the begin of the bin index range (inclusive), must be normal
   /// \param[in] end the end of the bin index range (exclusive), must be normal and >= begin
   /// \return a bin index range \f$[begin, end)\f$
   RBinIndexRange GetNormalRange(RBinIndex begin, RBinIndex end) const
   {
      if (!begin.IsNormal()) {
         throw std::invalid_argument("begin must be a normal bin");
      }
      if (begin.GetIndex() >= fNNormalBins) {
         throw std::invalid_argument("begin must be inside the axis");
      }
      if (!end.IsNormal()) {
         throw std::invalid_argument("end must be a normal bin");
      }
      if (end.GetIndex() > fNNormalBins) {
         throw std::invalid_argument("end must be inside or past the axis");
      }
      if (!(end >= begin)) {
         throw std::invalid_argument("end must be >= begin");
      }
      return Internal::CreateBinIndexRange(begin, end, 0);
   }

   /// Get the full range of all bins.
   ///
   /// This includes underflow and overflow bins, if enabled.
   ///
   /// \return the bin index range of all bins
   RBinIndexRange GetFullRange() const
   {
      return fEnableFlowBins ? Internal::CreateBinIndexRange(RBinIndex::Underflow(), RBinIndex(), fNNormalBins)
                             : GetNormalRange();
   }

   /// %ROOT Streamer function to throw when trying to store an object of this class.
   void Streamer(TBuffer &) { throw std::runtime_error("unable to store RRegularAxis"); }
};

} // namespace Experimental
} // namespace ROOT

#endif
