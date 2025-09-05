/// \file
/// \warning This is part of the %ROOT 7 prototype! It will change without notice. It might trigger earthquakes.
/// Feedback is welcome!

#ifndef ROOT_RVariableBinAxis
#define ROOT_RVariableBinAxis

#include "RBinIndex.hxx"
#include "RBinIndexRange.hxx"
#include "RLinearizedIndex.hxx"

#include <cassert>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

class TBuffer;

namespace ROOT {
namespace Experimental {

/**
An axis with variable bins defined by their edges.

For example, the following creates an axis with 3 log-spaced bins:
\code
std::vector<double> binEdges = {1, 10, 100, 1000};
ROOT::Experimental::RVariableBinAxis axis(binEdges);
\endcode

It is possible to disable underflow and overflow bins by passing `enableFlowBins = false`. In that case, arguments
outside the axis will be silently discarded.

\warning This is part of the %ROOT 7 prototype! It will change without notice. It might trigger earthquakes.
Feedback is welcome!
*/
class RVariableBinAxis final {
   /// The (ordered) edges of the normal bins
   std::vector<double> fBinEdges;
   /// Whether underflow and overflow bins are enabled
   bool fEnableFlowBins;

public:
   /// Construct an axis object with variable bins.
   ///
   /// \param[in] binEdges the (ordered) edges of the normal bins, must define at least one bin (i.e. size >= 2)
   /// \param[in] enableFlowBins whether to enable underflow and overflow bins
   explicit RVariableBinAxis(std::vector<double> binEdges, bool enableFlowBins = true)
      : fBinEdges(std::move(binEdges)), fEnableFlowBins(enableFlowBins)
   {
      if (fBinEdges.size() < 2) {
         throw std::invalid_argument("must have >= 2 bin edges");
      }
      for (std::size_t i = 1; i < fBinEdges.size(); i++) {
         if (fBinEdges[i - 1] >= fBinEdges[i]) {
            std::string msg = "binEdges must be sorted, but for bin " + std::to_string(i - 1) + ": ";
            msg += std::to_string(fBinEdges[i - 1]) + " >= " + std::to_string(fBinEdges[i]);
            throw std::invalid_argument(msg);
         }
      }
   }

   std::size_t GetNNormalBins() const { return fBinEdges.size() - 1; }
   std::size_t GetTotalNBins() const { return fEnableFlowBins ? fBinEdges.size() + 1 : fBinEdges.size() - 1; }
   const std::vector<double> &GetBinEdges() const { return fBinEdges; }
   bool HasFlowBins() const { return fEnableFlowBins; }

   friend bool operator==(const RVariableBinAxis &lhs, const RVariableBinAxis &rhs)
   {
      return lhs.fBinEdges == rhs.fBinEdges && lhs.fEnableFlowBins == rhs.fEnableFlowBins;
   }

   /// Compute the linarized index for a single argument.
   ///
   /// The normal bins have indices \f$0\f$ to \f$fBinEdges.size() - 2\f$, the underflow bin has index
   /// \f$fBinEdges.size() - 1\f$, and the overflow bin has index \f$fBinEdges.size()\f$. If the argument is outside all
   /// bin edges and the flow bins are disabled, the return value is invalid.
   ///
   /// \param[in] x the argument
   /// \return the linearized index that may be invalid
   RLinearizedIndex ComputeLinearizedIndex(double x) const
   {
      bool underflow = x < fBinEdges.front();
      // Put NaNs into overflow bin.
      bool overflow = !(x < fBinEdges.back());
      if (underflow) {
         return {fBinEdges.size() - 1, fEnableFlowBins};
      } else if (overflow) {
         return {fBinEdges.size(), fEnableFlowBins};
      }

      // TODO (for later): The following can be optimized with binary search...
      for (std::size_t bin = 0; bin < fBinEdges.size() - 2; bin++) {
         if (x < fBinEdges[bin + 1]) {
            return {bin, true};
         }
      }
      std::size_t bin = fBinEdges.size() - 2;
      return {bin, true};
   }

   /// Get the linearized index for an RBinIndex.
   ///
   /// The normal bins have indices \f$0\f$ to \f$fBinEdges.size() - 2\f$, the underflow bin has index
   /// \f$fBinEdges.size() - 1\f$, and the overflow bin has index \f$fBinEdges.size()\f$.
   ///
   /// \param[in] index the RBinIndex
   /// \return the linearized index that may be invalid
   RLinearizedIndex GetLinearizedIndex(RBinIndex index) const
   {
      if (index.IsUnderflow()) {
         return {fBinEdges.size() - 1, fEnableFlowBins};
      } else if (index.IsOverflow()) {
         return {fBinEdges.size(), fEnableFlowBins};
      } else if (index.IsInvalid()) {
         return {0, false};
      }
      assert(index.IsNormal());
      std::size_t bin = index.GetIndex();
      return {bin, bin < fBinEdges.size() - 1};
   }

   /// Get the range of all normal bins.
   ///
   /// \return the bin index range from the first to the last normal bin, inclusive
   RBinIndexRange GetNormalRange() const
   {
      return Internal::CreateBinIndexRange(RBinIndex(0), RBinIndex(fBinEdges.size() - 1), 0);
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
      if (begin.GetIndex() >= fBinEdges.size() - 1) {
         throw std::invalid_argument("begin must be inside the axis");
      }
      if (!end.IsNormal()) {
         throw std::invalid_argument("end must be a normal bin");
      }
      if (end.GetIndex() > fBinEdges.size() - 1) {
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
      return fEnableFlowBins ? Internal::CreateBinIndexRange(RBinIndex::Underflow(), RBinIndex(), fBinEdges.size() - 1)
                             : GetNormalRange();
   }

   /// %ROOT Streamer function to throw when trying to store an object of this class.
   void Streamer(TBuffer &) { throw std::runtime_error("unable to store RVariableBinAxis"); }
};

} // namespace Experimental
} // namespace ROOT

#endif
