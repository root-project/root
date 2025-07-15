/// \file
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

#ifndef ROOT_RVariableBinAxis
#define ROOT_RVariableBinAxis

#include "RLinearizedIndex.hxx"

#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

class TBuffer;

namespace ROOT {
namespace Experimental {

/**
An axis with variable bins defined by their edges.

For example, the following creates an axis with 3 log-spaced bins:
~~~ {.cxx}
std::vector<double> binEdges = {1, 10, 100, 1000};
ROOT::Experimental::RVariableBinAxis axis(binEdges);
~~~

It is possible to disable underflow and overflow bins by passing `enableFlowBins = false`. In that case, arguments
outside the axis will be silently discarded.

\warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is
welcome!
*/
class RVariableBinAxis final {
   /// The (ordered) edges of the normal bins
   std::vector<double> fBinEdges;
   /// Whether underflow and overflow bins are enabled
   bool fEnableFlowBins;

public:
   /// Construct an axis object with variable bins.
   ///
   /// \param[in] binEdges the (ordered) edges of the normal bins
   /// \param[in] enableFlowBins whether to enable underflow and overflow bins
   RVariableBinAxis(std::vector<double> binEdges, bool enableFlowBins = true)
      : fBinEdges(std::move(binEdges)), fEnableFlowBins(enableFlowBins)
   {
      // FIXME: should validate that fBinEdges is sorted
   }

   std::size_t GetNumNormalBins() const { return fBinEdges.size() - 1; }
   std::size_t GetTotalNumBins() const { return fEnableFlowBins ? fBinEdges.size() + 1 : fBinEdges.size() - 1; }
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

   /// ROOT Streamer function to throw when trying to store an object of this class.
   void Streamer(TBuffer &) { throw std::runtime_error("unable to store RVariableBinAxis"); }
};

} // namespace Experimental
} // namespace ROOT

#endif
