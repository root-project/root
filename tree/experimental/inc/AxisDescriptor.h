#ifndef RHnAxis_H
#define RHnAxis_H

#include <array>

namespace ROOT {
namespace Experimental {
struct AxisDescriptor {
   int fNbins;  ///< Number of bins(1D) WITH u/overflow
   double fMin; ///< Low edge of first bin
   double fMax; ///< Upper edge of last bin

   const double *kBinEdges; ///< Bin edges array, NULL when using fixed bins.
};

} // namespace Experimental
} // namespace ROOT

#endif