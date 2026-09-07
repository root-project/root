// @(#)root/hist:$Id$
// Author: Jonas Rembser, CERN  09/2026

/*************************************************************************
 * Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_Rebin2DHelpers
#define ROOT_Rebin2DHelpers

// Internal helpers shared by TH2::Rebin2D and TProfile2D::Rebin2D.

#include "TAxis.h"
#include "TH1.h"
#include "TMath.h"

#include <initializer_list>
#include <utility>
#include <vector>

namespace ROOT {
namespace Internal {

/// Define the axis of the rebinned histogram: either from the user-provided
/// bin edges, or by merging groups of ngroup bins of the old axis. The value
/// of xmax is the (possibly truncated) upper limit for the uniform-bin case.
inline void DefineRebinnedAxis(const TAxis &oldAxis, Int_t ngroup, Int_t nnew, const Double_t *userBins, Double_t xmin,
                               Double_t xmax, TAxis &newAxis)
{
   if (userBins) {
      newAxis.Set(nnew, userBins);
   } else if (oldAxis.GetXbins()->GetSize() > 0) {
      std::vector<Double_t> edges(nnew + 1);
      for (Int_t i = 0; i <= nnew; ++i)
         edges[i] = oldAxis.GetBinLowEdge(1 + i * ngroup);
      newAxis.Set(nnew, edges.data());
   } else {
      newAxis.Set(nnew, xmin, xmax);
   }
}

/// Map each cell of the old axis (including underflow 0 and overflow n+1) to
/// the cell of the new axis that contains its bin center. Old bins outside
/// the new axis range are mapped to the new under-/overflow.
inline std::vector<Int_t> MakeRebinMap(const TAxis &oldAxis, const TAxis &newAxis)
{
   const Int_t nOld = oldAxis.GetNbins();
   std::vector<Int_t> map(nOld + 2);
   map[0] = 0;
   map[nOld + 1] = newAxis.GetNbins() + 1;
   for (Int_t o = 1; o <= nOld; ++o)
      map[o] = newAxis.FindFixBin(oldAxis.GetBinCenter(o));
   return map;
}

/// Warn when a bin edge of the new axis that lies inside the old axis range
/// does not line up with a bin edge of the old axis: the entries of the old
/// bin that is split cannot be distributed correctly.
inline void WarnAboutMisalignedEdges(const TAxis &oldAxis, const TAxis &newAxis, TH1 &hist, const char *where)
{
   for (Int_t b = 0; b <= newAxis.GetNbins(); ++b) {
      const Double_t edge = newAxis.GetBinUpEdge(b); // GetBinUpEdge(0) is the axis minimum
      if (edge <= oldAxis.GetXmin() || edge >= oldAxis.GetXmax())
         continue;
      const Int_t o = oldAxis.FindFixBin(edge);
      const Double_t tol = TMath::Max(1.E-8 * oldAxis.GetBinWidth(o), 1.E-16);
      if (!TMath::AreEqualAbs(edge, oldAxis.GetBinLowEdge(o), tol) &&
          !TMath::AreEqualAbs(edge, oldAxis.GetBinUpEdge(o), tol)) {
         hist.Warning(where,
                      "Bin edge %d of rebinned histogram does not match any bin edges of the old histogram. "
                      "Result can be inconsistent",
                      b);
      }
   }
}

/// The definition of one axis of the rebinned histogram.
struct RebinnedAxisInfo {
   Int_t nNewBins = 0;        ///< number of bins of the rebinned axis
   TAxis newAxis;             ///< the rebinned axis
   std::vector<Int_t> binMap; ///< map from old cell (0..n+1) to new cell
   bool truncated = false;    ///< the group count does not divide the old bin count: top bins move to the overflow
};

/// Validate the rebinning parameters for one axis and fill the definition of
/// the rebinned axis and the map from old to new bins. For an axis with
/// user-provided bin edges, ngroup is directly the new number of bins,
/// otherwise the old bins are merged in groups of ngroup. Returns false on an
/// invalid group count.
inline bool SetupRebinnedAxis(const TAxis &oldAxis, Int_t ngroup, const Double_t *userBins, char axisName, TH1 &hist,
                              const char *where, RebinnedAxisInfo &info)
{
   const Int_t nOldBins = oldAxis.GetNbins();
   if (ngroup <= 0 || ngroup > nOldBins) {
      hist.Error(where, "Illegal value of n%cgroup=%d", axisName, ngroup);
      return false;
   }
   Double_t newMax = oldAxis.GetXmax();
   if (userBins) {
      info.nNewBins = ngroup;
   } else {
      info.nNewBins = nOldBins / ngroup;
      if (info.nNewBins * ngroup != nOldBins) {
         hist.Warning(where, "n%cgroup=%d is not an exact divider of n%cbins=%d.", axisName, ngroup, axisName,
                      nOldBins);
         // the top limit is truncated and the top bins move to the overflow
         newMax = oldAxis.GetBinUpEdge(info.nNewBins * ngroup);
         info.truncated = true;
      }
   }
   DefineRebinnedAxis(oldAxis, ngroup, info.nNewBins, userBins, oldAxis.GetXmin(), newMax, info.newAxis);
   if (userBins)
      WarnAboutMisalignedEdges(oldAxis, info.newAxis, hist, where);
   info.binMap = MakeRebinMap(oldAxis, info.newAxis);
   return true;
}

/// Warn when the range of the new axis extends beyond the old one while the
/// corresponding flow bins hold content: that content stays in the flow bins
/// and is not redistributed into the range of the new axis. The flow bins of
/// the mapped axis are addressed as flowIndex * stride + k * otherStride for
/// the nOther cells of the other axis.
inline void WarnAboutUnusedFlowContent(const TAxis &oldAxis, const RebinnedAxisInfo &info, const Double_t *userBins,
                                       char axisName, const Double_t *bins, Int_t stride, Int_t nOther,
                                       Int_t otherStride, TH1 &hist, const char *where)
{
   if (!userBins)
      return;
   auto flowContent = [&](Int_t flowIndex) {
      Double_t sum = 0.;
      for (Int_t k = 0; k < nOther; ++k)
         sum += bins[flowIndex * stride + k * otherStride];
      return sum;
   };
   if (userBins[0] < oldAxis.GetXmin() && flowContent(0) != 0.)
      hist.Warning(where, "underflow entries for %c axis will not be used when rebinning", axisName);
   if (userBins[info.nNewBins] > oldAxis.GetXmax() && flowContent(oldAxis.GetNbins() + 1) != 0.)
      hist.Warning(where, "overflow entries for %c axis will not be used when rebinning", axisName);
}

/// Apply the axes of the rebinned histogram, using explicit bin edges if any
/// of the two axes has non-uniform bins.
inline void SetRebinnedBins2D(TH1 &hnew, const TAxis &newXaxis, const TAxis &newYaxis)
{
   const Int_t nx = newXaxis.GetNbins();
   const Int_t ny = newYaxis.GetNbins();
   if (newXaxis.GetXbins()->GetSize() > 0 || newYaxis.GetXbins()->GetSize() > 0) {
      std::vector<Double_t> xEdges(nx + 1);
      std::vector<Double_t> yEdges(ny + 1);
      for (Int_t i = 0; i <= nx; ++i)
         xEdges[i] = newXaxis.GetBinUpEdge(i);
      for (Int_t i = 0; i <= ny; ++i)
         yEdges[i] = newYaxis.GetBinUpEdge(i);
      hnew.SetBins(nx, xEdges.data(), ny, yEdges.data()); // changes also errors array (if any)
   } else {
      hnew.SetBins(nx, newXaxis.GetXmin(), newXaxis.GetXmax(), ny, newYaxis.GetXmin(), newYaxis.GetXmax());
   }
}

/// Accumulate every old cell (including under- and overflow) into the new
/// cell given by the per-axis bin maps, for each (old array, new array) pair.
/// The new arrays must be zero-initialized by the caller.
inline void MergeRebinnedCells(Int_t nOldX, Int_t nOldY, Int_t nNewX, const std::vector<Int_t> &mapX,
                               const std::vector<Int_t> &mapY,
                               std::initializer_list<std::pair<const Double_t *, Double_t *>> arrays)
{
   for (Int_t oy = 0; oy < nOldY + 2; ++oy) {
      for (Int_t ox = 0; ox < nOldX + 2; ++ox) {
         const Int_t oldBin = ox + (nOldX + 2) * oy;
         const Int_t newBin = mapX[ox] + (nNewX + 2) * mapY[oy];
         for (auto const &arr : arrays)
            arr.second[newBin] += arr.first[oldBin];
      }
   }
}

} // namespace Internal
} // namespace ROOT

#endif
