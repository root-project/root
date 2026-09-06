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
/// the new axis range are mapped to the new under-/overflow. If checkEdges is
/// true (variable rebinning with user-provided edges), emit a warning through
/// hist when a new bin edge does not line up with an old bin edge, because
/// the entries of the bin that is split cannot be distributed correctly.
inline std::vector<Int_t>
MakeRebinMap(const TAxis &oldAxis, const TAxis &newAxis, bool checkEdges, TH1 &hist, const char *where)
{
   const Int_t nOld = oldAxis.GetNbins();
   const Int_t nNew = newAxis.GetNbins();
   std::vector<Int_t> map(nOld + 2);
   map[0] = 0;
   map[nOld + 1] = nNew + 1;
   Int_t prev = -1;
   for (Int_t o = 1; o <= nOld; ++o) {
      const Int_t b = newAxis.FindFixBin(oldAxis.GetBinCenter(o));
      if (checkEdges && b != prev && b >= 1 && b <= nNew &&
          !TMath::AreEqualAbs(oldAxis.GetBinLowEdge(o), newAxis.GetBinLowEdge(b),
                              TMath::Max(1.E-8 * oldAxis.GetBinWidth(o), 1.E-16))) {
         hist.Warning(where,
                      "Bin edge %d of rebinned histogram does not match any bin edges of the old histogram. "
                      "Result can be inconsistent",
                      b);
      }
      map[o] = b;
      prev = b;
   }
   return map;
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
