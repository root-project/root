/// \file RAxis.cxx
/// \ingroup Hist ROOT7
/// \author Axel Naumann <axel@cern.ch>
/// \date 2015-08-06
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2015, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/RAxis.hxx"

#include <cmath>
#include <limits>

constexpr const int ROOT::Experimental::RAxisBase::kNOverflowBins[4];

int ROOT::Experimental::RAxisEquidistant::GetBinIndexForLowEdge(double x) const noexcept
{
   // fracBinIdx is the fractional bin index of x in this axis. It's (close to)
   // an integer if it's an axis border.
   double fracBinIdx = *begin() + (x - GetMinimum()) * fInvBinWidth;

   // fracBinIdx might be 12.99999999. It's a bin border if the deviation from
   // an actual bin border is "fairly small".
   int binIdx = std::round(fracBinIdx);
   double binOffset = fracBinIdx - binIdx;
   if (std::fabs(binOffset) > 10 * std::numeric_limits<double>::epsilon())
      return -1;

   // If the bin index is below the first bin (i.e. x is the lower edge of the
   // underflow bin) then it's out of range.
   if (IsUnderflowBin(binIdx))
      return -1;
   // If x is the lower edge of the overflow bin then that's still okay - but if
   // even the bin before binIdx is an overflow it's out of range.
   if (IsOverflowBin(binIdx - 1))
      return -1;

   return binIdx;
}

ROOT::Experimental::EAxisCompatibility ROOT::Experimental::CanMap(RAxisEquidistant &target,
                                                                  RAxisEquidistant &source) noexcept
{
   if (source == target)
      return EAxisCompatibility::kIdentical;

   int idxTargetLow = target.GetBinIndexForLowEdge(source.GetMinimum());
   int idxTargetHigh = target.GetBinIndexForLowEdge(source.GetMaximum());
   if (idxTargetLow < 0 || idxTargetHigh < 0)
      return EAxisCompatibility::kIncompatible;

   // If both low and high exist in both axes and the bin width is identical then
   // one axis contains the other.
   if (source.GetInverseBinWidth() == target.GetInverseBinWidth())
      return EAxisCompatibility::kContains;

   // Now we are left with the case
   //   source: 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6
   //   target: ...0.0, 0.3, 0.6...
   // The question is: is the ratio of the bin width identical to the ratio of
   // the number of bin?
   if (std::fabs(target.GetInverseBinWidth() * source.GetNBinsNoOver() -
                 source.GetInverseBinWidth() * (idxTargetHigh - idxTargetLow)) > 1E-6 * target.GetInverseBinWidth())
      return EAxisCompatibility::kIncompatible;

   // source is a fine-grained version of target.
   return EAxisCompatibility::kSampling;
}

// RODO: the other CanMap() overloads
