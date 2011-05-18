// @(#)root/roostats:$Id$
// Author: Sven Kreiss   June 2010
/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/**
 * Class HypoTestPlot
 * This class provides the plots for the result of a study performed with e.g. the
 * HybridCalculator2 class.
 */


#include "RooStats/HypoTestPlot.h"


ClassImp(RooStats::HypoTestPlot)

using namespace RooStats;
#include "TStyle.h"

HypoTestPlot::HypoTestPlot(HypoTestResult& result, Int_t bins) :
   SamplingDistPlot(bins),
   fHypoTestResult(&result)
{
   ApplyResult(result);
}

void HypoTestPlot::ApplyResult(HypoTestResult& result) {
   fLegend = new TLegend(0.70,0.95-0.2*0.66,0.95,0.95);

   const SamplingDistribution *alt = result.GetAltDistribution();
   const SamplingDistribution *null = result.GetNullDistribution();
   if(!result.HasTestStatisticData()) {
      if(alt) AddSamplingDistribution(alt, "NORMALIZE HIST");
      if(null) AddSamplingDistribution(null, "NORMALIZE HIST");
   }else{
      if(!result.GetPValueIsRightTail()) {
         if(alt) AddSamplingDistributionShaded(alt, result.GetTestStatisticData(), RooNumber::infinity(), "NORMALIZE HIST");
         if(null) AddSamplingDistributionShaded(null, -RooNumber::infinity(), result.GetTestStatisticData(), "NORMALIZE HIST");
      }else{
         if(alt) AddSamplingDistributionShaded(alt, -RooNumber::infinity(), result.GetTestStatisticData(), "NORMALIZE HIST");
         if(null) AddSamplingDistributionShaded(null, result.GetTestStatisticData(), RooNumber::infinity(), "NORMALIZE HIST");
      }
   }

   if(result.HasTestStatisticData()) {
      Float_t theMin(0.), theMax(0.), theYMax(0.);
      GetAbsoluteInterval(theMin, theMax, theYMax);

      AddLine(result.GetTestStatisticData(), 0, result.GetTestStatisticData(), theYMax*0.66, "test statistic data");
   }

   ApplyDefaultStyle();
}

void HypoTestPlot::ApplyDefaultStyle(void) {
   if(!fHypoTestResult) return;

   const SamplingDistribution *alt = fHypoTestResult->GetAltDistribution();
   const SamplingDistribution *null = fHypoTestResult->GetNullDistribution();

   if(alt) {
      SetLineWidth(2, alt);
      SetLineColor(kBlue, alt);
   }
   if(null) {
      SetLineWidth(2, null);
      SetLineColor(kRed, null);
   }
}
