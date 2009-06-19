// @(#)root/roostats:$Id: MCMCInterval.cxx 26805 2009-06-17 14:31:02Z kbelasco $
// Author: Kevin Belasco        17/06/2009
/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif

#include "RooStats/MCMCInterval.h"
#include "RooRealVar.h"
#include "RooDataSet.h"
#include <cstdlib>
#include <string>
#include <algorithm>
#include "TIterator.h"
#include "TH1.h"
#include "TFile.h"
#include "RooMsgService.h"

ClassImp(RooStats::MCMCInterval);

using namespace RooFit;
using namespace RooStats;
using namespace std;

MCMCInterval::MCMCInterval() : ConfInterval()
{
   fParameters = NULL;
   fConfidenceLevel = 0.0;
   fData = NULL;
   fHist = NULL;
   fCutoff = 0;
   fDimension = 1;
   fIsStrict = true;
}

MCMCInterval::MCMCInterval(const char* name) : ConfInterval(name, name)
{
   fParameters = NULL;
   fConfidenceLevel = 0.0;
   fData = NULL;
   fHist = NULL;
   fCutoff = 0;
   fDimension = 1;
   fIsStrict = true;
}

MCMCInterval::MCMCInterval(const char* name, const char* title)
   : ConfInterval(name, title)
{
   fParameters = NULL;
   fConfidenceLevel = 0.0;
   fData = NULL;
   fHist = NULL;
   fCutoff = 0;
   fDimension = 1;
}

MCMCInterval::MCMCInterval(const char* name, const char* title,
        RooArgSet& parameters, RooDataSet& chain) : ConfInterval(name, title)
{
   fParameters = &parameters;
   fConfidenceLevel = 0.0;
   fData = &chain;
   fHist = NULL;
   fCutoff = 0;
   fDimension = fParameters->getSize();
   fIsStrict = true;
}

struct CompareBins { 

   CompareBins( TH1 * hist) : fHist(hist) {}
   bool operator() ( Int_t bin1 , Int_t bin2 ) { 
      // bins must have content >= 0, so this is safe:
      Double_t n1 = fHist->GetBinContent(bin1);
      Double_t n2 = fHist->GetBinContent(bin2);
      
      return    (n1 < n2) ;
   }
   TH1 * fHist; 
};


Bool_t MCMCInterval::IsInInterval(RooArgSet& point) 
{
   Int_t bin;
   if (fDimension == 1) {
      bin = fHist->FindBin(point.getRealValue(fAxes[0]->GetName()));
   } else if (fDimension == 2) {
      bin = fHist->FindBin(point.getRealValue(fAxes[0]->GetName()),
                           point.getRealValue(fAxes[1]->GetName()));
   } else if (fDimension == 3) {
      bin = fHist->FindBin(point.getRealValue(fAxes[0]->GetName()),
                           point.getRealValue(fAxes[1]->GetName()),
                           point.getRealValue(fAxes[2]->GetName()));
   } else {
      coutE(Eval) << "* Error in MCMCInterval::DetermineInterval: " <<
                     "Couldn't handle dimension: " << fDimension << endl;
      return false;
   }

   return fHist->GetBinContent(bin) >= (Double_t)fCutoff;
}

void MCMCInterval::SetConfidenceLevel(Double_t cl)
{
   fConfidenceLevel = cl;
   DetermineInterval();
}

void MCMCInterval::DetermineInterval()
{
   TIterator* it = fParameters->createIterator();
   fAxes = new RooRealVar*[fDimension];
   Int_t n = 0;
   fNumBins = new Int_t[fDimension];

   while ((fAxes[n] = (RooRealVar*)it->Next()) != NULL) {
      fNumBins[n] = DEFAULT_NUM_BINS;
      n++;
   }

   Int_t numBins;
   if (fDimension == 1) {
      fHist = fData->createHistogram("hist", *fAxes[0],
              Binning(fNumBins[0]), Scaling(kFALSE));
      numBins = fNumBins[0];
   }
   else if (fDimension == 2) {
      fHist = fData->createHistogram("hist", *fAxes[0],
            Binning(fNumBins[0]), YVar(*fAxes[1], Binning(fNumBins[1])),
            Scaling(kFALSE));
      numBins = fNumBins[0] * fNumBins[1];
   }
   else if (fDimension == 3) {
      fHist = fData->createHistogram("hist", *fAxes[0],
              Binning(fNumBins[0]), YVar(*fAxes[1], Binning(fNumBins[1])),
              ZVar(*fAxes[2], Binning(fNumBins[2])), Scaling(kFALSE));
      numBins = fNumBins[0] * fNumBins[1] * fNumBins[2];
   }
   else {
      coutE(Eval) << "* Error in MCMCInterval::DetermineInterval: " <<
                     "Couldn't handle dimension: " << fDimension << endl;
      numBins = 0;
   }

   //TFile chainHistFile("chainHist.root", "recreate");
   //fHist->Write();
   //chainHistFile.Close();

   std::vector<Int_t> bins(numBins);
   // index 1 to numBins because TH1 uses bin 0 for underflow and 
   // bin numBins+1 for overflow
   for (Int_t ibin = 1; ibin <= numBins; ibin++)
      bins[ibin - 1] = ibin;
   std::stable_sort( bins.begin(), bins.end(), CompareBins(fHist) );
//    qsort_r(bins, (size_t)numBins, sizeof(Int_t), fHist,
//           (int (*)(void*, const void*, const void*)) CompareBins);

   Double_t nEntries = fHist->GetSumOfWeights();
   Double_t sum = 0;
   Double_t content;
   Int_t i;
   for (i = numBins - 1; i >= 0; i--) {
      content = fHist->GetBinContent(bins[i]);
      if ((sum + content) / nEntries >= fConfidenceLevel) {
         fCutoff = content;
         if (fIsStrict) {
            sum += content;
            i--;
            break;
         } else {
            i++;
            break;
         }
      }
      sum += content;
   }

   if (fIsStrict) {
      // keep going to find the sum
      for ( ; i >= 0; i--) {
         content = fHist->GetBinContent(bins[i]);
         if (content == fCutoff)
            sum += content;
         else
            break; // content must be < fCutoff
      }
   } else {
      // backtrack to find the cutoff and sum
      for ( ; i < numBins; i++) {
         content = fHist->GetBinContent(bins[i]);
         if (content > fCutoff) {
            fCutoff = content;
            break;
         } else // content == fCutoff
            sum -= content;
         if (i == numBins - 1)
            // still haven't set fCutoff correctly yet, and we have no bins
            // left, so set fCutoff to something higher than the tallest bin
            fCutoff = fHist->GetBinContent(bins[i]) + 1.0;
      }
   }

   fIntervalSum = sum;
}

// Determine the lower limit for param on this interval
Double_t MCMCInterval::LowerLimit(RooRealVar& param)
{
   for (Int_t d = 0; d < fDimension; d++) {
      if (strcmp(fAxes[d]->GetName(), param.GetName()) == 0) {
         Int_t numBins = fNumBins[d];
         for (Int_t i = 1; i <= numBins; i++)
            if (fHist->GetBinContent(i) >= fCutoff)
                return fHist->GetBinCenter(i);
      }
   }
   return param.getMin();
}

// Determine the upper limit for each param on this interval
Double_t MCMCInterval::UpperLimit(RooRealVar& param)
{
   for (Int_t d = 0; d < fDimension; d++) {
      if (strcmp(fAxes[d]->GetName(), param.GetName()) == 0) {
         Int_t numBins = fNumBins[d];
         Double_t upperLimit = param.getMin();
         for (Int_t i = 1; i <= numBins; i++)
            if (fHist->GetBinContent(i) >= fCutoff)
               upperLimit = fHist->GetBinCenter(i);
         return upperLimit;
      }
   }
   return param.getMax();
}

TH1* MCMCInterval::GetPosteriorHist() {
  
  if(fConfidenceLevel == 0) {
      coutE(Eval) << "Error in MCMCInterval::GetPosteriorHist, confidence level not set " << endl;
  }
  return (TH1*) fHist->Clone("MCMCposterior");
}

RooArgSet* MCMCInterval::GetParameters() const
{  
   // returns list of parameters
   return (RooArgSet*) fParameters->clone((std::string(fParameters->GetName())+"_clone").c_str());
}

Bool_t MCMCInterval::CheckParameters(RooArgSet& parameterPoint) const
{  
   // check that the parameters are correct

   if (parameterPoint.getSize() != fParameters->getSize() ) {
     coutE(Eval) << "MCMCInterval: size is wrong, parameters don't match" << std::endl;
     return false;
   }
   if ( ! parameterPoint.equals( *fParameters ) ) {
     coutE(Eval) << "MCMCInterval: size is ok, but parameters don't match" << std::endl;
     return false;
   }
   return true;
}
