// @(#)root/roostats:$Id$
// Author: Kyle Cranmer, Lorenzo Moneta, Gregory Schott, Wouter Verkerke, Sven Kreiss
/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/*****************************************************************************
 * Project: RooStats
 * Package: RooFit/RooStats  
 * @(#)root/roofit/roostats:$Id$
 * Authors:                     
 *   Kyle Cranmer, Lorenzo Moneta, Gregory Schott, Wouter Verkerke, Sven Kreiss
 *
 *****************************************************************************/



//_________________________________________________
/*
BEGIN_HTML
<p>
HypoTestResult is a base class for results from hypothesis tests.
Any tool inheriting from HypoTestCalculator can return a HypoTestResult.
As such, it stores a p-value for the null-hypothesis (eg. background-only) 
and an alternate hypothesis (eg. signal+background).  
The p-values can also be transformed into confidence levels (CLb, CLsplusb) in a trivial way.
The ratio of the CLsplusb to CLb is often called CLs, and is considered useful, though it is 
not a probability.
Finally, the p-value of the null can be transformed into a number of equivalent Gaussian sigma using the 
Significance method.
END_HTML
*/
//

#include "RooStats/HypoTestResult.h"
#include "RooAbsReal.h"

#ifndef RooStats_RooStatsUtils
#include "RooStats/RooStatsUtils.h"
#endif

#include <limits>
#define NaN numeric_limits<float>::quiet_NaN()
#define IsNaN(a) isnan(a)

ClassImp(RooStats::HypoTestResult) ;

using namespace RooStats;

//____________________________________________________________________
HypoTestResult::HypoTestResult(const char* name) : 
   TNamed(name,name),
   fNullPValue(NaN), fAlternatePValue(NaN),
   fTestStatisticData(NaN),
   fNullDistr(NULL), fAltDistr(NULL),
   fPValueIsRightTail(kTRUE)
{
   // Default constructor
}


//____________________________________________________________________
HypoTestResult::HypoTestResult(const char* name, Double_t nullp, Double_t altp) :
   TNamed(name,name),
   fNullPValue(nullp), fAlternatePValue(altp),
   fTestStatisticData(NaN),
   fNullDistr(NULL), fAltDistr(NULL),
   fPValueIsRightTail(kTRUE)
{
   // Alternate constructor
}


//____________________________________________________________________
HypoTestResult::~HypoTestResult()
{
   // Destructor

}


void HypoTestResult::Append(const HypoTestResult* other) {
   // Add additional toy-MC experiments to the current results.
   // Use the data test statistics of the added object if it is not already
   // set (otherwise, ignore the new one).

   if(fNullDistr)
      fNullDistr->Add(other->GetNullDistribution());
   else
      fNullDistr = other->GetNullDistribution();

   if(fAltDistr)
      fAltDistr->Add(other->GetAltDistribution());
   else
      fAltDistr = other->GetAltDistribution();

   // if no data is present use the other HypoTestResult's data
   if(IsNaN(fTestStatisticData)) fTestStatisticData = other->GetTestStatisticData();

   UpdatePValue(fNullDistr, &fNullPValue, fPValueIsRightTail);
   UpdatePValue(fAltDistr, &fAlternatePValue, !fPValueIsRightTail);
}


//____________________________________________________________________
void HypoTestResult::SetAltDistribution(SamplingDistribution *alt) {
   fAltDistr = alt;
   UpdatePValue(fAltDistr, &fAlternatePValue, !fPValueIsRightTail);
}
//____________________________________________________________________
void HypoTestResult::SetNullDistribution(SamplingDistribution *null) {
   fNullDistr = null;
   UpdatePValue(fNullDistr, &fNullPValue, fPValueIsRightTail);
}
//____________________________________________________________________
void HypoTestResult::SetTestStatisticData(const Double_t tsd) {
   fTestStatisticData = tsd;

   UpdatePValue(fNullDistr, &fNullPValue, fPValueIsRightTail);
   UpdatePValue(fAltDistr, &fAlternatePValue, !fPValueIsRightTail);
}
//____________________________________________________________________
void HypoTestResult::SetPValueIsRightTail(Bool_t pr) {
   fPValueIsRightTail = pr;

   UpdatePValue(fNullDistr, &fNullPValue, fPValueIsRightTail);
   UpdatePValue(fAltDistr, &fAlternatePValue, !fPValueIsRightTail);
}

//____________________________________________________________________
Bool_t HypoTestResult::HasTestStatisticData(void) const {
   return !IsNaN(fTestStatisticData);
}



//____________________________________________________________________
Double_t HypoTestResult::CLbError() const {
   // Returns an estimate of the error on CLb assuming a binomial error on
   // CLb:
   // BEGIN_LATEX
   // #sigma_{CL_{b}} = #sqrt{CL_{b} #left( 1 - CL_{b} #right) /
   //   n_{toys}}
   // END_LATEX

   if(!fNullDistr) return 0.0;
   unsigned const int n = fNullDistr->GetSamplingDistribution().size();
   return TMath::Sqrt(CLb() * (1. - CLb()) / n);
}

//____________________________________________________________________
Double_t HypoTestResult::CLsplusbError() const {
   // Returns an estimate of the error on CLsplusb assuming a binomial
   // error on CLsplusb:
   // BEGIN_LATEX
   // #sigma_{CL_{s+b}} = #sqrt{CL_{s+b} #left( 1 - CL_{s+b} #right) /
   //   n_{toys}}
   // END_LATEX

   if(!fAltDistr) return 0.0;
   unsigned const int n = fAltDistr->GetSamplingDistribution().size();
   return TMath::Sqrt(CLsplusb() * (1. - CLsplusb()) / n);
}

//____________________________________________________________________
Double_t HypoTestResult::CLsError() const {
   // Returns an estimate of the error on CLs through combination of the
   // errors on CLb and CLsplusb:
   // BEGIN_LATEX
   // #sigma_{CL_s} = CL_s
   //   #sqrt{#left( #frac{#sigma_{CL_{s+b}}}{CL_{s+b}} #right)^2 +
   //     #left( #frac{#sigma_{CL_{b}}}{CL_{b}} #right)^2}
   // END_LATEX

   if(!fAltDistr || !fNullDistr) return 0.0;

   unsigned const int n_b = fNullDistr->GetSamplingDistribution().size();
   unsigned const int n_sb = fAltDistr->GetSamplingDistribution().size();

   if (CLb() == 0 || CLsplusb() == 0) return 0.0;

   double cl_b_err = (1. - CLb()) / (n_b * CLb());
   double cl_sb_err = (1. - CLsplusb()) / (n_sb * CLsplusb());

   return CLs() * TMath::Sqrt(cl_b_err + cl_sb_err);
}



// private
//____________________________________________________________________
void HypoTestResult::UpdatePValue(const SamplingDistribution* distr, Double_t *pvalue, Bool_t pIsRightTail) {
   // updates the pvalue if sufficient data is available

   if(IsNaN(fTestStatisticData)) return;

   if(distr) {
      if(pIsRightTail)
         *pvalue = distr->Integral(fTestStatisticData, RooNumber::infinity());
      else
         *pvalue = distr->Integral(-RooNumber::infinity(), fTestStatisticData);
   }
}

