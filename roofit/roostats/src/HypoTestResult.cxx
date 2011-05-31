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
   fPValueIsRightTail(kTRUE),
   fBackgroundIsAlt(kFALSE)
{
   // Default constructor
}


//____________________________________________________________________
HypoTestResult::HypoTestResult(const char* name, Double_t nullp, Double_t altp) :
   TNamed(name,name),
   fNullPValue(nullp), fAlternatePValue(altp),
   fTestStatisticData(NaN),
   fNullDistr(NULL), fAltDistr(NULL),
   fPValueIsRightTail(kTRUE),
   fBackgroundIsAlt(kFALSE)
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

   UpdatePValue(fNullDistr, fNullPValue, fNullPValueError, fPValueIsRightTail);
   UpdatePValue(fAltDistr, fAlternatePValue, fAlternatePValueError, !fPValueIsRightTail);
}


//____________________________________________________________________
void HypoTestResult::SetAltDistribution(SamplingDistribution *alt) {
   fAltDistr = alt;
   UpdatePValue(fAltDistr, fAlternatePValue, fAlternatePValueError, !fPValueIsRightTail);
}
//____________________________________________________________________
void HypoTestResult::SetNullDistribution(SamplingDistribution *null) {
   fNullDistr = null;
   UpdatePValue(fNullDistr, fNullPValue, fNullPValueError, fPValueIsRightTail);
}
//____________________________________________________________________
void HypoTestResult::SetTestStatisticData(const Double_t tsd) {
   fTestStatisticData = tsd;

   UpdatePValue(fNullDistr, fNullPValue, fNullPValueError, fPValueIsRightTail);
   UpdatePValue(fAltDistr, fAlternatePValue, fAlternatePValueError, !fPValueIsRightTail);
}
//____________________________________________________________________
void HypoTestResult::SetPValueIsRightTail(Bool_t pr) {
   fPValueIsRightTail = pr;

   UpdatePValue(fNullDistr, fNullPValue, fNullPValueError, fPValueIsRightTail);
   UpdatePValue(fAltDistr, fAlternatePValue, fAlternatePValueError, !fPValueIsRightTail);
}

//____________________________________________________________________
Bool_t HypoTestResult::HasTestStatisticData(void) const {
   return !IsNaN(fTestStatisticData);
}

Double_t HypoTestResult::NullPValueError() const {
   // compute error on Null pvalue 
   return fNullPValueError; 
}

//____________________________________________________________________
Double_t HypoTestResult::CLbError() const {
   // compute CLb error
   // Clb =  1 - NullPValue() 
   // must use opposite condition that routine above
   return fBackgroundIsAlt ? fAlternatePValueError : fNullPValueError;
}

//____________________________________________________________________
Double_t HypoTestResult::CLsplusbError() const {
   return fBackgroundIsAlt ? fNullPValueError : fAlternatePValueError;
}


//____________________________________________________________________
Double_t HypoTestResult::CLsError() const {
   // Returns an estimate of the error on CLs through combination of the
   // errors on CLb and CLsplusb:
   // BEGIN_LATEX
   // #sigma_{CL_s} = CL_s
   // #sqrt{#left( #frac{#sigma_{CL_{s+b}}}{CL_{s+b}} #right)^2 + #left( #frac{#sigma_{CL_{b}}}{CL_{b}} #right)^2}
   // END_LATEX

   if(!fAltDistr || !fNullDistr) return 0.0;

   // unsigned const int n_b = fNullDistr->GetSamplingDistribution().size();
   // unsigned const int n_sb = fAltDistr->GetSamplingDistribution().size();

   if (CLb() == 0 ) return numeric_limits<double>::infinity();

   double cl_b_err2 = pow(CLbError(),2);
   double cl_sb_err2 = pow(CLsplusbError(),2);

   return TMath::Sqrt(cl_sb_err2 + cl_b_err2 * pow(CLs(),2))/CLb();
}



// private
//____________________________________________________________________
void HypoTestResult::UpdatePValue(const SamplingDistribution* distr, Double_t &pvalue, Double_t &perror, Bool_t pIsRightTail) {
   // updates the pvalue if sufficient data is available

   if(IsNaN(fTestStatisticData)) return;

   /* Got to be careful for discrete distributions:
    * To get the right behaviour for limits, the p-value for the Null must not
    * include the value of fTestStatistic (ie the value is in the confidence level),
    * but for the Alt, it must be included.
    *
    * Technical note: to find out, whether we are doing the calc for Null or Alt,
    * we compare the pIsRightTail given to this function to fPValueIsRightTail which
    * always refers to the Null. Therefore, if they are equal it is the Null, if not
    * it is the Alt.
    */
   if(distr) {
      if(pIsRightTail) {
         pvalue = distr->IntegralAndError(perror, fTestStatisticData, RooNumber::infinity(), kTRUE,
            pIsRightTail == fPValueIsRightTail ? kFALSE : kTRUE,     // lowClosed
            kTRUE       // highClosed
         ); // [or( fTestStatistic, inf ]
      }else{
         pvalue = distr->IntegralAndError(perror, -RooNumber::infinity(), fTestStatisticData, kTRUE,
            kTRUE,      // lowClosed
            pIsRightTail == fPValueIsRightTail ? kFALSE : kTRUE      // highClosed
         ); // [ -inf, fTestStatistic )or]
      }
   }
}

