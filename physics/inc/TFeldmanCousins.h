// @(#)root/physics:$Name:$:$Id:$
// Author: Adrian Bevan   10/02/2002

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TFeldmanCousins
#define ROOT_TFeldmanCousins

////////////////////////////////////////////////////////////////////////////
// TFeldmanCousins
//
// Class to calculate the CL upper limit using
// the Feldman-Cousins method as described in PRD V57 #7, p3873-3889
//
// The default confidence interval calculated using this method is 90%
// This is set either by having a default the constructor, or using the
// appropriate fraction when instantiating an object of this class (e.g. 0.9)
//
// The simple extension to a gaussian resolution function bounded at zero
// has not been addressed as yet -> `time is of the essence' as they write
// on the wall of the maze in that classic game ...
//
// Author: Adrian Bevan, Liverpool University
//
// Copyright Liverpool University 2001       bevan@slac.stanford.edu
///////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TObject
#include "TObject.h"
#endif
#ifndef ROOT_TMath
#include "TMath.h"
#endif
#ifndef ROOT_TString
#include "TString.h"
#endif


////////////////////////////////////////
//    VARIABLES THAT CAN BE ALTERED   //
//    -----------------------------   //
// depending on your desired precision//
////////////////////////////////////////
const Int_t    NMAX     = 50;         //maximum number of observed events
const Double_t MUSTEP   = 0.005;      //step size to scan in mu

///////////////////////////////////////////
// Fixed variables and class declaration //
///////////////////////////////////////////
const Double_t MUMIN    = 0.0;
const Double_t MUMAX    = (Double_t)NMAX;
const Int_t    nNuSteps = (Int_t)((MUMAX - MUMIN)/MUSTEP);

class TFeldmanCousins : public TObject {

private:
  Double_t fCL;
  Double_t fUpperLimit;
  Double_t fLowerLimit;
  Double_t fNobserved;
  Double_t fNbackground;

  Int_t fQUICK;
  ////////////////////////////////////////////////
  // calculate the poissonian probability for   //
  // a mean of mu+B events with a variance of N //
  ////////////////////////////////////////////////
  Double_t Prob(Int_t N, Double_t mu, Double_t B);

  ////////////////////////////////////////////////
  // calculate the probability table and see if //
  // fNObserved is in the 100.0 * fCL %         //
  // interval                                   //
  ////////////////////////////////////////////////
  Int_t FindLimitsFromTable(Double_t mu);

public:
  TFeldmanCousins(TString options = "");
  TFeldmanCousins(Double_t newCL, TString options = "");
  virtual ~TFeldmanCousins();

  ////////////////////////////////////////////////
  // calculate the upper limit given Nobserved  //
  // and Nbackground events                     //
  // the variables fUpperLimit and fLowerLimit  //
  // are set before returning the upper limit   //
  ////////////////////////////////////////////////
  Double_t CalculateUpperLimit(Double_t Nobserved, Double_t Nbackground);
  Double_t CalculateLowerLimit(Double_t Nobserved, Double_t Nbackground);

  inline Double_t GetUpperLimit(void) { return fUpperLimit;  }
  inline Double_t GetLowerLimit(void) { return fLowerLimit;  }
  inline Double_t GetNobserved(void)  { return fNobserved;   }
  inline Double_t GetNbackground(void){ return fNbackground; }
  inline Double_t GetCL(void)         { return fCL;          }

  inline void SetNobserved(Double_t NObs)  { fNobserved   = NObs;  }
  inline void SetNbackground(Double_t Nbg) { fNbackground = Nbg;   }
  inline void SetCL(Double_t newCL)        { fCL          = newCL; }

  ClassDef(TFeldmanCousins,1)  // Calculate the confidence level using the Feldman-Cousins method
};

#endif
