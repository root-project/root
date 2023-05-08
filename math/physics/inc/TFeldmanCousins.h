// @(#)root/physics:$Id$
// Author: Adrian Bevan  2001

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * Copyright (C) 2001, Liverpool University.                             *
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
// class to calculate the CL upper limit using
// the Feldman-Cousins method as described in PRD V57 #7, p3873-3889
//
// The default confidence interval calvculated using this method is 90%
// This is set either by having a default the constructor, or using the
// appropriate fraction when instantiating an object of this class (e.g. 0.9)
//
// The simple extension to a gaussian resolution function bounded at zero
// has not been addressed as yet -> `time is of the essence' as they write
// on the wall of the maze in that classic game ...
//
//    VARIABLES THAT CAN BE ALTERED
//    -----------------------------
// => depending on your desired precision: The intial values of fMuMin,
// fMuMax, fMuStep and fNMax are those used in the PRD:
//   fMuMin = 0.0
//   fMuMax = 50.0
//   fMuStep= 0.005
// but there is total flexibility in changing this should you desire.
//
// Author: Adrian Bevan, Liverpool University
//
// Copyright Liverpool University 2001       bevan@slac.stanford.edu
///////////////////////////////////////////////////////////////////////////

#include "TObject.h"
#include "TString.h"

class TFeldmanCousins : public TObject {
protected:
   Double_t fCL;         // confidence level as a fraction [e.g. 90% = 0.9]
   Double_t fUpperLimit; // the calculated upper limit
   Double_t fLowerLimit; // the calculated lower limit
   Double_t fNobserved;  // input number of observed events
   Double_t fNbackground;// input number of background events
   Double_t fMuMin;      // minimum value of signal to use in calculating the tables
   Double_t fMuMax;      // maximum value of signal to use in calculating the tables
   Double_t fMuStep;     // the step in signal to use when generating tables
   Int_t    fNMuStep;    // = (int)(fMuStep)
   Int_t    fNMax;       // = (int)(fMuMax)
   Int_t    fQUICK;      // take a short cut to speed up the process of generating a
                        // lut.  This scans from Nobserved-Nbackground-fMuMin upwards
                        // assuming that UL > Nobserved-Nbackground.

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
   TFeldmanCousins(Double_t newCL=0.9, TString options = "");
   ~TFeldmanCousins() override;

   ////////////////////////////////////////////////
   // calculate the upper limit given Nobserved  //
   // and Nbackground events                     //
   // the variables fUpperLimit and fLowerLimit  //
   // are set before returning the upper limit   //
   ////////////////////////////////////////////////
   Double_t CalculateUpperLimit(Double_t Nobserved, Double_t Nbackground);
   Double_t CalculateLowerLimit(Double_t Nobserved, Double_t Nbackground);

   inline Double_t GetUpperLimit(void)  const { return fUpperLimit;  }
   inline Double_t GetLowerLimit(void)  const { return fLowerLimit;  }
   inline Double_t GetNobserved(void)   const { return fNobserved;   }
   inline Double_t GetNbackground(void) const { return fNbackground; }
   inline Double_t GetCL(void)          const { return fCL;          }

   inline Double_t GetMuMin(void)       const { return fMuMin;  }
   inline Double_t GetMuMax(void)       const { return fMuMax;  }
   inline Double_t GetMuStep(void)      const { return fMuStep; }
   inline Double_t GetNMax(void)        const { return fNMax;   }

   inline void     SetNobserved(Double_t NObs)         { fNobserved   = NObs;  }
   inline void     SetNbackground(Double_t Nbg)        { fNbackground = Nbg;   }
   inline void     SetCL(Double_t newCL)               { fCL          = newCL; }

   inline void     SetMuMin(Double_t  newMin    = 0.0)   { fMuMin = newMin;  }
   void            SetMuMax(Double_t  newMax    = 50.0);
   void            SetMuStep(Double_t newMuStep = 0.005);

   ClassDefOverride(TFeldmanCousins,1) //calculate the CL upper limit using the Feldman-Cousins method
};

#endif






