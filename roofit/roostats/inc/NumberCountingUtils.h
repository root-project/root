// @(#)root/roostats:$Id$
// Author: Kyle Cranmer   28/07/2008

/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_NumberCountingUtils
#define ROOT_NumberCountingUtils

#ifndef ROOT_TMath
#include "TMath.h"
#endif

namespace Statistics {
  inline Double_t PValueToSignificance(Double_t pvalue){
//    return sqrt(2.)*TMath::ErfInverse(1 - 2.*pvalue);
    return TMath::Abs(TMath::NormQuantile(pvalue) ); 
  }
}

/////////////////////////////////////////
// NumberCountingUtils
//
// Encapsulates common number counting utilities
/////////////////////////////////////////

namespace  NumberCountingUtils {

  ///////////////////////////////////
  // Standalone Functions.
  // Naming conventions:
  //  Exp = Expected
  //  Obs = Observed
  //  P   = p-value
  //  Z   = Z-value or significance in Sigma (one-sided convention)
  //////////////////////////////////

  
  Double_t BinomialExpZ(Double_t, Double_t, Double_t);
  Double_t BinomialWithTauExpZ(Double_t, Double_t, Double_t);   
  Double_t BinomialObsZ(Double_t, Double_t, Double_t);
  Double_t BinomialWithTauObsZ(Double_t, Double_t, Double_t);

  Double_t BinomialExpP(Double_t, Double_t, Double_t);
  Double_t BinomialWithTauExpP(Double_t, Double_t, Double_t);
  Double_t BinomialObsP(Double_t, Double_t, Double_t);
  Double_t BinomialWithTauObsP(Double_t, Double_t, Double_t);

  ///////////////////////////////////
  // RooFit based Functions
  //////////////////////////////////
  Double_t ProfileCombinationExpZ(Double_t*, Double_t*, Double_t*, Int_t );

}

#endif
