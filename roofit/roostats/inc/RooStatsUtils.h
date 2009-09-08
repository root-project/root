// @(#)root/roostats:$Id$
// Author: Kyle Cranmer   28/07/2008

/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef RooStats_RooStatsUtils
#define RooStats_RooStatsUtils

#ifndef ROOT_TMath
#include "TMath.h"
#endif

#ifndef ROOT_Math_DistFunc
#include"Math/DistFunc.h"
#endif

#include "RooArgSet.h"
#include "RooRealVar.h"

namespace RooStats {

  // returns one-sided significance corresponding to a p-value
  inline Double_t PValueToSignificance(Double_t pvalue){
    // return sqrt(2.)*TMath::ErfInverse(1 - 2.*pvalue);
     return TMath::Abs(::ROOT::Math::normal_quantile(pvalue,1) ); 
  }

  // returns p-value corresponding to a 1-sided significance
  inline Double_t SignificanceToPValue(Double_t Z){
    //    return .5*TMath::Erfc( Z /sqrt(2.));
    return ::ROOT::Math::normal_cdf_c(Z);
  }


  inline void SetParameters(const RooArgSet* desiredVals, RooArgSet* paramsToChange){
     *paramsToChange=*desiredVals;
  }

  inline void RemoveConstantParameters(RooArgSet* set){
    RooArgSet constSet;
    TIter it = set->createIterator();
    RooRealVar *myarg; 
    while ((myarg = (RooRealVar *)it.Next())) { 
      if(!myarg) continue;
      if(myarg->isConstant()) constSet.add(*myarg);
    }
    set->remove(constSet);
  }


}

#endif
