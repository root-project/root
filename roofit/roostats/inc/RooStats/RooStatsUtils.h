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

#ifndef ROOT_Math_DistFuncMathCore
#include"Math/DistFuncMathCore.h"
#endif

#include "RooArgSet.h"
#include "RooRealVar.h"
#include "RooAbsCollection.h"
#include "TIterator.h"

#include <iostream>
using namespace std ;

namespace RooStats {

  // returns one-sided significance corresponding to a p-value
  inline Double_t PValueToSignificance(Double_t pvalue){
     return ::ROOT::Math::normal_quantile_c(pvalue,1); 
  }

  // returns p-value corresponding to a 1-sided significance
  inline Double_t SignificanceToPValue(Double_t Z){
    return ::ROOT::Math::normal_cdf_c(Z);
  }


  inline void SetParameters(const RooArgSet* desiredVals, RooArgSet* paramsToChange){
    *paramsToChange=*desiredVals ;
  }

  inline void RemoveConstantParameters(RooArgSet* set){
    RooArgSet constSet;
    RooLinkedListIter it = set->iterator();
    RooRealVar *myarg; 
    while ((myarg = (RooRealVar *)it.Next())) { 
      if(myarg->isConstant()) constSet.add(*myarg);
    }
    set->remove(constSet);
  }

  // assuming all values in set are RooRealVars, randomize their values
  inline void RandomizeCollection(RooAbsCollection& set,
                                  Bool_t randomizeConstants = kTRUE)
  {
    RooLinkedListIter it = set.iterator();
    RooRealVar* var;

    // repeat loop tpo avoid calling isConstant for nothing 
    if (randomizeConstants) { 
       while ((var = (RooRealVar*)it.Next()) != NULL)
         var->randomize();
    }
    else {
       // exclude constants variables
      while ((var = (RooRealVar*)it.Next()) != NULL)
      if (!var->isConstant() )
         var->randomize();
    }


  }

}


#endif
