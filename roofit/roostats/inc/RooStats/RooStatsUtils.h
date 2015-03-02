// @(#)root/roostats:$Id$
// Author: Kyle Cranmer   28/07/2008

/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOSTATS_RooStatsUtils
#define ROOSTATS_RooStatsUtils

#ifndef ROOT_TMath
#include "TMath.h"
#endif

#ifndef ROOT_TTree
#include "TTree.h"
#endif

#ifndef ROOT_Math_DistFuncMathCore
#include"Math/DistFuncMathCore.h"
#endif

#include "RooArgSet.h"
#include "RooRealVar.h"
#include "RooAbsCollection.h"
#include "TIterator.h"
#include "RooStats/ModelConfig.h"
#include "RooProdPdf.h"
#include "RooDataSet.h"


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

  inline void RemoveConstantParameters(RooArgList& set){
    RooArgSet constSet;
    RooLinkedListIter it = set.iterator();
    RooRealVar *myarg; 
    while ((myarg = (RooRealVar *)it.Next())) { 
      if(myarg->isConstant()) constSet.add(*myarg);
    }
    set.remove(constSet);
  }

  inline bool SetAllConstant(const RooAbsCollection &coll, bool constant = true) {
       // utility function to set all variable constant in a collection
       // (from G. Petrucciani)
       bool changed = false;
       RooLinkedListIter iter = coll.iterator();
       for (RooAbsArg *a = (RooAbsArg *) iter.Next(); a != 0; a = (RooAbsArg *) iter.Next()) {
          RooRealVar *v = dynamic_cast<RooRealVar *>(a);
          if (v && (v->isConstant() != constant)) {
             changed = true;
             v->setConstant(constant);
          }
       }
       return changed;
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

   void FactorizePdf(const RooArgSet &observables, RooAbsPdf &pdf, RooArgList &obsTerms, RooArgList &constraints);

   void FactorizePdf(RooStats::ModelConfig &model, RooAbsPdf &pdf, RooArgList &obsTerms, RooArgList &constraints);

   // extract constraint terms from pdf
   RooAbsPdf * MakeNuisancePdf(RooAbsPdf &pdf, const RooArgSet &observables, const char *name);
   RooAbsPdf * MakeNuisancePdf(const RooStats::ModelConfig &model, const char *name);
   // remove constraints from pdf and return the unconstrained pdf
   RooAbsPdf * MakeUnconstrainedPdf(RooAbsPdf &pdf, const RooArgSet &observables, const char *name = NULL);
   RooAbsPdf * MakeUnconstrainedPdf(const RooStats::ModelConfig &model, const char *name = NULL);
   
   // Create a TTree with the given name and description. All RooRealVars in the RooDataSet are represented as branches that contain values of type Double_t.
   TTree* GetAsTTree(TString name, TString desc, const RooDataSet& data);

   // useful function to print in one line the content of a set with their values 
   void PrintListContent(const RooArgList & l, std::ostream & os = std::cout);

   // function to set a global flag in RooStats to use NLL offset when performing nll computations
   // Note that not all ROoStats tools implement this capabilities
   void UseNLLOffset(bool on);

   // function returning if the flag to check if the flag to use  NLLOffset is set
   bool IsNLLOffset();
   


}


#endif
