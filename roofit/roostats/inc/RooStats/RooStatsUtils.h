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

#include "TMath.h"

#include "TTree.h"

#include "Math/DistFuncMathCore.h"

#include "RooArgSet.h"
#include "RooRealVar.h"
#include "RooAbsCollection.h"
#include "RooStats/ModelConfig.h"
#include "RooProdPdf.h"
#include "RooDataSet.h"
#include "RooAbsCategoryLValue.h"


/** \namespace RooStats
    \ingroup Roostats

Namespace for the RooStats classes

All the classes of the %RooStats package are in the RooStats namespace.
In addition the namespace contain a set of utility functions.

*/

namespace RooStats {
   struct RooStatsConfig {
      bool useLikelihoodOffset{false}; ///< Offset the likelihood by passing RooFit::Offset to fitTo().
      bool useEvalErrorWall{true};     ///< Use the error wall RooFit::EvalErrorWall to drive the fitter away from disallowed parameter values.
   };

   /// Retrieve the config object which can be used to set flags for things like offsetting the likelihood
   /// or using the error wall for the minimiser.
   RooStatsConfig& GetGlobalRooStatsConfig();

   /// returns one-sided significance corresponding to a p-value
   inline double PValueToSignificance(double pvalue){
      return ::ROOT::Math::normal_quantile_c(pvalue,1);
   }

   /// returns p-value corresponding to a 1-sided significance
   inline double SignificanceToPValue(double Z){
      return ::ROOT::Math::normal_cdf_c(Z);
   }

   /// Compute the Asimov Median significance for a Poisson process
   /// with s = expected number of signal events, b = expected number of background events
   /// and optionally sigma_b = expected uncertainty of backgorund events
   double AsimovSignificance(double s, double b, double sigma_b = 0.0 );

   inline void SetParameters(const RooArgSet* desiredVals, RooArgSet* paramsToChange){
      paramsToChange->assign(*desiredVals) ;
   }

   inline void RemoveConstantParameters(RooArgSet* set){
      RooArgSet constSet;
      for (auto const *myarg : static_range_cast<RooRealVar *>(*set)) {
         if(myarg->isConstant()) constSet.add(*myarg);
      }
      set->remove(constSet);
   }

   inline void RemoveConstantParameters(RooArgList& set){
      RooArgSet constSet;
      for (auto const *myarg : static_range_cast<RooRealVar *>(set)) {
         if(myarg->isConstant()) constSet.add(*myarg);
      }
      set.remove(constSet);
   }

   /// utility function to set all variable constant in a collection
   /// (from G. Petrucciani)
   inline bool SetAllConstant(const RooAbsCollection &coll, bool constant = true) {
      bool changed = false; 
      for (auto *v : dynamic_range_cast<RooRealVar *>(coll)) {
         if (v && (v->isConstant() != constant)) {
            changed = true;
            v->setConstant(constant);
         }
      }
      return changed;
   }


   /// assuming all values in set are RooRealVars, randomize their values
   inline void RandomizeCollection(RooAbsCollection& set,
                                   bool randomizeConstants = true)
   {

      // repeat loop to avoid calling isConstant for nothing
      if (randomizeConstants) {
         for (auto *var : static_range_cast<RooRealVar *>(set))
            var->randomize();
      }
      else {
         // exclude constants variables
         for (auto *var : static_range_cast<RooRealVar *>(set))
            if (!var->isConstant() )
               var->randomize();
      }

   }

   void FactorizePdf(const RooArgSet &observables, RooAbsPdf &pdf, RooArgList &obsTerms, RooArgList &constraints);

   void FactorizePdf(RooStats::ModelConfig &model, RooAbsPdf &pdf, RooArgList &obsTerms, RooArgList &constraints);

   /// extract constraint terms from pdf
   RooAbsPdf * MakeNuisancePdf(RooAbsPdf &pdf, const RooArgSet &observables, const char *name);
   RooAbsPdf * MakeNuisancePdf(const RooStats::ModelConfig &model, const char *name);
   /// remove constraints from pdf and return the unconstrained pdf
   RooAbsPdf * MakeUnconstrainedPdf(RooAbsPdf &pdf, const RooArgSet &observables, const char *name = nullptr);
   RooAbsPdf * MakeUnconstrainedPdf(const RooStats::ModelConfig &model, const char *name = nullptr);

   /// Create a TTree with the given name and description. All RooRealVars in the RooDataSet are represented as branches that contain values of type double.
   TTree* GetAsTTree(TString name, TString desc, const RooDataSet& data);

   /// useful function to print in one line the content of a set with their values
   void PrintListContent(const RooArgList & l, std::ostream & os = std::cout);

   /// function to set a global flag in RooStats to use NLL offset when performing nll computations
   /// Note that not all ROoStats tools implement this capabilities
   void UseNLLOffset(bool on);

   /// function returning if the flag to check if the flag to use  NLLOffset is set
   bool IsNLLOffset();

   /// function that clones a workspace, copying all needed components and discarding all others
   RooWorkspace* MakeCleanWorkspace(RooWorkspace *oldWS, const char *newName, bool copySnapshots,
                                    const char *mcname, const char *newmcname);

}


#endif
