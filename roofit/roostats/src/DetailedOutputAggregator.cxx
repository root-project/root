// @(#)root/roostats:$Id$
// Author: Sven Kreiss, Kyle Cranmer, Lorenzo Moneta  Nov 2010
/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

// implementation file of DetailedOutputAggregator

#include <limits>


#include "RooFitResult.h"
#include "RooPullVar.h"
#include "RooRealVar.h"
#include "RooDataSet.h"

#include "RooStats/DetailedOutputAggregator.h"

namespace RooStats {

   DetailedOutputAggregator::~DetailedOutputAggregator() {
      // destructor
      if (fResult != NULL) delete fResult;
      if (fBuiltSet != NULL) delete fBuiltSet;
   }


   RooArgSet * DetailedOutputAggregator::GetAsArgSet(RooFitResult *result, TString prefix, bool withErrorsAndPulls) {
      // static function to translate the given fit result to a RooArgSet in a generic way.
      // Prefix is prepended to all variable names.
      // LM: caller is responsible to delete the returned list and eventually also the content of the list
      //    Note that the returned list is not owning the returned content
      RooArgSet *detailedOutput = new RooArgSet;
      const RooArgList &detOut = result->floatParsFinal();
      const RooArgList &truthSet = result->floatParsInit();
      TIterator *it = detOut.createIterator();
      while(RooAbsArg* v = dynamic_cast<RooAbsArg*>(it->Next())) {
         RooAbsArg* clone = v->cloneTree(TString().Append(prefix).Append(v->GetName()));
         clone->SetTitle( TString().Append(prefix).Append(v->GetTitle()) );
         RooRealVar* var = dynamic_cast<RooRealVar*>(v);
         if (var) clone->setAttribute("StoreError");
         detailedOutput->add(*clone);
         
         if( withErrorsAndPulls && var ) {
            clone->setAttribute("StoreAsymError");

            TString pullname = TString().Append(prefix).Append(TString::Format("%s_pull", var->GetName()));
            //             TString pulldesc = TString::Format("%s pull for fit %u", var->GetTitle(), fitNumber);
            RooRealVar* truth = dynamic_cast<RooRealVar*>(truthSet.find(var->GetName()));
            RooPullVar pulltemp("temppull", "temppull", *var, *truth);
            RooRealVar* pull = new RooRealVar(pullname, pullname, pulltemp.getVal());
            detailedOutput->add(*pull);
         }
      }
      delete it;

      // monitor a few more variables
      detailedOutput->add( *new RooRealVar(TString().Append(prefix).Append("minNLL"), TString().Append(prefix).Append("minNLL"), result->minNll() ) );
      detailedOutput->add( *new RooRealVar(TString().Append(prefix).Append("fitStatus"), TString().Append(prefix).Append("fitStatus"), result->status() ) );
      detailedOutput->add( *new RooRealVar(TString().Append(prefix).Append("covQual"), TString().Append(prefix).Append("covQual"), result->covQual() ) );
      detailedOutput->add( *new RooRealVar(TString().Append(prefix).Append("numInvalidNLLEval"), TString().Append(prefix).Append("numInvalidNLLEval"), result->numInvalidNLL() ) );
      return detailedOutput;
   }

   void DetailedOutputAggregator::AppendArgSet(const RooAbsCollection *aset, TString prefix) {
      // For each variable in aset, prepend prefix to its name and add
      // to the internal store. Note this will not appear in the produced
      // dataset unless CommitSet is called.

      if (aset == NULL) {
         // silently ignore
         //std::cout << "Attempted to append NULL" << endl;
         return;
      }
      if (fBuiltSet == NULL) {
         fBuiltSet = new RooArgList();
      }
      TIterator* iter = aset->createIterator();
      while(RooAbsArg* v = dynamic_cast<RooAbsArg*>( iter->Next() ) ) {
         TString renamed(TString::Format("%s%s", prefix.Data(), v->GetName()));
         if (fResult == NULL) {
            // we never commited, so by default all columns are expected to not exist
            RooAbsArg* var = v->createFundamental();
            assert(var != NULL);
            (RooArgSet(*var)) = RooArgSet(*v);
            var->SetName(renamed);
            if (RooRealVar* rvar= dynamic_cast<RooRealVar*>(var)) {
               if (v->getAttribute("StoreError"))     var->setAttribute("StoreError");
               else rvar->removeError();
               if (v->getAttribute("StoreAsymError")) var->setAttribute("StoreAsymError");
               else rvar->removeAsymError();
            }
            if (fBuiltSet->addOwned(*var)) continue;  // OK - can skip past setting value
         }
         if (RooAbsArg* var = fBuiltSet->find(renamed)) {
            // we already commited an argset once, so we expect all columns to already be in the set
            var->SetName(v->GetName());
            (RooArgSet(*var)) = RooArgSet(*v); // copy values and errors
            var->SetName(renamed);
         }
      }
      delete iter;
   }

   // Commit to the result RooDataSet.
   void DetailedOutputAggregator::CommitSet(double weight) {
      if (fResult == NULL) {
         // Store dataset as a tree - problem with VectorStore and StoreError (bug #94908)
         RooRealVar wgt("weight","weight",1.0);
         fResult = new RooDataSet("", "", RooArgSet(*fBuiltSet,wgt), RooFit::WeightVar(wgt));
      }
      fResult->add(RooArgSet(*fBuiltSet), weight);
      TIterator* iter = fBuiltSet->createIterator();
      while(RooAbsArg* v = dynamic_cast<RooAbsArg*>( iter->Next() ) ) {
         if (RooRealVar* var= dynamic_cast<RooRealVar*>(v)) {
            // Invalidate values in case we don't set some of them next time round (eg. if fit not done)
            var->setVal(std::numeric_limits<Double_t>::quiet_NaN());
            var->removeError();
            var->removeAsymError();
         }
      }
      delete iter;
   }


   RooDataSet * DetailedOutputAggregator::GetAsDataSet(TString name, TString title) {
      // Returns all detailed output as a dataset.
      // Ownership of the dataset is transferred to the caller.
      RooDataSet* temp = NULL;
      if( fResult ) {
         temp = fResult;
         fResult = NULL;   // we no longer own the dataset
         temp->SetNameTitle( name.Data(), title.Data() );
      }else{
         RooRealVar wgt("weight","weight",1.0);
         temp = new RooDataSet(name.Data(), title.Data(), RooArgSet(wgt), RooFit::WeightVar(wgt));
      }
      delete fBuiltSet;
      fBuiltSet = NULL;

      return temp;
   }


}  // end namespace RooStats

