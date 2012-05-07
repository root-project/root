// @(#)root/roostats:$Id: DetailedOutputAggregator.h 37084 2010-11-29 21:37:13Z moneta $
// Author: Sven Kreiss, Kyle Cranmer   Nov 2010
/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOSTATS_DetailedOutputAggregator
#define ROOSTATS_DetailedOutputAggregator

//_________________________________________________
/*
   BEGIN_HTML
   <p>
   This class is designed to aid in the construction of RooDataSets and RooArgSets,
   particularly those naturally arising in fitting operations.

   Typically, the usage of this class is as follows:
   <ol>
   <li> create DetailedOutputAggregator instance </li>
   <li> use AppendArgSet to add value sets to be stored as one row of the dataset </li>
   <li> call CommitSet when an entire row's worth of values has been added </li>
   <li> repeat steps 2 and 3 until all rows have been added </li>
   <li> call GetAsDataSet to extract result RooDataSet </li>
   </ol>

   </p>
   END_HTML
   */
//

#include "RooFitResult.h"
#include "RooPullVar.h"
#include "RooDataSet.h"

namespace RooStats {

	class DetailedOutputAggregator {

		public:
			// Translate the given fit result to a RooArgSet in a generic way.
			// Prefix is prepended to all variable names.
			static RooArgSet *GetAsArgSet(RooFitResult *result, TString prefix="", bool withErrorsAndPulls=false) {
				RooArgSet *detailedOutput = new RooArgSet;
				const RooArgSet &detOut = result->floatParsFinal();
				const RooArgSet &truthSet = result->floatParsInit();
				RooRealVar* var(0);
				TIterator *it = detOut.createIterator();
				for(;(var = dynamic_cast<RooRealVar*>(it->Next()));) {
					RooAbsArg* clone = var->cloneTree(TString().Append(prefix).Append(var->GetName()));
					clone->SetTitle( TString().Append(prefix).Append(var->GetTitle()) );
					detailedOutput->addOwned(*clone);

               if( withErrorsAndPulls ) {
                  TString pullname = TString().Append(prefix).Append(TString::Format("%s_pull", var->GetName()));
                  //					TString pulldesc = TString::Format("%s pull for fit %u", var->GetTitle(), fitNumber);
                  RooRealVar* truth = dynamic_cast<RooRealVar*>(truthSet.find(var->GetName()));
                  RooPullVar pulltemp("temppull", "temppull", *var, *truth);
                  RooRealVar* pull = new RooRealVar(pullname, pullname, pulltemp.getVal());
                  detailedOutput->addOwned(*pull);
   
                  TString errloname = TString().Append(prefix).Append(TString::Format("%s_errlo", var->GetName()));
                  //					TString errlodesc = TString::Format("%s low error for fit %u", var->GetTitle(), fitNumber);
                  RooRealVar* errlo = new RooRealVar(errloname, errloname, var->getErrorLo());
                  detailedOutput->addOwned(*errlo);
   
                  TString errhiname = TString().Append(prefix).Append(TString::Format("%s_errhi", var->GetName()));
                  //					TString errhidesc = TString::Format("%s high error for fit %u", var->GetTitle(), fitNumber);
                  RooRealVar* errhi = new RooRealVar(errhiname, errhiname, var->getErrorHi());
                  detailedOutput->addOwned(*errhi);
               }
				}
				delete it;

				// monitor a few more variables
				detailedOutput->addOwned( *new RooRealVar(TString().Append(prefix).Append("minNLL"), TString().Append(prefix).Append("minNLL"), result->minNll() ) );
				detailedOutput->addOwned( *new RooRealVar(TString().Append(prefix).Append("fitStatus"), TString().Append(prefix).Append("fitStatus"), result->status() ) );
				detailedOutput->addOwned( *new RooRealVar(TString().Append(prefix).Append("covQual"), TString().Append(prefix).Append("covQual"), result->covQual() ) );
				detailedOutput->addOwned( *new RooRealVar(TString().Append(prefix).Append("numInvalidNLLEval"), TString().Append(prefix).Append("numInvalidNLLEval"), result->numInvalidNLL() ) );
				return detailedOutput;
			}

			DetailedOutputAggregator() {
				result = NULL;
				builtSet = NULL;
			}

			// For each variable in aset, prepend prefix to its name and add
			// to the internal store. Note this will not appear in the produced
			// dataset unless CommitSet is called.
			void AppendArgSet(const RooArgSet *aset, TString prefix="") {
				if (aset == NULL) {
				   // silently ignore
					//std::cout << "Attempted to append NULL" << endl;
					return;
				}
				if (builtSet == NULL) {
					builtSet = new RooArgSet();
				}
				TIterator* iter = aset->createIterator();
				while(RooRealVar* v = dynamic_cast<RooRealVar*>( iter->Next() ) ) {
					TString renamed(TString::Format("%s%s", prefix.Data(), v->GetName()));
					if (result != NULL) {
						// we already commited an argset once, so we expect all columns to already be in the set
						builtSet->setRealValue(renamed, v->getVal());
					}
					else {
						// we never commited, so by default all columns are expected to not exist
						RooRealVar *var = new RooRealVar(renamed, v->GetTitle(), v->getVal());
						if (!builtSet->addOwned(*var)) {
							delete var;
							builtSet->setRealValue(renamed, v->getVal());
						}
					}
				}
				delete iter;
			}

			// Commit to the result RooDataSet.
			void CommitSet(double weight=1.0) {
				if (result == NULL) {
					result = new RooDataSet("", "",
							RooArgSet( *(new RooRealVar("weight","weight",1.0)), "tmpSet" ), "weight");
					InitializeColumns(result, builtSet);
				}
				result->add(*builtSet, weight);
				TIterator* iter = builtSet->createIterator();
				while(RooAbsArg* v = dynamic_cast<RooAbsArg*>( iter->Next() ) ) {
					builtSet->setRealValue(v->GetName(), -999.0);
				}
				delete iter;
			}

			RooDataSet *GetAsDataSet(TString name, TString title) {
			   RooDataSet* temp = NULL;
				if( result ) {
				   temp = new RooDataSet( *result );
				   temp->SetNameTitle( name.Data(), title.Data() );
				}else{
					temp = new RooDataSet(name.Data(), title.Data(),
							RooArgSet( *(new RooRealVar("weight","weight",1.0)), "tmpSet" ), "weight");
				}

				return temp;
			}

			virtual ~DetailedOutputAggregator() {
				if (result != NULL) delete result;
				if (builtSet != NULL) delete builtSet;
			}

		private:
			RooDataSet *result;
			RooArgSet *builtSet;

			void InitializeColumns(RooDataSet *dset, RooArgSet *aset) {
				TIterator* iter = aset->createIterator();
				while(RooAbsArg* v = dynamic_cast<RooAbsArg*>( iter->Next() ) ) {
					dset->addColumn( *(new RooRealVar(v->GetName(), v->GetTitle(), -1.0)));
				}
				delete iter;
			}

		protected:
			ClassDef(DetailedOutputAggregator,1)
	};
}

#endif
