// @(#)root/roostats:$Id$
// Authors: Kevin Belasco        17/06/2009
// Authors: Kyle Cranmer         17/06/2009
/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class RooStats::MarkovChain
    \ingroup Roostats

Stores the steps in a Markov Chain of points.  Allows user to access the
weight and NLL value (if applicable) with which a point was added to the
MarkovChain.

*/

#include "Rtypes.h"

#include "TNamed.h"
#include "RooStats/MarkovChain.h"
#include "RooDataSet.h"
#include "RooArgSet.h"
#include "RooRealVar.h"
#include "RooStats/RooStatsUtils.h"
#include "RooDataHist.h"
#include "THnSparse.h"

using namespace std;

ClassImp(RooStats::MarkovChain);

using namespace RooFit;
using namespace RooStats;

static const char* WEIGHT_NAME = "weight_MarkovChain_local_";
static const char* NLL_NAME = "nll_MarkovChain_local_";
static const char* DATASET_NAME = "dataset_MarkovChain_local_";
static const char* DEFAULT_NAME = "_markov_chain";
static const char* DEFAULT_TITLE = "Markov Chain";

MarkovChain::MarkovChain() :
   TNamed(DEFAULT_NAME, DEFAULT_TITLE)
{
   fParameters = nullptr;
   fDataEntry = nullptr;
   fChain = nullptr;
   fNLL = nullptr;
   fWeight = nullptr;
}

MarkovChain::MarkovChain(RooArgSet& parameters) :
   TNamed(DEFAULT_NAME, DEFAULT_TITLE)
{
   fParameters = nullptr;
   fDataEntry = nullptr;
   fChain = nullptr;
   fNLL = nullptr;
   fWeight = nullptr;
   SetParameters(parameters);
}

MarkovChain::MarkovChain(const char* name, const char* title,
      RooArgSet& parameters) : TNamed(name, title)
{
   fParameters = nullptr;
   fDataEntry = nullptr;
   fChain = nullptr;
   fNLL = nullptr;
   fWeight = nullptr;
   SetParameters(parameters);
}

void MarkovChain::SetParameters(RooArgSet& parameters)
{
   delete fChain;
   delete fParameters;
   delete fDataEntry;

   fParameters = new RooArgSet();
   fParameters->addClone(parameters);

   // kbelasco: consider setting fDataEntry = fChain->get()
   // to see if that makes it possible to get values of variables without
   // doing string comparison
   RooRealVar nll(NLL_NAME, "-log Likelihood", 0);
   RooRealVar weight(WEIGHT_NAME, "weight", 0);

   fDataEntry = new RooArgSet();
   fDataEntry->addClone(parameters);
   fDataEntry->addClone(nll);
   fDataEntry->addClone(weight);
   fNLL = (RooRealVar*)fDataEntry->find(NLL_NAME);
   fWeight = (RooRealVar*)fDataEntry->find(WEIGHT_NAME);

   fChain = new RooDataSet(DATASET_NAME, "Markov Chain", *fDataEntry,WEIGHT_NAME);
}

void MarkovChain::Add(RooArgSet& entry, double nllValue, double weight)
{
   if (fParameters == nullptr)
      SetParameters(entry);
   RooStats::SetParameters(&entry, fDataEntry);
   fNLL->setVal(nllValue);
   //kbelasco: this is stupid, but some things might require it, so be doubly sure
   fWeight->setVal(weight);
   fChain->add(*fDataEntry, weight);
   //fChain->add(*fDataEntry);
}

void MarkovChain::AddWithBurnIn(MarkovChain& otherChain, Int_t burnIn)
{
   // Discards the first n accepted points.

   if(fParameters == nullptr) SetParameters(*(RooArgSet*)otherChain.Get());
   int counter = 0;
   for( int i=0; i < otherChain.Size(); i++ ) {
      RooArgSet* entry = (RooArgSet*)otherChain.Get(i);
      counter += 1;
      if( counter > burnIn ) {
         AddFast( *entry, otherChain.NLL(), otherChain.Weight() );
      }
   }
}
void MarkovChain::Add(MarkovChain& otherChain, double discardEntries)
{
   // Discards the first entries. This is different to the definition of
   // burn-in used in the Bayesian calculator where the first n accepted
   // terms from the proposal function are discarded.

   if(fParameters == nullptr) SetParameters(*(RooArgSet*)otherChain.Get());
   double counter = 0.0;
   for( int i=0; i < otherChain.Size(); i++ ) {
      RooArgSet* entry = (RooArgSet*)otherChain.Get(i);
      counter += otherChain.Weight();
      if( counter > discardEntries ) {
         AddFast( *entry, otherChain.NLL(), otherChain.Weight() );
      }
   }
}

void MarkovChain::AddFast(RooArgSet& entry, double nllValue, double weight)
{
   RooStats::SetParameters(&entry, fDataEntry);
   fNLL->setVal(nllValue);
   //kbelasco: this is stupid, but some things might require it, so be doubly sure
   fWeight->setVal(weight);
   fChain->addFast(*fDataEntry, weight);
   //fChain->addFast(*fDataEntry);
}

RooDataSet* MarkovChain::GetAsDataSet(RooArgSet* whichVars) const
{
   RooArgSet args;
   if (whichVars == nullptr) {
      //args.add(*fParameters);
      //args.add(*fNLL);
      args.add(*fDataEntry);
   } else {
      args.add(*whichVars);
   }

   RooDataSet* data;
   //data = dynamic_cast<RooDataSet*>(fChain->reduce(args));
   data = (RooDataSet*)fChain->reduce(args);

   return data;
}

RooDataSet* MarkovChain::GetAsDataSet(const RooCmdArg& arg1, const RooCmdArg& arg2,
                                      const RooCmdArg& arg3, const RooCmdArg& arg4, const RooCmdArg& arg5,
                                      const RooCmdArg& arg6, const RooCmdArg& arg7, const RooCmdArg& arg8) const
{
   RooDataSet* data;
   data = (RooDataSet*)fChain->reduce(arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8);
   return data;
}

RooDataHist* MarkovChain::GetAsDataHist(RooArgSet* whichVars) const
{
   RooArgSet args;
   if (whichVars == nullptr) {
      args.add(*fParameters);
      //args.add(*fNLL);
      //args.add(*fDataEntry);
   } else {
      args.add(*whichVars);
   }

   RooDataSet* data = (RooDataSet*)fChain->reduce(args);
   RooDataHist* hist = data->binnedClone();
   delete data;

   return hist;
}

RooDataHist* MarkovChain::GetAsDataHist(const RooCmdArg& arg1, const RooCmdArg& arg2,
                                        const RooCmdArg& arg3, const RooCmdArg& arg4, const RooCmdArg& arg5,
                                        const RooCmdArg& arg6, const RooCmdArg& arg7, const RooCmdArg& arg8) const
{
   RooDataSet* data;
   RooDataHist* hist;
   data = (RooDataSet*)fChain->reduce(arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8);
   hist = data->binnedClone();
   delete data;

   return hist;
}

THnSparse* MarkovChain::GetAsSparseHist(RooAbsCollection* whichVars) const
{
   RooArgList axes;
   if (whichVars == nullptr)
      axes.add(*fParameters);
   else
      axes.add(*whichVars);

   Int_t dim = axes.getSize();
   std::vector<double> min(dim);
   std::vector<double> max(dim);
   std::vector<Int_t> bins(dim);
   std::vector<const char *> names(dim);
   Int_t i = 0;
   for (auto const *var : static_range_cast<RooRealVar *>(axes)) {
      names[i] = var->GetName();
      min[i] = var->getMin();
      max[i] = var->getMax();
      bins[i] = var->numBins();
      ++i;
   }

   THnSparseF* sparseHist = new THnSparseF("posterior", "MCMC Posterior Histogram",
         dim, &bins[0], &min[0], &max[0]);

   // kbelasco: it appears we need to call Sumw2() just to get the
   // histogram to keep a running total of the weight so that Getsumw doesn't
   // just return 0
   sparseHist->Sumw2();

   // Fill histogram
   Int_t size = fChain->numEntries();
   const RooArgSet* entry;
   std::vector<double> x(dim);
   for ( i = 0; i < size; i++) {
      entry = fChain->get(i);

      for (Int_t ii = 0; ii < dim; ii++) {
         //LM:  doing this is probably quite slow
         x[ii] = entry->getRealValue( names[ii]);
         sparseHist->Fill(x.data(), fChain->weight());
      }
   }

   return sparseHist;
}

double MarkovChain::NLL(Int_t i) const
{
   // kbelasco: how to do this?
   //fChain->get(i);
   //return fNLL->getVal();
   return fChain->get(i)->getRealValue(NLL_NAME);
}

double MarkovChain::NLL() const
{
   // kbelasco: how to do this?
   //fChain->get();
   //return fNLL->getVal();
   return fChain->get()->getRealValue(NLL_NAME);
}

double MarkovChain::Weight() const
{
   return fChain->weight();
}

double MarkovChain::Weight(Int_t i) const
{
   fChain->get(i);
   return fChain->weight();
}
