// @(#)root/roostats:$Id$

/*************************************************************************
 * Project: RooStats                                                     *
 * Package: RooFit/RooStats                                              *
 * Authors:                                                              *
 *   Kyle Cranmer, Lorenzo Moneta, Gregory Schott, Wouter Verkerke       *
 *************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

///////////////////////////////////////////////////////////////////////////
/// BEGIN_HTML
/// HybridCalculator class: this class is a fresh rewrite in RooStats of
/// 	RooStatsCms/LimitCalculator developped by D. Piparo and G. Schott
/// END_HTML
///////////////////////////////////////////////////////////////////////////

#include "RooDataHist.h"
#include "RooDataSet.h"
#include "RooGlobalFunc.h" // for RooFit::Extended()
#include "RooNLLVar.h"
#include "RooRealVar.h"
#include "RooTreeData.h"

#include "RooStats/HybridCalculator.h"


/// ClassImp for building the THtml documentation of the class
ClassImp(RooStats::HybridCalculator)

using namespace RooStats;

///////////////////////////////////////////////////////////////////////////

HybridCalculator::HybridCalculator( const char *name,
                                    const char *title,
                                    RooAbsPdf& sbModel,
                                    RooAbsPdf& bModel,
                                    RooArgList& observables,
                                    RooArgSet& parameters,
                                    RooAbsPdf& priorPdf ) :
   /*HypoTestCalculator(name,title),*/ /// TO DO
   fName(name),
   fTitle(title),
   fSbModel(sbModel),
   fBModel(bModel),
   fObservables(observables),
   fParameters(parameters),
   fPriorPdf(priorPdf)
{
   /// HybridCalculator constructor:
   /// the user need to specify the models in the S+B case and B-only case,
   /// the list of observables of the model(s) (for MC-generation), the list of parameters 
   /// that are marginalised and the prior distribution of those parameters

   this->SetTestStatistics(1); /// set to default

   /* if ( _verbose ) */ this->Print("v"); /// TO DO: add the verbose mode
}

///////////////////////////////////////////////////////////////////////////

HybridCalculator::~HybridCalculator()
{
   /// HybridCalculator destructor
}

///////////////////////////////////////////////////////////////////////////

void HybridCalculator::SetTestStatistics(int index)
{
   /// set the desired test statistics:
   /// index=1 : 2 * log( L_sb / L_b )  (DEFAULT)
   /// index=2 : number of generated events
   /// if the index is different to any of those values, the default is used
   fTestStatisticsIdx = index;
}

///////////////////////////////////////////////////////////////////////////

void HybridCalculator::Calculate(RooAbsData& data, unsigned int nToys, bool usePriors)
{
   /// prepare and run the toy-MC experiments in order to calculate the hypothesis test
   /// for the given data

   /// TO DO: compute for data
   data.Print(); // TO DO: just for the warnings

   return RunToys(nToys,usePriors);
}
///////////////////////////////////////////////////////////////////////////

/// TO DO: add other data types constructors (?)

///////////////////////////////////////////////////////////////////////////

void HybridCalculator::RunToys(unsigned int nToys, bool usePriors)
{
   /// do the actual run-MC processing
   std::cout << "HybridCalculator: run " << nToys << " toy-MC experiments\n";
   std::cout << "with test statistics index: " << fTestStatisticsIdx << "\n";

   assert(nToys > 0);

   /// backup the initial values of the parameters that are varied by the prior MC-integration
   int nParameters = fParameters.getSize();
   double* parameterValues = 0; /// array to hold the initial parameter values
   RooArgList parametersList("parametersList"); /// transforms the RooArgSet in a RooArgList (needed for .at())
   if (usePriors && nParameters>0) {
      parametersList.add(fParameters);
      parameterValues = new double[nParameters];
      for (int iParameter=0; iParameter<nParameters; iParameter++) {
         RooRealVar* oneParam = (RooRealVar*) parametersList.at(iParameter);
         parameterValues[iParameter] = oneParam->getVal();
      }
   }

   for (unsigned int iToy=0; iToy<nToys; iToy++) {

      /// prints a progress report every 500 iterations
      /// TO DO: add a verbose flag inherited from HypoTestCalculator
      if ( /* _verbose && */ iToy>0 && iToy%500==0) {
         std::cout << "Running toy number " << iToy << " / " << nToys << std::endl;
      }

      /// vary the value of the integrated parameters according to the prior pdf
      if (usePriors && nParameters>0) {
         /// generation from the prior pdf (TO DO: RooMCStudy could be used here)
         RooDataSet* tmpValues = (RooDataSet*) fPriorPdf.generate(fParameters,1);
         for (int iParameter=0; iParameter<nParameters; iParameter++) {
            RooRealVar* oneParam = (RooRealVar*) parametersList.at(iParameter);
            oneParam->setVal(tmpValues->get()->getRealValue(oneParam->GetName()));
         }
         delete tmpValues;
      }

      /// generate the dataset in the S+B hypothesis
      RooTreeData* sbData = static_cast<RooTreeData*> (fSbModel.generate(fObservables,RooFit::Extended()));

      /// work-around in case of an empty dataset (TO DO: need a debug in RooFit?)
      bool sbIsEmpty = false;
      if (sbData==NULL) {
         sbIsEmpty = true;
         // if ( _verbose ) std::cout << "empty S+B dataset!\n";
         RooDataSet* sbDataDummy=new RooDataSet("sbDataDummy","empty dataset",fObservables);
         sbData = static_cast<RooTreeData*>(new RooDataHist ("sbDataEmpty","",fObservables,*sbDataDummy));
         delete sbDataDummy;
      }

      /// generate the dataset in the B-only hypothesis
      RooTreeData* bData = static_cast<RooTreeData*> (fBModel.generate(fObservables,RooFit::Extended()));

      /// work-around in case of an empty dataset (TO DO: need a debug in RooFit?)
      bool bIsEmpty = false;
      if (bData==NULL) {
         bIsEmpty = true;
         // if ( _verbose ) std::cout << "empty B-only dataset!\n";
         RooDataSet* bDataDummy=new RooDataSet("bDataDummy","empty dataset",fObservables);
         bData = static_cast<RooTreeData*>(new RooDataHist ("bDataEmpty","",fObservables,*bDataDummy));
         delete bDataDummy;
      }

      /// restore the parameters to their initial values
      if (usePriors && nParameters>0) {
         for (int iParameter=0; iParameter<nParameters; iParameter++) {
            RooRealVar* oneParam = (RooRealVar*) parametersList.at(iParameter);
            oneParam->setVal(parameterValues[iParameter]);
         }
      }

      /// TO DO: add test statistics index variable

      /// evaluate the test statistic in the S+B case
      if ( fTestStatisticsIdx==2 ) {
         /// number of events used as test statistics
         int nEvents = 0;
         if ( !sbIsEmpty ) sbData->numEntries();
         /// TO DO: store it somewhere!!!
         std::cout << nEvents << std::endl; // for the warnings
         // sb_vals.push_back(m2lnQ);
      } else {
         /// likelihood ratio used as test statistics (default)
         RooNLLVar sb_sb_nll("sb_sb_nll","sb_sb_nll",fSbModel,*sbData,RooFit::Extended());
         RooNLLVar b_sb_nll("b_sb_nll","b_sb_nll",fBModel,*sbData,RooFit::Extended());
         double m2lnQ = 2*(sb_sb_nll.getVal()-b_sb_nll.getVal());
         /// TO DO: store it somewhere!!!
         std::cout << m2lnQ << std::endl; // for the warnings
         // sb_vals.push_back(m2lnQ);
      }

      /// evaluate the test statistic in the B-only case
      if ( fTestStatisticsIdx==2 ) {
         /// number of events used as test statistics
         int nEvents = 0;
         if ( !bIsEmpty ) bData->numEntries();
         /// TO DO: store it somewhere!!!
         std::cout << nEvents << std::endl; // for the warnings
         // b_vals.push_back(m2lnQ);
      } else {
         /// likelihood ratio used as test statistics (default)
         RooNLLVar sb_b_nll("sb_b_nll","sb_b_nll",fSbModel,*bData,RooFit::Extended());
         RooNLLVar b_b_nll("b_b_nll","b_b_nll",fBModel,*bData,RooFit::Extended());
         double m2lnQ = 2*(sb_b_nll.getVal()-b_b_nll.getVal());
         /// TO DO: store it somewhere!!!
         std::cout << m2lnQ << std::endl; // for the warnings
         // b_vals.push_back(m2lnQ);
      }

      /// delete the toy-MC datasets
      delete sbData;
      delete bData;

   } /// end of loop over toy-MC experiments

   /// restore the parameters to their initial values (for safety) and delete the array of values
   if (usePriors && nParameters>0) {
      for (int iParameter=0; iParameter<nParameters; iParameter++) {
         RooRealVar* oneParam = (RooRealVar*) parametersList.at(iParameter);
         oneParam->setVal(parameterValues[iParameter]);
      }
      delete parameterValues;
   }

   return;
}

///////////////////////////////////////////////////////////////////////////

void HybridCalculator::Print(const char* options)
{
   /// Print out some information about the input models

   std::cout << "Signal plus background model:\n";
   fSbModel.Print(options);

   std::cout << "\nBackground model:\n";
   fBModel.Print(options);

   std::cout << "\nObservables:\n";
   fObservables.Print(options);

   std::cout << "\nParameters being integrated:\n";
   fParameters.Print(options);

   std::cout << "\nPrior PDF model for integration:\n";
   fPriorPdf.Print(options);

   return;
}
