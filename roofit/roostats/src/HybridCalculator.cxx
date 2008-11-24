// @(#)root/roostats:$Id$

/*************************************************************************
 * Project: RooStats                                                     *
 * Package: RooFit/RooStats                                              *
 * Authors:                                                              *
 *   Kyle Cranmer, Lorenzo Moneta, Gregory Schott, Wouter Verkerke       *
 * Other author of this class: Danilo Piparo                             *
 *************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//_________________________________________________________________
/**
HybridCalculator class: this class is a fresh rewrite in RooStats of
	RooStatsCms/LimitCalculator developped by D. Piparo and G. Schott
Authors: D. Piparo, G. Schott - Universitaet Karlsruhe

The class is born from the need to have an implementation of the CLs 
method that could take advantage from the RooFit Package.
The basic idea is the following: 
- Instantiate an object specifying a signal+background model, a background model and a dataset.
- Perform toy MC experiments to know the distributions of -2lnQ 
- Calculate the CLsb and CLs values as "integrals" of these distributions.

The class allows the user to input models as RooAbsPdf or TH1 object 
pointers (the pdfs must be "extended": for more information please refer to 
http://roofit.sourceforge.net). The dataset can be entered as a 
RooTreeData or TH1 object pointer. 

Unlike the TLimit Class a complete MC generation is performed at each step 
and not a simple Poisson fluctuation of the contents of the bins.
Another innovation is the treatment of the nuisance parameters. The user 
can input in the constructor nuisance parameters.
To include the information that we have about the nuisance parameters a prior
PDF (RooAbsPdf) should be specified

The result of the calculations is returned as a HybridResult object pointer.

see also the following interesting references:
- Alex Read, "Presentation of search results: the CLs technique" Journal of Physics G: Nucl. // Part. Phys. 28 2693-2704 (2002). http://www.iop.org/EJ/abstract/0954-3899/28/10/313/

- Alex Read, "Modified Frequentist Analysis of Search Results (The CLs Method)" CERN 2000-005 (30 May 2000)

- V. Bartsch, G.Quast, "Expected signal observability at future experiments" CMS NOTE 2005/004

- http://root.cern.ch/root/html/src/TLimit.html
*/


#include "RooDataHist.h"
#include "RooDataSet.h"
#include "RooGlobalFunc.h"
#include "RooNLLVar.h"
#include "RooRealVar.h"
#include "RooTreeData.h"

#include "RooStats/HybridCalculator.h"

ClassImp(RooStats::HybridCalculator)

using namespace RooStats;

///////////////////////////////////////////////////////////////////////////

HybridCalculator::HybridCalculator( const char *name,
                                    const char *title,
                                    RooAbsPdf& sbModel,
                                    RooAbsPdf& bModel,
                                    RooArgList& observables,
                                    RooArgSet& nuisance_parameters,
                                    RooAbsPdf& priorPdf ) :
   TNamed(name,title),
   fSbModel(sbModel),
   fBModel(bModel),
   fObservables(observables),
   fParameters(nuisance_parameters),
   fPriorPdf(priorPdf)
{
   /// HybridCalculator constructor:
   /// the user need to specify the models in the S+B case and B-only case,
   /// the list of observables of the model(s) (for MC-generation), the list of parameters 
   /// that are marginalised and the prior distribution of those parameters

   SetTestStatistics(1); /// set to default

   // this->Print();
   /* if ( _verbose ) */ //this->PrintMore("v"); /// TO DO: add the verbose mode
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

HybridResult* HybridCalculator::Calculate(TH1& data, unsigned int nToys, bool usePriors)
{
   /// first compute the test statistics for data and then prepare and run the toy-MC experiments

   /// convert data TH1 histogram to a RooDataHist
   TString dataHistName = GetName(); dataHistName += "_roodatahist";
   RooDataHist dataHist(dataHistName,"Data distribution as RooDataHist converted from TH1",fObservables,&data);

   HybridResult* result = Calculate(dataHist,nToys,usePriors);

   return result;
}

///////////////////////////////////////////////////////////////////////////

HybridResult* HybridCalculator::Calculate(RooTreeData& data, unsigned int nToys, bool usePriors)
{
   /// first compute the test statistics for data and then prepare and run the toy-MC experiments

   double testStatData = 0;
   if ( fTestStatisticsIdx==2 ) {
      /// number of events used as test statistics
      double nEvents = data.sumEntries();
      testStatData = nEvents;
   } else {
      /// likelihood ratio used as test statistics (default)
      RooNLLVar sb_nll("sb_nll","sb_nll",fSbModel,data,RooFit::Extended());
      RooNLLVar b_nll("b_nll","b_nll",fBModel,data,RooFit::Extended());
      double m2lnQ = 2*(sb_nll.getVal()-b_nll.getVal());
      testStatData = m2lnQ;
   }

   HybridResult* result = Calculate(nToys,usePriors);
   result->SetDataTestStatistics(testStatData);

   return result;
}

///////////////////////////////////////////////////////////////////////////

HybridResult* HybridCalculator::Calculate(unsigned int nToys, bool usePriors)
{
   std::vector<double> bVals;
   bVals.reserve(nToys);

   std::vector<double> sbVals;
   sbVals.reserve(nToys);

   RunToys(bVals,sbVals,nToys,usePriors);

   HybridResult* result = new HybridResult(GetName(),GetTitle(),sbVals,bVals);

   return result;
}

///////////////////////////////////////////////////////////////////////////

void HybridCalculator::RunToys(std::vector<double>& bVals, std::vector<double>& sbVals, unsigned int nToys, bool usePriors)
{
   /// do the actual run-MC processing
   std::cout << "HybridCalculator: run " << nToys << " toy-MC experiments\n";
   std::cout << "with test statistics index: " << fTestStatisticsIdx << "\n";

   assert(nToys > 0);

   /// backup the initial values of the parameters that are varied by the prior MC-integration
   int nParameters = fParameters.getSize();
   double* parameterValues = 0;  /// array to hold the initial parameter values
   RooArgList parametersList("parametersList");  /// transforms the RooArgSet in a RooArgList (needed for .at())
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

      /// evaluate the test statistic in the S+B case
      if ( fTestStatisticsIdx==2 ) {
         /// number of events used as test statistics
         double nEvents = 0;
         if ( !sbIsEmpty ) nEvents = sbData->numEntries();
         sbVals.push_back(nEvents);
      } else {
         /// likelihood ratio used as test statistics (default)
         RooNLLVar sb_sb_nll("sb_sb_nll","sb_sb_nll",fSbModel,*sbData,RooFit::Extended());
         RooNLLVar b_sb_nll("b_sb_nll","b_sb_nll",fBModel,*sbData,RooFit::Extended());
         double m2lnQ = 2*(sb_sb_nll.getVal()-b_sb_nll.getVal());
         sbVals.push_back(m2lnQ);
      }

      /// evaluate the test statistic in the B-only case
      if ( fTestStatisticsIdx==2 ) {
         /// number of events used as test statistics
         double nEvents = 0;
         if ( !bIsEmpty ) nEvents = bData->numEntries();
         bVals.push_back(nEvents);
      } else {
         /// likelihood ratio used as test statistics (default)
         RooNLLVar sb_b_nll("sb_b_nll","sb_b_nll",fSbModel,*bData,RooFit::Extended());
         RooNLLVar b_b_nll("b_b_nll","b_b_nll",fBModel,*bData,RooFit::Extended());
         double m2lnQ = 2*(sb_b_nll.getVal()-b_b_nll.getVal());
         bVals.push_back(m2lnQ);
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

void HybridCalculator::PrintMore(const char* options)
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
