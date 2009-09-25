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
RooAbsData or TH1 object pointer.  

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
#include "RooAbsData.h"
#include "RooWorkspace.h"

#include "TH1.h"

#include "RooStats/HybridCalculator.h"

ClassImp(RooStats::HybridCalculator)

using namespace RooStats;

///////////////////////////////////////////////////////////////////////////

HybridCalculator::HybridCalculator(const char *name,
                                   const char *title ) : 
   TNamed(TString(name), TString(title)),
   fSbModel(0),
   fBModel(0),
   fObservables(0),
   fNuisanceParameters(0),
   fPriorPdf(0),
   fData(0)
{
   // constructor with name and title
   // set default parameters
   SetTestStatistics(1); 
   SetNumberOfToys(1000); 
   UseNuisance(false); 
}


/// constructor without the data - is it needed ???????????
HybridCalculator::HybridCalculator( const char *name,
                                    const char *title,
                                    RooAbsPdf& sbModel,
                                    RooAbsPdf& bModel,
                                    RooArgList& observables,
                                    const RooArgSet* nuisance_parameters,
                                    RooAbsPdf* priorPdf ,
				    bool GenerateBinned ) :
   TNamed(name,title),
   fSbModel(&sbModel),
   fBModel(&bModel),
   fNuisanceParameters(nuisance_parameters),
   fPriorPdf(priorPdf),
   fData(0),
   //fWS(0),
   fGenerateBinned(GenerateBinned)
{
   /// specific HybridCalculator constructor:
   /// the user need to specify the models in the S+B case and B-only case,
   /// the list of observables of the model(s) (for MC-generation), the list of parameters 
   /// that are marginalised and the prior distribution of those parameters

   // observables are managed by the class (they are copied in) 
  fObservables = new RooArgList(observables);
  //Try to recover the informations from the pdf's
  //fObservables=new RooArgList("fObservables");
  //fNuisanceParameters=new RooArgSet("fNuisanceParameters");
  // if (priorPdf){
    
    
  // }
  


  SetTestStatistics(1); /// set to default
  SetNumberOfToys(1000); 
  if (priorPdf) UseNuisance(true); 
  
   // this->Print();
   /* if ( _verbose ) */ //this->PrintMore("v"); /// TO DO: add the verbose mode
}


HybridCalculator::HybridCalculator( RooAbsData & data, 
                                    RooAbsPdf& sbModel,
                                    RooAbsPdf& bModel,
                                    const RooArgSet* nuisance_parameters,
                                    RooAbsPdf* priorPdf,
				    bool GenerateBinned ) :
   fSbModel(&sbModel),
   fBModel(&bModel),
   fObservables(0),
   fNuisanceParameters(nuisance_parameters),
   fPriorPdf(priorPdf),
   fData(&data),
//   fWS(0),
   fGenerateBinned(GenerateBinned)
{
   /// HybridCalculator constructor for performing hypotesis test 
   /// the user need to specify the data set, the models in the S+B case and B-only case. 
   /// In case of treatment of nuisance parameter, the user need to specify the  
   /// the list of parameters  that are marginalised and the prior distribution of those parameters

   //Initialize(data, bModel, sbModel, 0, 0, nuisance_parameters, priorPdf); 

   SetTestStatistics(1); /// set to default
   SetNumberOfToys(1000); 
   if (priorPdf) UseNuisance(true); 
}

HybridCalculator::HybridCalculator( const char *name,
                                    const char *title,
                                    RooAbsData & data, 
                                    RooAbsPdf& sbModel,
                                    RooAbsPdf& bModel,
                                    const RooArgSet* nuisance_parameters,
                                    RooAbsPdf* priorPdf, 
				    bool GenerateBinned ) :
   TNamed(name,title),
   fSbModel(&sbModel),
   fBModel(&bModel),
   fObservables(0),
   fNuisanceParameters(nuisance_parameters),
   fPriorPdf(priorPdf),
   fData(&data),
   // fWS(0),
   fGenerateBinned(GenerateBinned)
{
   /// HybridCalculator constructor for performing hypotesis test 
   /// the user need to specify the data set, the models in the S+B case and B-only case. 
   /// In case of treatment of nuisance parameter, the user need to specify the  
   /// the list of parameters  that are marginalised and the prior distribution of those parameters

   //Initialize(data, bModel, sbModel, 0, 0, nuisance_parameters, priorPdf); 

   SetTestStatistics(1); /// set to default
   SetNumberOfToys(1000); 
   if (priorPdf) UseNuisance(true); 

}


HybridCalculator::HybridCalculator( const char *name,
                                    const char *title,
                                    RooAbsData& data, 
                                    const ModelConfig& sbModel, 
                                    const ModelConfig& bModel) :
   TNamed(name,title),
   fSbModel(sbModel.GetPdf()),
   fBModel(bModel.GetPdf()),
   fObservables(0),  // no need to set them - can be taken from the data
   fNuisanceParameters((sbModel.GetNuisanceParameters()) ? sbModel.GetNuisanceParameters()  :  bModel.GetNuisanceParameters()),
   fPriorPdf((sbModel.GetPriorPdf()) ? sbModel.GetPriorPdf()  :  bModel.GetPriorPdf()),
   fData(&data)
   //fWS(0) 
{
  /// Constructor with a ModelConfig object representing the signal + background model and 
  /// another model config representig the background only model
  /// a Prior pdf for the nuiscane parameter of the signal and background can be specified in 
  /// the s+b model or the b model. If it is specified in the s+b model, the one of the s+b model will be used 

   if (fPriorPdf) 
      UseNuisance(true);

  SetTestStatistics(1);
  SetNumberOfToys(1000); 
}

///////////////////////////////////////////////////////////////////////////

HybridCalculator::~HybridCalculator()
{
   /// HybridCalculator destructor
   //if( fOwnsWorkspace && fWS) delete fWS;
   if (fObservables) delete fObservables; 
}

///////////////////////////////////////////////////////////////////////////

void HybridCalculator::SetNullModel(const ModelConfig& model)
{
   fBModel = model.GetPdf();
   // only if it has not been set before
   if (!fPriorPdf) fPriorPdf = model.GetPriorPdf(); 
   if (!fNuisanceParameters) fNuisanceParameters = model.GetNuisanceParameters(); 
}

void HybridCalculator::SetAlternateModel(const ModelConfig& model)
{
   fSbModel = model.GetPdf();
   fPriorPdf = model.GetPriorPdf(); 
   fNuisanceParameters = model.GetNuisanceParameters(); 
}

void HybridCalculator::SetTestStatistics(int index)
{
   /// set the desired test statistics:
   /// index=1 : 2 * log( L_sb / L_b )  (DEFAULT)
   /// index=2 : number of generated events
   /// index=3 : profiled likelihood ratio
   /// if the index is different to any of those values, the default is used
   fTestStatisticsIdx = index;
}

///////////////////////////////////////////////////////////////////////////

HybridResult* HybridCalculator::Calculate(TH1& data, unsigned int nToys, bool usePriors) const
{
   /// first compute the test statistics for data and then prepare and run the toy-MC experiments

   /// convert data TH1 histogram to a RooDataHist
   TString dataHistName = GetName(); dataHistName += "_roodatahist";
   RooDataHist dataHist(dataHistName,"Data distribution as RooDataHist converted from TH1",*fObservables,&data);

   HybridResult* result = Calculate(dataHist,nToys,usePriors);

   return result;
}

///////////////////////////////////////////////////////////////////////////

HybridResult* HybridCalculator::Calculate(RooAbsData& data, unsigned int nToys, bool usePriors) const
{
   /// first compute the test statistics for data and then prepare and run the toy-MC experiments

   double testStatData = 0;
   if ( fTestStatisticsIdx==2 ) {
      /// number of events used as test statistics
      double nEvents = data.sumEntries();
      testStatData = nEvents;
   } else if ( fTestStatisticsIdx==3 ) {
      /// profiled likelihood ratio used as test statistics
      RooNLLVar sb_nll("sb_nll","sb_nll",*fSbModel,data,RooFit::Extended());
      fSbModel->fitTo(data);
      double sb_nll_val = sb_nll.getVal();
      RooNLLVar b_nll("b_nll","b_nll",*fBModel,data,RooFit::Extended());
      fBModel->fitTo(data);
      double b_nll_val = b_nll.getVal();
      double m2lnQ = 2*(sb_nll_val-b_nll_val);
      testStatData = m2lnQ;
   } else if ( fTestStatisticsIdx==1 ) {
      /// likelihood ratio used as test statistics (default)
      RooNLLVar sb_nll("sb_nll","sb_nll",*fSbModel,data,RooFit::Extended());
      RooNLLVar b_nll("b_nll","b_nll",*fBModel,data,RooFit::Extended());
      double m2lnQ = 2*(sb_nll.getVal()-b_nll.getVal());
      testStatData = m2lnQ;
   }

   HybridResult* result = Calculate(nToys,usePriors);
   result->SetDataTestStatistics(testStatData);

   return result;
}

///////////////////////////////////////////////////////////////////////////

HybridResult* HybridCalculator::Calculate(unsigned int nToys, bool usePriors) const
{
   std::vector<double> bVals;
   bVals.reserve(nToys);

   std::vector<double> sbVals;
   sbVals.reserve(nToys);

   RunToys(bVals,sbVals,nToys,usePriors);

   HybridResult* result;

   if ( fTestStatisticsIdx==2 )
     result = new HybridResult(GetName(),GetTitle(),sbVals,bVals,false);
   else 
     result = new HybridResult(GetName(),GetTitle(),sbVals,bVals);

   return result;
}

///////////////////////////////////////////////////////////////////////////

void HybridCalculator::RunToys(std::vector<double>& bVals, std::vector<double>& sbVals, unsigned int nToys, bool usePriors) const
{
   /// do the actual run-MC processing
   std::cout << "HybridCalculator: run " << nToys << " toy-MC experiments\n";
   std::cout << "with test statistics index: " << fTestStatisticsIdx << "\n";
   if (usePriors) std::cout << "marginalize nuisance parameters \n";

   assert(nToys > 0);
   assert(fBModel);
   assert(fSbModel);
   if (usePriors)  { 
      assert(fPriorPdf); 
      assert(fNuisanceParameters);
   }

   std::vector<double> parameterValues; /// array to hold the initial parameter values
   /// backup the initial values of the parameters that are varied by the prior MC-integration
   int nParameters = (fNuisanceParameters) ? fNuisanceParameters->getSize() : 0;
   RooArgList parametersList("parametersList");  /// transforms the RooArgSet in a RooArgList (needed for .at())
   if (usePriors && nParameters>0) {
      parametersList.add(*fNuisanceParameters);
      parameterValues.resize(nParameters);
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
         RooDataSet* tmpValues = (RooDataSet*) fPriorPdf->generate(*fNuisanceParameters,1);
         for (int iParameter=0; iParameter<nParameters; iParameter++) {
            RooRealVar* oneParam = (RooRealVar*) parametersList.at(iParameter);
            oneParam->setVal(tmpValues->get()->getRealValue(oneParam->GetName()));
         }
         delete tmpValues;
      }

      /// generate the dataset in the B-only hypothesis
      RooAbsData* bData;
      if (fGenerateBinned)
	bData = static_cast<RooAbsData*> (fBModel->generateBinned(*fObservables,RooFit::Extended()));	
      else 
	bData = static_cast<RooAbsData*> (fBModel->generate(*fObservables,RooFit::Extended()));

      /// work-around in case of an empty dataset (TO DO: need a debug in RooFit?)
      bool bIsEmpty = false;
      if (bData==NULL) {
         bIsEmpty = true;
         // if ( _verbose ) std::cout << "empty B-only dataset!\n";
         RooDataSet* bDataDummy=new RooDataSet("bDataDummy","empty dataset",*fObservables);
         bData = static_cast<RooAbsData*>(new RooDataHist ("bDataEmpty","",*fObservables,*bDataDummy));
         delete bDataDummy;
      }

      /// generate the dataset in the S+B hypothesis
      RooAbsData* sbData;
      if (fGenerateBinned)    
	sbData = static_cast<RooAbsData*> (fSbModel->generateBinned(*fObservables,RooFit::Extended()));
      else
	sbData = static_cast<RooAbsData*> (fSbModel->generate(*fObservables,RooFit::Extended()));

      /// work-around in case of an empty dataset (TO DO: need a debug in RooFit?)
      bool sbIsEmpty = false;
      if (sbData==NULL) {
         sbIsEmpty = true;
         // if ( _verbose ) std::cout << "empty S+B dataset!\n";
         RooDataSet* sbDataDummy=new RooDataSet("sbDataDummy","empty dataset",*fObservables);
         sbData = static_cast<RooAbsData*>(new RooDataHist ("sbDataEmpty","",*fObservables,*sbDataDummy));
         delete sbDataDummy;
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
      } else if ( fTestStatisticsIdx==3 ) {
         /// profiled likelihood ratio used as test statistics
         RooNLLVar sb_nll("sb_nll","sb_nll",*fSbModel,*sbData,RooFit::Extended());
         fSbModel->fitTo(*sbData);
         double sb_nll_val = sb_nll.getVal();
         RooNLLVar b_nll("b_nll","b_nll",*fBModel,*sbData,RooFit::Extended());
         fBModel->fitTo(*sbData);
         double b_nll_val = b_nll.getVal();
         double m2lnQ = 2*(sb_nll_val-b_nll_val);
         sbVals.push_back(m2lnQ);
      } else if ( fTestStatisticsIdx==1 ) {
         /// likelihood ratio used as test statistics (default)
         RooNLLVar sb_nll("sb_nll","sb_nll",*fSbModel,*sbData,RooFit::Extended());
         RooNLLVar b_nll("b_nll","b_nll",*fBModel,*sbData,RooFit::Extended());
         double m2lnQ = 2*(sb_nll.getVal()-b_nll.getVal());
         sbVals.push_back(m2lnQ);
      }

      /// evaluate the test statistic in the B-only case
      if ( fTestStatisticsIdx==2 ) {
         /// number of events used as test statistics
         double nEvents = 0;
         if ( !bIsEmpty ) nEvents = bData->numEntries();
         bVals.push_back(nEvents);
      } else if ( fTestStatisticsIdx==3 ) {
         /// profiled likelihood ratio used as test statistics
         RooNLLVar sb_nll("sb_nll","sb_nll",*fSbModel,*bData,RooFit::Extended());
         fSbModel->fitTo(*bData);
         double sb_nll_val = sb_nll.getVal();
         RooNLLVar b_nll("b_nll","b_nll",*fBModel,*bData,RooFit::Extended());
         fBModel->fitTo(*bData);
         double b_nll_val = b_nll.getVal();
         double m2lnQ = 2*(sb_nll_val-b_nll_val);
         bVals.push_back(m2lnQ);
      } else if ( fTestStatisticsIdx==1 ) {
         /// likelihood ratio used as test statistics (default)
         RooNLLVar sb_nll("sb_nll","sb_nll",*fSbModel,*bData,RooFit::Extended());
         RooNLLVar b_nll("b_nll","b_nll",*fBModel,*bData,RooFit::Extended());
         double m2lnQ = 2*(sb_nll.getVal()-b_nll.getVal());
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
   }

   return;
}

///////////////////////////////////////////////////////////////////////////

void HybridCalculator::PrintMore(const char* options) const
{
   /// Print out some information about the input models

   if (fSbModel) { 
      std::cout << "Signal plus background model:\n";
      fSbModel->Print(options);
   }

   if (fBModel) { 
      std::cout << "\nBackground model:\n";
      fBModel->Print(options);
   }
      
   if (fObservables) {  
      std::cout << "\nObservables:\n";
      fObservables->Print(options);
   }

   if (fNuisanceParameters) { 
      std::cout << "\nParameters being integrated:\n";
      fNuisanceParameters->Print(options);
   }

   if (fPriorPdf) { 
      std::cout << "\nPrior PDF model for integration:\n";
      fPriorPdf->Print(options);
   }

   return;
}
///////////////////////////////////////////////////////////////////////////
// implementation of inherited methods from HypoTestCalculator

HybridResult* HybridCalculator::GetHypoTest() const {  
   // perform the hypothesis test and return result of hypothesis test 

   // check first that everything needed is there 
   if (!DoCheckInputs()) return 0;  
   RooAbsData * treeData = dynamic_cast<RooAbsData *> (fData); 
   if (!treeData) { 
      std::cerr << "Error in HybridCalculator::GetHypoTest - invalid data type - return NULL" << std::endl;
      return 0; 
   }
   bool usePrior = (fUsePriorPdf && fPriorPdf ); 
   return Calculate( *treeData, fNToys, usePrior);  
}


bool HybridCalculator::DoCheckInputs() const { 
   if (!fData) { 
      std::cerr << "Error in HybridCalculator - data have not been set" << std::endl;
      return false; 
   }

   // if observable have not been set take them from data 
   if (!fObservables && fData->get() ) fObservables =  new RooArgList( *fData->get() );
   if (!fObservables) { 
      std::cerr << "Error in HybridCalculator - no observables" << std::endl;
      return false; 
   }

   if (!fSbModel) { 
      std::cerr << "Error in HybridCalculator - S+B pdf has not been set " << std::endl;
      return false; 
   }

   if (!fBModel) { 
      std::cerr << "Error in HybridCalculator - B pdf has not been set" << std::endl;
      return false; 
   }
   if (fUsePriorPdf && !fNuisanceParameters) { 
      std::cerr << "Error in HybridCalculator - nuisance parameters have not been set " << std::endl;
      return false; 
   }
   if (fUsePriorPdf && !fPriorPdf) { 
      std::cerr << "Error in HybridCalculator - prior pdf has not been set " << std::endl;
      return false; 
   }
   return true; 
}


