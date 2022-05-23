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

/** \class RooStats::HybridCalculatorOriginal
 \ingroup Roostats


HybridCalculatorOriginal class. This class is deprecated and it is replaced by
the HybridCalculator.

This is a fresh rewrite in RooStats of `RooStatsCms/LimitCalculator` developed
by D. Piparo and G. Schott - Universitaet Karlsruhe

The class is born from the need to have an implementation of the CLs
method that could take advantage from the RooFit Package.
The basic idea is the following:

  - Instantiate an object specifying a signal+background model, a background model and a dataset.
  - Perform toy MC experiments to know the distributions of -2lnQ
  - Calculate the CLsb and CLs values as "integrals" of these distributions.

The class allows the user to input models as RooAbsPdf ( TH1 object could be used
by using the RooHistPdf class)

The pdfs must be "extended": for more information please refer to
http://roofit.sourceforge.net). The dataset can be entered as a
RooAbsData objects.

Unlike the TLimit Class a complete MC generation is performed at each step
and not a simple Poisson fluctuation of the contents of the bins.
Another innovation is the treatment of the nuisance parameters. The user
can input in the constructor nuisance parameters.
To include the information that we have about the nuisance parameters a prior
PDF (RooAbsPdf) should be specified

Different test statistic can be used (likelihood ratio, number of events or
profile likelihood ratio. The default is the likelihood ratio.
See the method SetTestStatistic.

The number of toys to be generated is controlled by SetNumberOfToys(n).

The result of the calculations is returned as a HybridResult object pointer.

see also the following interesting references:

  - Alex Read, "Presentation of search results: the CLs technique",
    Journal of Physics G: Nucl. Part. Phys. 28 2693-2704 (2002).
    see http://www.iop.org/EJ/abstract/0954-3899/28/10/313/

  - Alex Read, "Modified Frequentist Analysis of Search Results (The CLs Method)" CERN 2000-005 (30 May 2000)

  - V. Bartsch, G.Quast, "Expected signal observability at future experiments" CMS NOTE 2005/004

  - TLimit
*/

#include "RooStats/HybridCalculatorOriginal.h"

#include "RooStats/ModelConfig.h"

#include "RooDataHist.h"
#include "RooDataSet.h"
#include "RooGlobalFunc.h"
#include "RooNLLVar.h"
#include "RooRealVar.h"
#include "RooAbsData.h"
#include "RooWorkspace.h"

#include "TH1.h"

using namespace std;

ClassImp(RooStats::HybridCalculatorOriginal);

using namespace RooStats;

////////////////////////////////////////////////////////////////////////////////
/// constructor with name and title

HybridCalculatorOriginal::HybridCalculatorOriginal(const char *name) :
   TNamed(name,name),
   fSbModel(0),
   fBModel(0),
   fObservables(0),
   fNuisanceParameters(0),
   fPriorPdf(0),
   fData(0),
   fGenerateBinned(false),
   fUsePriorPdf(false),   fTmpDoExtended(true)
{
   // set default parameters
   SetTestStatistic(1);
   SetNumberOfToys(1000);
}

////////////////////////////////////////////////////////////////////////////////
/// HybridCalculatorOriginal constructor without specifying a data set
/// the user need to specify the models in the S+B case and B-only case,
/// the list of observables of the model(s) (for MC-generation), the list of parameters
/// that are marginalised and the prior distribution of those parameters

HybridCalculatorOriginal::HybridCalculatorOriginal( RooAbsPdf& sbModel,
                                    RooAbsPdf& bModel,
                                    RooArgList& observables,
                                    const RooArgSet* nuisance_parameters,
                                    RooAbsPdf* priorPdf ,
                                    bool GenerateBinned,
                                    int testStatistics,
                                    int numToys) :
   fSbModel(&sbModel),
   fBModel(&bModel),
   fNuisanceParameters(nuisance_parameters),
   fPriorPdf(priorPdf),
   fData(0),
   fGenerateBinned(GenerateBinned),
   fUsePriorPdf(false),
   fTmpDoExtended(true)
{

   // observables are managed by the class (they are copied in)
  fObservables = new RooArgList(observables);
  //Try to recover the information from the pdf's
  //fObservables=new RooArgList("fObservables");
  //fNuisanceParameters=new RooArgSet("fNuisanceParameters");
  // if (priorPdf){


  SetTestStatistic(testStatistics);
  SetNumberOfToys(numToys);

  if (priorPdf) UseNuisance(true);

   // this->Print();
   /* if ( _verbose ) */ //this->PrintMore("v"); /// TO DO: add the verbose mode
}

////////////////////////////////////////////////////////////////////////////////
/// HybridCalculatorOriginal constructor for performing hypotesis test
/// the user need to specify the data set, the models in the S+B case and B-only case.
/// In case of treatment of nuisance parameter, the user need to specify the
/// the list of parameters  that are marginalised and the prior distribution of those parameters

HybridCalculatorOriginal::HybridCalculatorOriginal( RooAbsData & data,
                                    RooAbsPdf& sbModel,
                                    RooAbsPdf& bModel,
                                    const RooArgSet* nuisance_parameters,
                                    RooAbsPdf* priorPdf,
                bool GenerateBinned,
                                    int testStatistics,
                                    int numToys) :
   fSbModel(&sbModel),
   fBModel(&bModel),
   fObservables(0),
   fNuisanceParameters(nuisance_parameters),
   fPriorPdf(priorPdf),
   fData(&data),
   fGenerateBinned(GenerateBinned),
   fUsePriorPdf(false),
   fTmpDoExtended(true)
{


   SetTestStatistic(testStatistics);
   SetNumberOfToys(numToys);

   if (priorPdf) UseNuisance(true);
}

////////////////////////////////////////////////////////////////////////////////
/// Constructor with a ModelConfig object representing the signal + background model and
/// another model config representig the background only model
/// a Prior pdf for the nuiscane parameter of the signal and background can be specified in
/// the s+b model or the b model. If it is specified in the s+b model, the one of the s+b model will be used

HybridCalculatorOriginal::HybridCalculatorOriginal( RooAbsData& data,
                                    const ModelConfig& sbModel,
                                    const ModelConfig& bModel,
                bool GenerateBinned,
                                    int testStatistics,
                                    int numToys) :
   fSbModel(sbModel.GetPdf()),
   fBModel(bModel.GetPdf()),
   fObservables(0),  // no need to set them - can be taken from the data
   fNuisanceParameters((sbModel.GetNuisanceParameters()) ? sbModel.GetNuisanceParameters()  :  bModel.GetNuisanceParameters()),
   fPriorPdf((sbModel.GetPriorPdf()) ? sbModel.GetPriorPdf()  :  bModel.GetPriorPdf()),
   fData(&data),
   fGenerateBinned(GenerateBinned),
   fUsePriorPdf(false),
   fTmpDoExtended(true)
{

  if (fPriorPdf) UseNuisance(true);

  SetTestStatistic(testStatistics);
  SetNumberOfToys(numToys);
}

////////////////////////////////////////////////////////////////////////////////
/// HybridCalculatorOriginal destructor

HybridCalculatorOriginal::~HybridCalculatorOriginal()
{
   if (fObservables) delete fObservables;
}

////////////////////////////////////////////////////////////////////////////////
/// Set the model describing the null hypothesis

void HybridCalculatorOriginal::SetNullModel(const ModelConfig& model)
{
   fBModel = model.GetPdf();
   // only if it has not been set before
   if (!fPriorPdf) fPriorPdf = model.GetPriorPdf();
   if (!fNuisanceParameters) fNuisanceParameters = model.GetNuisanceParameters();
}

////////////////////////////////////////////////////////////////////////////////
/// Set the model describing the alternate hypothesis

void HybridCalculatorOriginal::SetAlternateModel(const ModelConfig& model)
{
   fSbModel = model.GetPdf();
   fPriorPdf = model.GetPriorPdf();
   fNuisanceParameters = model.GetNuisanceParameters();
}

////////////////////////////////////////////////////////////////////////////////
/// set the desired test statistics:
///  - index=1 : likelihood ratio: 2 * log( L_sb / L_b )  (DEFAULT)
///  - index=2 : number of generated events
///  - index=3 : profiled likelihood ratio
/// if the index is different to any of those values, the default is used

void HybridCalculatorOriginal::SetTestStatistic(int index)
{
   fTestStatisticsIdx = index;
}

////////////////////////////////////////////////////////////////////////////////
/// first compute the test statistics for data and then prepare and run the toy-MC experiments

HybridResult* HybridCalculatorOriginal::Calculate(TH1& data, unsigned int nToys, bool usePriors) const
{

   /// convert data TH1 histogram to a RooDataHist
   auto dataHistName = std::string(GetName()) + "_roodatahist";
   RooDataHist dataHist(dataHistName,"Data distribution as RooDataHist converted from TH1",*fObservables,&data);

   HybridResult* result = Calculate(dataHist,nToys,usePriors);

   return result;
}

////////////////////////////////////////////////////////////////////////////////
/// first compute the test statistics for data and then prepare and run the toy-MC experiments

HybridResult* HybridCalculatorOriginal::Calculate(RooAbsData& data, unsigned int nToys, bool usePriors) const
{

   double testStatData = 0;
   if ( fTestStatisticsIdx==2 ) {
      /// number of events used as test statistics
      double nEvents = data.sumEntries();
      testStatData = nEvents;
   } else if ( fTestStatisticsIdx==3 ) {
      /// profiled likelihood ratio used as test statistics
      if ( fTmpDoExtended ) {
   RooNLLVar sb_nll("sb_nll","sb_nll",*fSbModel,data,RooFit::CloneData(false),RooFit::Extended());
   fSbModel->fitTo(data,RooFit::Hesse(false),RooFit::Strategy(0),RooFit::Extended());
   double sb_nll_val = sb_nll.getVal();
   RooNLLVar b_nll("b_nll","b_nll",*fBModel,data,RooFit::CloneData(false),RooFit::Extended());
   fBModel->fitTo(data,RooFit::Hesse(false),RooFit::Strategy(0),RooFit::Extended());
   double b_nll_val = b_nll.getVal();
   double m2lnQ = 2*(sb_nll_val-b_nll_val);
   testStatData = m2lnQ;
      } else {
   RooNLLVar sb_nll("sb_nll","sb_nll",*fSbModel,data,RooFit::CloneData(false));
   fSbModel->fitTo(data,RooFit::Hesse(false),RooFit::Strategy(0));
   double sb_nll_val = sb_nll.getVal();
   RooNLLVar b_nll("b_nll","b_nll",*fBModel,data,RooFit::CloneData(false));
   fBModel->fitTo(data,RooFit::Hesse(false),RooFit::Strategy(0));
   double b_nll_val = b_nll.getVal();
   double m2lnQ = 2*(sb_nll_val-b_nll_val);
   testStatData = m2lnQ;
      }
    } else if ( fTestStatisticsIdx==1 ) {
      /// likelihood ratio used as test statistics (default)
      if ( fTmpDoExtended ) {
   RooNLLVar sb_nll("sb_nll","sb_nll",*fSbModel,data,RooFit::Extended());
   RooNLLVar b_nll("b_nll","b_nll",*fBModel,data,RooFit::Extended());
   double m2lnQ = 2*(sb_nll.getVal()-b_nll.getVal());
   testStatData = m2lnQ;
      } else {
   RooNLLVar sb_nll("sb_nll","sb_nll",*fSbModel,data);
   RooNLLVar b_nll("b_nll","b_nll",*fBModel,data);
   double m2lnQ = 2*(sb_nll.getVal()-b_nll.getVal());
   testStatData = m2lnQ;
      }
   }

   std::cout << "Test statistics has been evaluated for data\n";

   HybridResult* result = Calculate(nToys,usePriors);
   result->SetDataTestStatistics(testStatData);

   return result;
}

////////////////////////////////////////////////////////////////////////////////

HybridResult* HybridCalculatorOriginal::Calculate(unsigned int nToys, bool usePriors) const
{
   std::vector<double> bVals;
   bVals.reserve(nToys);

   std::vector<double> sbVals;
   sbVals.reserve(nToys);

   RunToys(bVals,sbVals,nToys,usePriors);

   HybridResult* result;

   TString name = "HybridResult_" + TString(GetName() );

   if ( fTestStatisticsIdx==2 )
     result = new HybridResult(name,sbVals,bVals,false);
   else
     result = new HybridResult(name,sbVals,bVals);

   return result;
}

////////////////////////////////////////////////////////////////////////////////
/// do the actual run-MC processing

void HybridCalculatorOriginal::RunToys(std::vector<double>& bVals, std::vector<double>& sbVals, unsigned int nToys, bool usePriors) const
{
   std::cout << "HybridCalculatorOriginal: run " << nToys << " toy-MC experiments\n";
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

   // create a cloned list of all parameters need in case of test statistics 3 where those
   // changed by the best fit
   RooArgSet  originalSbParams;
   RooArgSet  originalBParams;
   if (fTestStatisticsIdx == 3) {
      RooArgSet * sbparams = fSbModel->getParameters(*fObservables);
      RooArgSet * bparams = fBModel->getParameters(*fObservables);
      if (sbparams) originalSbParams.addClone(*sbparams);
      if (bparams) originalBParams.addClone(*bparams);
      delete sbparams;
      delete bparams;
//       originalSbParams.Print("V");
//       originalBParams.Print("V");
   }


   for (unsigned int iToy=0; iToy<nToys; iToy++) {

      /// prints a progress report every 500 iterations
      /// TO DO: add a global verbose flag
     if ( /*verbose && */ iToy%500==0 ) {
           std::cout << "....... toy number " << iToy << " / " << nToys << std::endl;
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
      else {
   if ( fTmpDoExtended ) bData = static_cast<RooAbsData*> (fBModel->generate(*fObservables,RooFit::Extended()));
   else bData = static_cast<RooAbsData*> (fBModel->generate(*fObservables,1));
      }

      /// work-around in case of an empty dataset (TO DO: need a debug in RooFit?)
      bool bIsEmpty = false;
      if (bData==nullptr) {
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
      else {
   if ( fTmpDoExtended ) sbData = static_cast<RooAbsData*> (fSbModel->generate(*fObservables,RooFit::Extended()));
   else sbData = static_cast<RooAbsData*> (fSbModel->generate(*fObservables,1));
      }

      /// work-around in case of an empty dataset (TO DO: need a debug in RooFit?)
      bool sbIsEmpty = false;
      if (sbData==nullptr) {
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

      // test first the S+B hypothesis and the the B-only hypothesis
      for (int hypoTested=0; hypoTested<=1; hypoTested++) {
   RooAbsData* dataToTest = sbData;
   bool dataIsEmpty = sbIsEmpty;
   if ( hypoTested==1 ) { dataToTest = bData; dataIsEmpty = bIsEmpty; }
   /// evaluate the test statistic in the tested hypothesis case
   if ( fTestStatisticsIdx==2 ) {  /// number of events used as test statistics
     double nEvents = 0;
     if ( !dataIsEmpty ) nEvents = dataToTest->numEntries();
     if ( hypoTested==0 ) sbVals.push_back(nEvents);
     else bVals.push_back(nEvents);
   } else if ( fTestStatisticsIdx==3 ) {  /// profiled likelihood ratio used as test statistics
     if ( fTmpDoExtended ) {
       RooNLLVar sb_nll("sb_nll","sb_nll",*fSbModel,*dataToTest,RooFit::CloneData(false),RooFit::Extended());
       fSbModel->fitTo(*dataToTest,RooFit::PrintLevel(-1), RooFit::Hesse(false),RooFit::Strategy(0),RooFit::Extended());
       double sb_nll_val = sb_nll.getVal();
       RooNLLVar b_nll("b_nll","b_nll",*fBModel,*dataToTest,RooFit::CloneData(false),RooFit::Extended());
       fBModel->fitTo(*dataToTest,RooFit::PrintLevel(-1),RooFit::Hesse(false),RooFit::Strategy(0),RooFit::Extended());
       double b_nll_val = b_nll.getVal();
       double m2lnQ = -2*(b_nll_val-sb_nll_val);
       if ( hypoTested==0 ) sbVals.push_back(m2lnQ);
       else bVals.push_back(m2lnQ);
     } else {
       RooNLLVar sb_nll("sb_nll","sb_nll",*fSbModel,*dataToTest,RooFit::CloneData(false));
       fSbModel->fitTo(*dataToTest,RooFit::PrintLevel(-1), RooFit::Hesse(false),RooFit::Strategy(0));
       double sb_nll_val = sb_nll.getVal();
       RooNLLVar b_nll("b_nll","b_nll",*fBModel,*dataToTest,RooFit::CloneData(false));
       fBModel->fitTo(*dataToTest,RooFit::PrintLevel(-1), RooFit::Hesse(false),RooFit::Strategy(0));
       double b_nll_val = b_nll.getVal();
       double m2lnQ = -2*(b_nll_val-sb_nll_val);
       if ( hypoTested==0 ) sbVals.push_back(m2lnQ);
       else bVals.push_back(m2lnQ);
     }
   } else if ( fTestStatisticsIdx==1 ) {  /// likelihood ratio used as test statistics (default)
     if ( fTmpDoExtended ) {
       RooNLLVar sb_nll("sb_nll","sb_nll",*fSbModel,*dataToTest,RooFit::CloneData(false),RooFit::Extended());
       RooNLLVar b_nll("b_nll","b_nll",*fBModel,*dataToTest,RooFit::CloneData(false),RooFit::Extended());
       double m2lnQ = -2*(b_nll.getVal()-sb_nll.getVal());
       if ( hypoTested==0 ) sbVals.push_back(m2lnQ);
       else bVals.push_back(m2lnQ);
     } else {
       RooNLLVar sb_nll("sb_nll","sb_nll",*fSbModel,*dataToTest,RooFit::CloneData(false));
       RooNLLVar b_nll("b_nll","b_nll",*fBModel,*dataToTest,RooFit::CloneData(false));
       double m2lnQ = -2*(b_nll.getVal()-sb_nll.getVal());
       if ( hypoTested==0 ) sbVals.push_back(m2lnQ);
       else bVals.push_back(m2lnQ);
     }
   }
      }  // tested both hypotheses

      /// delete the toy-MC datasets
      delete sbData;
      delete bData;

      /// restore the parameters to their initial values in case fitting is done
      if (fTestStatisticsIdx == 3) {
         RooArgSet * sbparams = fSbModel->getParameters(*fObservables);
         if (sbparams) {
            assert(originalSbParams.getSize() == sbparams->getSize());
            sbparams->assign(originalSbParams);
            delete sbparams;
         }
         RooArgSet * bparams = fBModel->getParameters(*fObservables);
         if (bparams) {
            assert(originalBParams.getSize() == bparams->getSize());
            bparams->assign(originalBParams);
            delete bparams;
         }
      }



   } /// end of loop over toy-MC experiments


   /// restore the parameters to their initial values
   if (usePriors && nParameters>0) {
      for (int iParameter=0; iParameter<nParameters; iParameter++) {
         RooRealVar* oneParam = (RooRealVar*) parametersList.at(iParameter);
         oneParam->setVal(parameterValues[iParameter]);
      }
   }

   return;
}

////////////////////////////////////////////////////////////////////////////////
/// Print out some information about the input models

void HybridCalculatorOriginal::PrintMore(const char* options) const
{

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

////////////////////////////////////////////////////////////////////////////////
/// implementation of inherited methods from HypoTestCalculator

HybridResult* HybridCalculatorOriginal::GetHypoTest() const {
   // perform the hypothesis test and return result of hypothesis test

   // check first that everything needed is there
   if (!DoCheckInputs()) return 0;
   RooAbsData * treeData = dynamic_cast<RooAbsData *> (fData);
   if (!treeData) {
      std::cerr << "Error in HybridCalculatorOriginal::GetHypoTest - invalid data type - return nullptr" << std::endl;
      return 0;
   }
   bool usePrior = (fUsePriorPdf && fPriorPdf );
   return Calculate( *treeData, fNToys, usePrior);
}


bool HybridCalculatorOriginal::DoCheckInputs() const {
   if (!fData) {
      std::cerr << "Error in HybridCalculatorOriginal - data have not been set" << std::endl;
      return false;
   }

   // if observable have not been set take them from data
   if (!fObservables && fData->get() ) fObservables =  new RooArgList( *fData->get() );
   if (!fObservables) {
      std::cerr << "Error in HybridCalculatorOriginal - no observables" << std::endl;
      return false;
   }

   if (!fSbModel) {
      std::cerr << "Error in HybridCalculatorOriginal - S+B pdf has not been set " << std::endl;
      return false;
   }

   if (!fBModel) {
      std::cerr << "Error in HybridCalculatorOriginal - B pdf has not been set" << std::endl;
      return false;
   }
   if (fUsePriorPdf && !fNuisanceParameters) {
      std::cerr << "Error in HybridCalculatorOriginal - nuisance parameters have not been set " << std::endl;
      return false;
   }
   if (fUsePriorPdf && !fPriorPdf) {
      std::cerr << "Error in HybridCalculatorOriginal - prior pdf has not been set " << std::endl;
      return false;
   }
   return true;
}
