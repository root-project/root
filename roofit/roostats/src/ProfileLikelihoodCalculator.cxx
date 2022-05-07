// @(#)root/roostats:$Id$
// Author: Kyle Cranmer   28/07/2008

/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class RooStats::ProfileLikelihoodCalculator
    \ingroup Roostats

The ProfileLikelihoodCalculator is a concrete implementation of CombinedCalculator
(the interface class for tools which can produce both a RooStats HypoTestResult
and ConfInterval). The tool uses the profile likelihood ratio as a test statistic,
and assumes that Wilks' theorem is valid. Wilks' theorem states that \f$ -2 \cdot \ln(\lambda) \f$
(profile likelihood ratio) is asymptotically distributed as a \f$ \chi^2 \f$ distribution
with \f$ N \f$ degrees of freedom. Thus, \f$p\f$-values can be
constructed, and the profile likelihood ratio can be used to construct a
LikelihoodInterval. (In the future, this class could be extended to use toy
Monte Carlo to calibrate the distribution of the test statistic).

Usage: It uses the interface of the CombinedCalculator, so that it can be
configured by specifying:

  - A model common model (*e.g.* a family of specific models, which includes both
    the null and alternate)
  - A data set
  - A set of parameters of interest. The nuisance parameters will be all other
    parameters of the model.
  - A set of parameters which specify the null hypothesis (including values
    and const/non-const status).

The interface allows one to pass the model, data, and parameters either directly
or via a ModelConfig class. The alternate hypothesis leaves the parameter free
to take any value other than those specified by the null hypothesis. There is
therefore no need to specify the alternate parameters.

After configuring the calculator, one only needs to call GetHypoTest() (which
will return a HypoTestResult pointer) or GetInterval() (which will return a
ConfInterval pointer).

This calculator can work with both one-dimensional intervals or multi-
dimensional ones (contours).

Note that for hypothesis tests, it is often better to use the
AsymptoticCalculator, which can compute in addition the expected
\f$p\f$-value using an Asimov data set.

*/

#include "RooStats/ProfileLikelihoodCalculator.h"

#include "RooStats/RooStatsUtils.h"

#include "RooStats/LikelihoodInterval.h"
#include "RooStats/HypoTestResult.h"

#include "RooFitResult.h"
#include "RooRealVar.h"
#include "RooProfileLL.h"
#include "RooNLLVar.h"
#include "RooGlobalFunc.h"
#include "RooMsgService.h"

#include "Math/MinimizerOptions.h"
#include "RooMinimizer.h"
//#include "RooProdPdf.h"

using namespace std;

ClassImp(RooStats::ProfileLikelihoodCalculator); ;

using namespace RooFit;
using namespace RooStats;


////////////////////////////////////////////////////////////////////////////////
/// default constructor

ProfileLikelihoodCalculator::ProfileLikelihoodCalculator() :
   CombinedCalculator(), fFitResult(0), fGlobalFitDone(false)
{
}

ProfileLikelihoodCalculator::ProfileLikelihoodCalculator(RooAbsData& data, RooAbsPdf& pdf, const RooArgSet& paramsOfInterest,
                                                         Double_t size, const RooArgSet* nullParams ) :
   CombinedCalculator(data,pdf, paramsOfInterest, size, nullParams ),
   fFitResult(0), fGlobalFitDone(false)
{
   // constructor from pdf and parameters
   // the pdf must contain eventually the nuisance parameters
}

ProfileLikelihoodCalculator::ProfileLikelihoodCalculator(RooAbsData& data,  ModelConfig& model, Double_t size) :
   CombinedCalculator(data, model, size),
   fFitResult(0), fGlobalFitDone(false)
{
   // construct from a ModelConfig. Assume data model.GetPdf() will provide full description of model including
   // constraint term on the nuisances parameters
   assert(model.GetPdf() );
}


////////////////////////////////////////////////////////////////////////////////
/// destructor
/// cannot delete prod pdf because it will delete all the composing pdf's
///    if (fOwnPdf) delete fPdf;
///    fPdf = 0;

ProfileLikelihoodCalculator::~ProfileLikelihoodCalculator(){
   if (fFitResult) delete fFitResult;
}

void ProfileLikelihoodCalculator::DoReset() const {
   // reset and clear fit result
   // to be called when a new model or data are set in the calculator
   if (fFitResult) delete fFitResult;
   fFitResult = 0;
}

RooAbsReal *  ProfileLikelihoodCalculator::DoGlobalFit() const {
   // perform a global fit of the likelihood letting with all parameter of interest and
   // nuisance parameters
   // keep the list of fitted parameters

   DoReset();
   RooAbsPdf * pdf = GetPdf();
   RooAbsData* data = GetData();
   if (!data || !pdf ) return 0;

   // get all non-const parameters
   RooArgSet* constrainedParams = pdf->getParameters(*data);
   if (!constrainedParams) return 0;
   RemoveConstantParameters(constrainedParams);

   const auto& config = GetGlobalRooStatsConfig();
   RooAbsReal * nll = pdf->createNLL(*data, CloneData(true), Constrain(*constrainedParams),ConditionalObservables(fConditionalObs), GlobalObservables(fGlobalObs),
       RooFit::Offset(config.useLikelihoodOffset) );

   // check if global fit has been already done
   if (fFitResult && fGlobalFitDone) {
      delete constrainedParams;
      return nll;
   }

      // calculate MLE
   oocoutP(nullptr,Minimization) << "ProfileLikelihoodCalcultor::DoGLobalFit - find MLE " << std::endl;

   if (fFitResult) delete fFitResult;
   fFitResult = DoMinimizeNLL(nll);

   // print fit result
   if (fFitResult) {
      fFitResult->printStream( oocoutI(nullptr,Minimization), fFitResult->defaultPrintContents(0), fFitResult->defaultPrintStyle(0) );

      if (fFitResult->status() != 0)
         oocoutW(nullptr,Minimization) << "ProfileLikelihoodCalcultor::DoGlobalFit -  Global fit failed - status = " << fFitResult->status() << std::endl;
      else
         fGlobalFitDone = true;
   }

   delete constrainedParams;
   return nll;
}

RooFitResult * ProfileLikelihoodCalculator::DoMinimizeNLL(RooAbsReal * nll)  {
   // Minimizer the given NLL using the default options

   const char * minimType = ROOT::Math::MinimizerOptions::DefaultMinimizerType().c_str();
   const char * minimAlgo = ROOT::Math::MinimizerOptions::DefaultMinimizerAlgo().c_str();
   int strategy = ROOT::Math::MinimizerOptions::DefaultStrategy();
   int level = ROOT::Math::MinimizerOptions::DefaultPrintLevel() -1;// RooFit level starts from  -1
   int tolerance = ROOT::Math::MinimizerOptions::DefaultTolerance();
   oocoutP(nullptr,Minimization) << "ProfileLikelihoodCalcultor::DoMinimizeNLL - using " << minimType << " / " << minimAlgo << " with strategy " << strategy << std::endl;
   // do global fit and store fit result for further use

   const auto& config = GetGlobalRooStatsConfig();

   RooMinimizer minim(*nll);
   minim.setStrategy(strategy);
   minim.setEps(tolerance);
   minim.setPrintLevel(level);
   minim.optimizeConst(2); // to optimize likelihood calculations
   minim.setEvalErrorWall(config.useEvalErrorWall);

   int status = -1;
   for (int tries = 1, maxtries = 4; tries <= maxtries; ++tries) {
      status = minim.minimize(minimType,minimAlgo);
      if (status%1000 == 0) {  // ignore erros from Improve
         break;
      } else if (tries < maxtries) {
         cout << "    ----> Doing a re-scan first" << endl;
         minim.minimize(minimType,"Scan");
         if (tries == 2) {
            if (strategy == 0 ) {
               cout << "    ----> trying with strategy = 1" << endl;;
               minim.setStrategy(1);
            }
            else
               tries++; // skip this trial if strategy is already 1
         }
         if (tries == 3) {
            cout << "    ----> trying with improve" << endl;;
            minimType = "Minuit";
            minimAlgo = "migradimproved";
         }
      }
   }

   RooFitResult * result = minim.save();


   return result;
}

////////////////////////////////////////////////////////////////////////////////
/// Main interface to get a RooStats::ConfInterval.
/// It constructs a profile likelihood ratio, and uses that to construct a RooStats::LikelihoodInterval.

LikelihoodInterval* ProfileLikelihoodCalculator::GetInterval() const {
//    RooAbsPdf* pdf   = fWS->pdf(fPdfName);
//    RooAbsData* data = fWS->data(fDataName);
   RooAbsPdf * pdf = GetPdf();
   RooAbsData* data = GetData();
   if (!data || !pdf || fPOI.getSize() == 0) return 0;

   RooArgSet* constrainedParams = pdf->getParameters(*data);
   RemoveConstantParameters(constrainedParams);


   /*
   RooNLLVar* nll = new RooNLLVar("nll","",*pdf,*data, Extended(),Constrain(*constrainedParams));
   RooProfileLL* profile = new RooProfileLL("pll","",*nll, *fPOI);
   profile->addOwnedComponents(*nll) ;  // to avoid memory leak
   */

   // do a global fit cloning the data
   RooAbsReal * nll = DoGlobalFit();
   if (!nll) return 0;

   if (!fFitResult)   {
      delete nll;
      return 0;
   }

   RooAbsReal* profile = nll->createProfile(fPOI);
   profile->addOwnedComponents(*nll) ;  // to avoid memory leak

   // t.b.f. " RooProfileLL should keep and provide possibility to query on global minimum
   // set POI to fit value (this will speed up profileLL calculation of global minimum)
   const RooArgList & fitParams = fFitResult->floatParsFinal();
   for (int i = 0; i < fitParams.getSize(); ++i) {
      RooRealVar & fitPar =  (RooRealVar &) fitParams[i];
      RooRealVar * par = (RooRealVar*) fPOI.find( fitPar.GetName() );
      if (par) {
         par->setVal( fitPar.getVal() );
         par->setError( fitPar.getError() );
      }
   }

   // do this so profile will cache inside the absolute minimum and
   // minimum values of nuisance parameters
   // (no need to this here)
   // profile->getVal();
   //RooMsgService::instance().setGlobalKillBelow(RooFit::DEBUG) ;
   //  profile->Print();

   TString name = TString("LikelihoodInterval_");// + TString(GetName() );

   // make a list of fPOI with fit result values and pass to LikelihoodInterval class
   // bestPOI is a cloned list of POI only with their best fit values
   RooArgSet fitParSet(fitParams);
   RooArgSet * bestPOI = new RooArgSet();
   for (auto const *arg : fPOI){
      RooAbsArg * p  =  fitParSet.find( arg->GetName() );
      if (p) bestPOI->addClone(*p);
      else bestPOI->addClone(*arg);
   }
   // fPOI contains the parameter of interest of the PL object
   // and bestPOI contains a snapshot with the best fit values
   LikelihoodInterval* interval = new LikelihoodInterval(name, profile, &fPOI, bestPOI);
   interval->SetConfidenceLevel(1.-fSize);
   delete constrainedParams;
   return interval;
}

////////////////////////////////////////////////////////////////////////////////
/// Main interface to get a HypoTestResult.
/// It does two fits:
/// 1. The first lets the null parameters float, so it's a maximum likelihood estimate.
/// 2. The second is to the null model (fixing null parameters to their specified values): *e.g.* a conditional maximum likelihood.
/// Since not all parameters are floating, this likelihood will be lower than the unconditional model.
///
/// The ratio of the likelihood obtained from the conditional MLE to the MLE is the profile likelihood ratio.
/// Wilks' theorem is used to get \f$p\f$-values.

HypoTestResult* ProfileLikelihoodCalculator::GetHypoTest() const {
//    RooAbsPdf* pdf   = fWS->pdf(fPdfName);
//    RooAbsData* data = fWS->data(fDataName);
   RooAbsPdf * pdf = GetPdf();
   RooAbsData* data = GetData();


   if (!data || !pdf) return 0;

   if (fNullParams.getSize() == 0) return 0;

   // make a clone and ordered list since a vector will be associated to keep parameter values
   // clone the list since first fit will changes the fNullParams values
   RooArgList poiList;
   poiList.addClone(fNullParams); // make a clone list


   // do a global fit
   RooAbsReal * nll = DoGlobalFit();
   if (!nll) return 0;

   if (!fFitResult) {
      delete nll;
      return 0;
   }

   RooArgSet* constrainedParams = pdf->getParameters(*data);
   RemoveConstantParameters(constrainedParams);

   Double_t nLLatMLE = fFitResult->minNll();
   // in case of using offset need to save offset value
   Double_t nlloffset = (RooStats::IsNLLOffset() ) ? nll->getVal() - nLLatMLE : 0;

   // set POI to given values, set constant, calculate conditional MLE
   std::vector<double> oldValues(poiList.getSize() );
   for (unsigned int i = 0; i < oldValues.size(); ++i) {
      RooRealVar * mytarget = (RooRealVar*) constrainedParams->find(poiList[i].GetName());
      if (mytarget) {
         oldValues[i] = mytarget->getVal();
         mytarget->setVal( ( (RooRealVar&) poiList[i] ).getVal() );
         mytarget->setConstant(true);
      }
   }



   // perform the fit only if nuisance parameters are available
   // get nuisance parameters
   // nuisance parameters are the non const parameters from the likelihood parameters
   RooArgSet nuisParams(*constrainedParams);

   // need to remove the parameter of interest
   RemoveConstantParameters(&nuisParams);

   // check there are variable parameter in order to do a fit
   bool existVarParams = false;
   for (auto const *myarg : static_range_cast<RooRealVar *> (nuisParams)) {
      if ( !myarg->isConstant() ) {
         existVarParams = true;
         break;
      }
   }

   Double_t nLLatCondMLE = nLLatMLE;
   if (existVarParams) {
      oocoutP(nullptr,Minimization) << "ProfileLikelihoodCalcultor::GetHypoTest - do conditional fit " << std::endl;

      RooFitResult * fit2 = DoMinimizeNLL(nll);

      // print fit result
      if (fit2) {
         nLLatCondMLE = fit2->minNll();
         fit2->printStream( oocoutI(nullptr,Minimization), fit2->defaultPrintContents(0), fit2->defaultPrintStyle(0) );

         if (fit2->status() != 0)
            oocoutW(nullptr,Minimization) << "ProfileLikelihoodCalcultor::GetHypotest -  Conditional fit failed - status = " << fit2->status() << std::endl;
      }

   }
   else {
      // get just the likelihood value (no need to do a fit since the likelihood is a constant function)
      nLLatCondMLE = nll->getVal();
      // this value contains the offset
      if (RooStats::IsNLLOffset() ) nLLatCondMLE -= nlloffset;
   }

   // Use Wilks' theorem to translate -2 log lambda into a significance/p-value
   Double_t deltaNLL = std::max( nLLatCondMLE-nLLatMLE, 0.);

   // get number of free parameter of interest
   RemoveConstantParameters(poiList);
   int ndf = poiList.getSize();

   Double_t pvalue = ROOT::Math::chisquared_cdf_c( 2* deltaNLL, ndf);

   // in case of one dimension (1 poi) do the one-sided p-value (need to divide by 2)
   if (ndf == 1) pvalue = 0.5 * pvalue;

   TString name = TString("ProfileLRHypoTestResult_");// + TString(GetName() );
   HypoTestResult* htr = new HypoTestResult(name, pvalue, 0 );

   // restore previous value of poi
   for (unsigned int i = 0; i < oldValues.size(); ++i) {
      RooRealVar * mytarget = (RooRealVar*) constrainedParams->find(poiList[i].GetName());
      if (mytarget) {
         mytarget->setVal(oldValues[i] );
         mytarget->setConstant(false);
      }
   }

   delete constrainedParams;
   delete nll;
   return htr;

}
