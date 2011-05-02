// @(#)root/roostats:$Id$
// Author: Kyle Cranmer   28/07/2008

/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//_________________________________________________
/*
BEGIN_HTML
<p>
ProfileLikelihoodCalculator is a concrete implementation of CombinedCalculator 
(the interface class for a tools which can produce both RooStats HypoTestResults and ConfIntervals).  
The tool uses the profile likelihood ratio as a test statistic, and assumes that Wilks' theorem is valid.  
Wilks' theorem states that -2* log (profile likelihood ratio) is asymptotically distributed as a chi^2 distribution 
with N-dof, where N is the number of degrees of freedom.  Thus, p-values can be constructed and the profile likelihood ratio
can be used to construct a LikelihoodInterval.
(In the future, this class could be extended to use toy Monte Carlo to calibrate the distribution of the test statistic).
</p>
<p> Usage: It uses the interface of the CombinedCalculator, so that it can be configured by specifying:
<ul>
 <li>a model common model (eg. a family of specific models which includes both the null and alternate),</li>
 <li>a data set, </li>
 <li>a set of parameters of interest. The nuisance parameters will be all other parameters of the model </li>
 <li>a set of parameters of which specify the null hypothesis (including values and const/non-const status)  </li>
</ul>
The interface allows one to pass the model, data, and parameters either directly or via a ModelConfig class.
The alternate hypothesis leaves the parameter free to take any value other than those specified by the null hypotesis. There is therefore no need to 
specify the alternate parameters. 
</p>
<p>
After configuring the calculator, one only needs to ask GetHypoTest() (which will return a HypoTestResult pointer) or GetInterval() (which will return an ConfInterval pointer).
</p>
<p>
The concrete implementations of this interface should deal with the details of how the nuisance parameters are
dealt with (eg. integration vs. profiling) and which test-statistic is used (perhaps this should be added to the interface).
</p>
<p>
The motivation for this interface is that we hope to be able to specify the problem in a common way for several concrete calculators.
</p>
END_HTML
*/
//

#ifndef RooStats_ProfileLikelihoodCalculator
#include "RooStats/ProfileLikelihoodCalculator.h"
#endif

#ifndef RooStats_RooStatsUtils
#include "RooStats/RooStatsUtils.h"
#endif

#include "RooStats/LikelihoodInterval.h"
#include "RooStats/HypoTestResult.h"

#include "RooFitResult.h"
#include "RooRealVar.h"
#include "RooProfileLL.h"
#include "RooNLLVar.h"
#include "RooGlobalFunc.h"

#include "Math/MinimizerOptions.h"
//#include "RooProdPdf.h"

ClassImp(RooStats::ProfileLikelihoodCalculator) ;

using namespace RooFit;
using namespace RooStats;


//_______________________________________________________
ProfileLikelihoodCalculator::ProfileLikelihoodCalculator() : 
   CombinedCalculator(), fFitResult(0)
{
   // default constructor
}

ProfileLikelihoodCalculator::ProfileLikelihoodCalculator(RooAbsData& data, RooAbsPdf& pdf, const RooArgSet& paramsOfInterest, 
                                                         Double_t size, const RooArgSet* nullParams ) :
   CombinedCalculator(data,pdf, paramsOfInterest, size, nullParams ), 
   fFitResult(0)
{
   // constructor from pdf and parameters
   // the pdf must contain eventually the nuisance parameters
}

ProfileLikelihoodCalculator::ProfileLikelihoodCalculator(RooAbsData& data,  ModelConfig& model, Double_t size) :
   CombinedCalculator(data, model, size), 
   fFitResult(0)
{
   // construct from a ModelConfig. Assume data model.GetPdf() will provide full description of model including 
   // constraint term on the nuisances parameters
   assert(model.GetPdf() );
}


//_______________________________________________________
ProfileLikelihoodCalculator::~ProfileLikelihoodCalculator(){
   // destructor
   // cannot delete prod pdf because it will delete all the composing pdf's
//    if (fOwnPdf) delete fPdf; 
//    fPdf = 0; 
   if (fFitResult) delete fFitResult; 
}

void ProfileLikelihoodCalculator::DoReset() const { 
   // reset and clear fit result 
   // to be called when a new model or data are set in the calculator 
   if (fFitResult) delete fFitResult; 
   fFitResult = 0; 
}

void  ProfileLikelihoodCalculator::DoGlobalFit() const { 
   // perform a global fit of the likelihood letting with all parameter of interest and 
   // nuisance parameters 
   // keep the list of fitted parameters 

   DoReset(); 
   RooAbsPdf * pdf = GetPdf();
   RooAbsData* data = GetData(); 
   if (!data || !pdf ) return;

   // get all non-const parameters
   RooArgSet* constrainedParams = pdf->getParameters(*data);
   if (!constrainedParams) return ; 
   RemoveConstantParameters(constrainedParams);

   // calculate MLE 
   const char * minimType = ROOT::Math::MinimizerOptions::DefaultMinimizerType().c_str();
   const char * minimAlgo = ROOT::Math::MinimizerOptions::DefaultMinimizerAlgo().c_str();
   int strategy = ROOT::Math::MinimizerOptions::DefaultStrategy();
   int level = ROOT::Math::MinimizerOptions::DefaultPrintLevel() -1;// RooFit level starts from  -1
   ooccoutI((TObject*)0,Minimization) << "ProfileLikelihoodCalcultor::DoGlobalFit - using " << minimType << " / " << minimAlgo << " with strategy " << strategy << std::endl;
   RooFitResult* fit = pdf->fitTo(*data, Constrain(*constrainedParams),Strategy(strategy),PrintLevel(level),
                                  Hesse(kFALSE),Save(kTRUE),Minimizer(minimType,minimAlgo));
  
   // for debug 
   fit->Print();

   delete constrainedParams; 
   // store fit result for further use 
   fFitResult =  fit; 
   if (fFitResult == 0) 
      oocoutW((TObject*)0,Minimization) << "ProfileLikelihoodCalcultor::DoGlobalFit -  Global fit failed " << std::endl;      

}

//_______________________________________________________
LikelihoodInterval* ProfileLikelihoodCalculator::GetInterval() const {
   // Main interface to get a RooStats::ConfInterval.  
   // It constructs a profile likelihood ratio and uses that to construct a RooStats::LikelihoodInterval.

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

   RooAbsReal* nll = pdf->createNLL(*data, CloneData(kTRUE), Constrain(*constrainedParams));
   RooAbsReal* profile = nll->createProfile(fPOI);
   profile->addOwnedComponents(*nll) ;  // to avoid memory leak


   //RooMsgService::instance().setGlobalKillBelow(RooFit::FATAL) ;
   // perform a Best Fit 
   if (!fFitResult) DoGlobalFit();
   // if fit fails return
   if (!fFitResult)   return 0;

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
   TIter iter = fPOI.createIterator(); 
   RooArgSet fitParSet(fitParams); 
   RooArgSet * bestPOI = new RooArgSet();  
   while (RooAbsArg * arg =  (RooAbsArg*) iter.Next() ) { 
      RooAbsArg * p  =  fitParSet.find( arg->GetName() );
      if (p) bestPOI->addClone(*p);
      else bestPOI->addClone(*arg);
   }
   // fPOI contains the paramter of interest of the PL object 
   // and bestPOI contains a snapshot with the best fit values 
   LikelihoodInterval* interval = new LikelihoodInterval(name, profile, &fPOI, bestPOI);
   interval->SetConfidenceLevel(1.-fSize);
   delete constrainedParams;
   return interval;
}

//_______________________________________________________
HypoTestResult* ProfileLikelihoodCalculator::GetHypoTest() const {
   // Main interface to get a HypoTestResult.
   // It does two fits:
   // the first lets the null parameters float, so it's a maximum likelihood estimate
   // the second is to the null (fixing null parameters to their specified values): eg. a conditional maximum likelihood
   // the ratio of the likelihood at the conditional MLE to the MLE is the profile likelihood ratio.
   // Wilks' theorem is used to get p-values 

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
   if (!fFitResult) DoGlobalFit(); 
   if (!fFitResult) return 0; 

   RooArgSet* constrainedParams = pdf->getParameters(*data);
   RemoveConstantParameters(constrainedParams);

   // perform a global fit if it is not done before
   if (!fFitResult) DoGlobalFit(); 
   Double_t NLLatMLE= fFitResult->minNll();

   // set POI to given values, set constant, calculate conditional MLE
   std::vector<double> oldValues(poiList.getSize() ); 
   for (unsigned int i = 0; i < oldValues.size(); ++i) { 
      RooRealVar * mytarget = (RooRealVar*) constrainedParams->find(poiList[i].GetName());
      if (mytarget) { 
         oldValues[i] = mytarget->getVal(); 
         mytarget->setVal( ( (RooRealVar&) poiList[i] ).getVal() );
         mytarget->setConstant(kTRUE);
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
   TIter it = nuisParams.createIterator();
   RooRealVar * myarg = 0; 
   while ((myarg = (RooRealVar *)it.Next())) { 
      if ( !myarg->isConstant() ) {
         existVarParams = true; 
         break;
      }
   }

   Double_t NLLatCondMLE = NLLatMLE; 
   if (existVarParams) {

      const char * minimType = ROOT::Math::MinimizerOptions::DefaultMinimizerType().c_str();
      const char * minimAlgo = ROOT::Math::MinimizerOptions::DefaultMinimizerType().c_str();
      int level = ROOT::Math::MinimizerOptions::DefaultPrintLevel()-1; // RooFit levels starts from -1
      RooFitResult* fit2 = pdf->fitTo(*data,Constrain(*constrainedParams),Hesse(kFALSE),Strategy(0), Minos(kFALSE),
                                      Minimizer(minimType,minimAlgo), Save(kTRUE),PrintLevel(level));
     
      NLLatCondMLE = fit2->minNll();
      fit2->Print();
   }
   else { 
      // get just the likelihood value (no need to do a fit since the likelihood is a constant function)
      RooAbsReal* nll = pdf->createNLL(*data, CloneData(kTRUE), Constrain(*constrainedParams));
      NLLatCondMLE = nll->getVal();
      delete nll;
   }

   // Use Wilks' theorem to translate -2 log lambda into a signifcance/p-value
   Double_t deltaNLL = std::max( NLLatCondMLE-NLLatMLE, 0.);

   TString name = TString("ProfileLRHypoTestResult_");// + TString(GetName() ); 
   HypoTestResult* htr = 
      new HypoTestResult(name, SignificanceToPValue(sqrt( 2*deltaNLL)), 0 );


   // restore previous value of poi
   for (unsigned int i = 0; i < oldValues.size(); ++i) { 
      RooRealVar * mytarget = (RooRealVar*) constrainedParams->find(poiList[i].GetName());
      if (mytarget) { 
         mytarget->setVal(oldValues[i] ); 
         mytarget->setConstant(false); 
      }
   }

   delete constrainedParams;
   return htr;

}

