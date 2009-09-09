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
 <li>a set of parameters of which specify the null (including values and const/non-const status), </li>
 <li>a set of parameters of which specify the alternate (including values and const/non-const status),</li>
 <li>a set of parameters of nuisance parameters  (including values and const/non-const status).</li>
</ul>
The interface allows one to pass the model, data, and parameters via a workspace and then specify them with names.
The interface will be extended so that one does not need to use a workspace.
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

ClassImp(RooStats::ProfileLikelihoodCalculator) ;

using namespace RooFit;
using namespace RooStats;


//_______________________________________________________
ProfileLikelihoodCalculator::ProfileLikelihoodCalculator() : 
   CombinedCalculator() {
   // default constructor

}

ProfileLikelihoodCalculator::ProfileLikelihoodCalculator(RooWorkspace& ws, RooAbsData& data, RooAbsPdf& pdf, RooArgSet& paramsOfInterest, 
                                                         Double_t size, RooArgSet* nullParams, RooArgSet* altParams) :
   CombinedCalculator(ws,data,pdf, paramsOfInterest, size, nullParams, altParams)
{}

ProfileLikelihoodCalculator::ProfileLikelihoodCalculator(RooAbsData& data, RooAbsPdf& pdf, RooArgSet& paramsOfInterest, 
                                                         Double_t size, RooArgSet* nullParams, RooArgSet* altParams):
   CombinedCalculator(data,pdf, paramsOfInterest, size, nullParams, altParams)
{}


//_______________________________________________________
ProfileLikelihoodCalculator::~ProfileLikelihoodCalculator(){
   // destructor
}


//_______________________________________________________
ConfInterval* ProfileLikelihoodCalculator::GetInterval() const {
   // Main interface to get a RooStats::ConfInterval.  
   // It constructs a profile likelihood ratio and uses that to construct a RooStats::LikelihoodInterval.

   RooAbsPdf* pdf   = fWS->pdf(fPdfName);
   RooAbsData* data = fWS->data(fDataName);
   if (!data || !pdf || !fPOI) return 0;

   RooArgSet* constrainedParams = pdf->getParameters(*data);
   RemoveConstantParameters(constrainedParams);


   /*
   RooNLLVar* nll = new RooNLLVar("nll","",*pdf,*data, Extended(),Constrain(*constrainedParams));
   RooProfileLL* profile = new RooProfileLL("pll","",*nll, *fPOI);
   profile->addOwnedComponents(*nll) ;  // to avoid memory leak
   */

   RooAbsReal* nll = pdf->createNLL(*data, CloneData(kTRUE), Constrain(*constrainedParams));
   RooAbsReal* profile = nll->createProfile(*fPOI);
   profile->addOwnedComponents(*nll) ;  // to avoid memory leak


   RooMsgService::instance().setGlobalKillBelow(RooFit::FATAL) ;
   profile->getVal(); // do this so profile will cache the minimum
   RooMsgService::instance().setGlobalKillBelow(RooFit::DEBUG) ;

   LikelihoodInterval* interval 
      = new LikelihoodInterval("LikelihoodInterval", profile, fPOI);
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

   RooAbsPdf* pdf   = fWS->pdf(fPdfName);
   RooAbsData* data = fWS->data(fDataName);
   if (!data || !pdf) return 0;

   // calculate MLE
   RooArgSet* constrainedParams = pdf->getParameters(*data);
   RemoveConstantParameters(constrainedParams);

   RooFitResult* fit = pdf->fitTo(*data, Constrain(*constrainedParams),Strategy(0),Hesse(kFALSE),Save(kTRUE),PrintLevel(-1));
  

   fit->Print();
   Double_t NLLatMLE= fit->minNll();


   // set POI to null values, set constant, calculate conditional MLE
   TIter it = fNullParams->createIterator();
   RooRealVar *myarg; 
   RooRealVar *mytarget; 
   while ((myarg = (RooRealVar *)it.Next())) { 
      if(!myarg) continue;

      mytarget = fWS->var(myarg->GetName());
      if(!mytarget) continue;
      mytarget->setVal( myarg->getVal() );
      mytarget->setConstant(kTRUE);
      mytarget->Print();
   }

   // perform the fit only if nuisance parameters are available
   // get nuisance parameters
   RooArgSet nuisParams(*constrainedParams);
   // need to remove the parameter of interest
   nuisParams.remove(*fNullParams);
   // check there are variable parameter in order to do a fit 
   bool existVarParams = false; 
   TIter it2 = nuisParams.createIterator();
   while ((myarg = (RooRealVar *)it2.Next())) { 
      if ( !myarg->isConstant() ) {
         existVarParams = true; 
         break;
      }
   }

   Double_t NLLatCondMLE= NLLatMLE;
   if (existVarParams) {
      RooFitResult* fit2 = pdf->fitTo(*data,Constrain(*constrainedParams),Hesse(kFALSE),Strategy(0), Minos(kFALSE), Save(kTRUE),PrintLevel(-1));
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

   HypoTestResult* htr = 
      new HypoTestResult("ProfileLRHypoTestResult",
                         SignificanceToPValue(sqrt( 2*deltaNLL)), 0 );
   delete constrainedParams;
   return htr;

}

