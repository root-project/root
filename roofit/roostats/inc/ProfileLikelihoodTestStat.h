// @(#)root/roostats:$Id: ProfileLikelihoodTestStat.h 26805 2009-01-13 17:45:57Z cranmer $
// Author: Kyle Cranmer, Lorenzo Moneta, Gregory Schott, Wouter Verkerke
/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOSTATS_ProfileLikelihoodTestStat
#define ROOSTATS_ProfileLikelihoodTestStat

//_________________________________________________
/*
BEGIN_HTML
<p>
ProfileLikelihoodTestStat is an implementation of the TestStatistic interface that calculates the profile
likelihood ratio at a particular parameter point given a dataset.  It does not constitute a statistical test, for that one may either use:
<ul>
 <li> the ProfileLikelihoodCalculator that relies on asymptotic properties of the Profile Likelihood Ratio</li>
 <li> the Neyman Construction classes with this class as a test statistic</li>
 <li> the Hybrid Calculator class with this class as a test statistic</li>
</ul>

</p>
END_HTML
*/
//

#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif

#include <vector>

//#include "RooStats/DistributionCreator.h"
#include "RooStats/SamplingDistribution.h"
#include "RooStats/TestStatistic.h"

#include "RooRealVar.h"
#include "RooProfileLL.h"
#include "RooNLLVar.h"

#include "RooMinuit.h"

namespace RooStats {

  class ProfileLikelihoodTestStat : public TestStatistic{

   public:
     ProfileLikelihoodTestStat(RooAbsPdf& pdf) {
       fPdf = &pdf;
       fProfile = 0;
       fNll = 0;
       fCachedBestFitParams = 0;
       fLastData = 0;
     }
     virtual ~ProfileLikelihoodTestStat() {
       //       delete fRand;
       //       delete fTestStatistic;
     }
    
     // Main interface to evaluate the test statistic on a dataset
     virtual Double_t Evaluate(RooAbsData& data, RooArgSet& paramsOfInterest)  {       
       if(!&data){ cout << "problem with data" << endl;}
       
       RooMsgService::instance().setGlobalKillBelow(RooMsgService::FATAL) ;
       bool needToRebuild = true; // try to avoid rebuilding if possible

       if(fLastData == &data) // simple pointer comparison for now (note NLL makes COPY of data)
	 needToRebuild=false;
       else
	 fLastData = &data; // keep a copy of pointer to original data

       // pointer comparison causing problems.  See multiple datasets with same value of pointer
       // but actually a new dataset
       needToRebuild = true; 

       // check mem leak in NLL or Profile. Should remove.
       // if(fProfile) needToRebuild = false; 


       if(needToRebuild){
	 if(fProfile) delete fProfile; 
	 if (fNll)    delete fNll;

	 RooNLLVar* nll = new RooNLLVar("nll","",*fPdf,data, RooFit::Extended());
	 fNll = nll;
	 fProfile = new RooProfileLL("pll","",*nll, paramsOfInterest);


	 // set parameters to previous best fit params, to speed convergence
	 // and to avoid local minima
	 if(fCachedBestFitParams){
	   // store original values, since minimization will change them.
	   RooArgSet* origParamVals = (RooArgSet*) paramsOfInterest.snapshot();

	   // these parameters are not guaranteed to be the best for this data	   
	   SetParameters(fCachedBestFitParams, fProfile->getParameters(data) );
	   // now evaluate to force this profile to evaluate and store
	   // best fit parameters for this data
	   fProfile->getVal();

	   // possibly store last MLE for reference
	   //	 Double mle = fNll->getVal();

	   // restore parameters
	   SetParameters(origParamVals, &paramsOfInterest );
	   
	   // cleanup
	   delete origParamVals;

	 } else {
	   
	   // store best fit parameters
	   // RooProfileLL::bestFitParams returns best fit of nuisance parameters only
	   //	   fCachedBestFitParams = (RooArgSet*) (fProfile->bestFitParams().clone("lastBestFit"));
	   // ProfileLL::getParameters returns current value of the parameters
	   //	   fCachedBestFitParams = (RooArgSet*) (fProfile->getParameters(data)->clone("lastBestFit"));
	   //cout << "making fCachedBestFitParams: " << fCachedBestFitParams << fCachedBestFitParams->getSize() << endl;

	   // store original values, since minimization will change them.
	   RooArgSet* origParamVals = (RooArgSet*) paramsOfInterest.snapshot();

	   // find minimum
	   RooMinuit minuit(*nll);
	   minuit.setPrintLevel(-999);
	   minuit.setNoWarn();
	   minuit.migrad();

	   // store the best fit values for future use
	   fCachedBestFitParams = (RooArgSet*) (nll->getParameters(data)->snapshot());

	   // restore parameters
	   SetParameters(origParamVals, &paramsOfInterest );

	   // evaluate to force this profile to evaluate and store
	   // best fit parameters for this data
	   fProfile->getVal();

	   // cleanup
	   delete origParamVals;

	 }

       }
       // issue warning if problems
       if(!fProfile){ cout << "problem making profile" << endl;}

       // set parameters to point being requested
       SetParameters(&paramsOfInterest, fProfile->getParameters(data) );

       Double_t value = fProfile->getVal();
       RooMsgService::instance().setGlobalKillBelow(RooMsgService::DEBUG) ;

       /*
       // for debugging caching
       cout << "current value of input params: " << endl;
       paramsOfInterest.Print("verbose");

       cout << "current value of params in profile: " << endl;
       fProfile->getParameters(data)->Print("verbose");

       cout << "cached last best fit: " << endl;
       fCachedBestFitParams->Print("verbose");
       */     

       // catch false minimum 
       if(value<0){
	 //	 cout << "ProfileLikelihoodTestStat: problem that profileLL = " << value 
	 //	      << " < 0, indicates false min.  Try again."<<endl;
	 delete fNll;
	 delete fProfile;
	 RooNLLVar* nll = new RooNLLVar("nll","",*fPdf,data, RooFit::Extended());
	 fNll = nll;
	 fProfile = new RooProfileLL("pll","",*nll, paramsOfInterest);

	 // set parameters to point being requested
	 SetParameters(&paramsOfInterest, fProfile->getParameters(data) );

	 value = fProfile->getVal();
	 //	 cout << "now profileLL = " << value << endl;
       }
       return value;
     }

      // Get the TestStatistic
      virtual const RooAbsArg* GetTestStatistic()  const {return fProfile;}  
    
      
   private:
      RooProfileLL* fProfile;
      RooAbsPdf* fPdf;
      RooNLLVar* fNll;
      const RooArgSet* fCachedBestFitParams;
      RooAbsData* fLastData;
      //      Double_t fLastMLE;

   protected:
      ClassDef(ProfileLikelihoodTestStat,1)   
   };

}


#endif
