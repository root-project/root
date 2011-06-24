// @(#)root/roostats:$Id$
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

#include "RooStats/RooStatsUtils.h"

//#include "RooStats/DistributionCreator.h"
#include "RooStats/SamplingDistribution.h"
#include "RooStats/TestStatistic.h"

#include "RooStats/RooStatsUtils.h"

#include "RooRealVar.h"
#include "RooProfileLL.h"
#include "RooNLLVar.h"

#include "RooMinuit.h"

namespace RooStats {

  class ProfileLikelihoodTestStat : public TestStatistic{

   public:
     ProfileLikelihoodTestStat() {
        // Proof constructor. Do not use.
        fPdf = 0;
        fProfile = 0;
        fNll = 0;
        fCachedBestFitParams = 0;
        fLastData = 0;
	fOneSided = false;
     }
     ProfileLikelihoodTestStat(RooAbsPdf& pdf) {
       fPdf = &pdf;
       fProfile = 0;
       fNll = 0;
       fCachedBestFitParams = 0;
       fLastData = 0;
       fOneSided = false;
     }
     virtual ~ProfileLikelihoodTestStat() {
       //       delete fRand;
       //       delete fTestStatistic;
       if(fProfile) delete fProfile;
       if(fNll) delete fNll;
       if(fCachedBestFitParams) delete fCachedBestFitParams;
     }
     void SetOneSided(Bool_t flag=true) {fOneSided = flag;}

     static void setReuseNLL(Bool_t flag) { fReuseNll = flag ; }
    
     // Main interface to evaluate the test statistic on a dataset
     virtual Double_t Evaluate(RooAbsData& data, RooArgSet& paramsOfInterest) {
       if (!&data) {
	 cout << "problem with data" << endl;
	 return 0 ;
       }
       
       RooRealVar* firstPOI = (RooRealVar*) paramsOfInterest.first();
       double initial_mu_value  = firstPOI->getVal();
       //paramsOfInterest.getRealValue(firstPOI->GetName());

       RooFit::MsgLevel msglevel = RooMsgService::instance().globalKillBelow();
       RooMsgService::instance().setGlobalKillBelow(RooFit::FATAL);

       // simple
       Bool_t reuse=fReuseNll ;
       
       Bool_t created(kFALSE) ;
       if (!reuse || fNll==0) {
	 fNll = (RooNLLVar*) fPdf->createNLL(data, RooFit::CloneData(kFALSE));
	 fProfile = (RooProfileLL*) fNll->createProfile(paramsOfInterest);
	 created = kTRUE ;
	 //cout << "creating profile LL " << fNll << " " << fProfile << " data = " << &data << endl ;
       }
       if (reuse && !created) {
	 //cout << "reusing profile LL " << fNll << " new data = " << &data << endl ;
	 fNll->setData(data,kFALSE) ;
 	 if (fProfile) delete fProfile ; 
 	 fProfile = (RooProfileLL*) fNll->createProfile(paramsOfInterest) ; 
	 //fProfile->clearAbsMin() ;
       }


       // make sure we set the variables attached to this nll
       RooArgSet* attachedSet = fNll->getVariables();

       *attachedSet = paramsOfInterest;


       //       fPdf->setEvalErrorLoggingMode(RooAbsReal::CountErrors);
       //       profile->setEvalErrorLoggingMode(RooAbsReal::CountErrors);
       //       ((RooProfileLL*)profile)->nll().setEvalErrorLoggingMode(RooAbsReal::CountErrors);
       //       nll->setEvalErrorLoggingMode(RooAbsReal::CountErrors);
       //cout << "evaluating profile LL" << endl ;
       double ret = fProfile->getVal();
       //       cout << "profile value = " << ret << endl ;
       //       cout <<"eval errors pdf = "<<fPdf->numEvalErrors() << endl;
       //       cout <<"eval errors profile = "<<profile->numEvalErrors() << endl;
       //       cout <<"eval errors profile->nll = "<<((RooProfileLL*)profile)->nll().numEvalErrors() << endl;
       //       cout <<"eval errors nll = "<<nll->numEvalErrors() << endl;
       //       if(profile->numEvalErrors()>0)
       //       	 cout <<"eval errors = "<<profile->numEvalErrors() << endl;
       //       paramsOfInterest.Print("v");
       //       cout << "ret = " << ret << endl;

       if(fOneSided){
	 double fit_favored_mu = ((RooProfileLL*) fProfile)->bestFitObs().getRealValue(firstPOI->GetName()) ;
       
	 if( fit_favored_mu > initial_mu_value)
	   // cout <<"fit-favored_mu, initial value" << fit_favored_mu << " " << initial_mu_value<<endl;
	   ret = 0 ;
       }
       delete attachedSet;

       if (!reuse) {
	 //cout << "deleting ProfileLL " << fNll << " " << fProfile << endl ;
	 delete fNll;
	 fNll = 0; 
	 delete fProfile;
	 fProfile = 0 ;
       }

       RooMsgService::instance().setGlobalKillBelow(msglevel);

       //////////////////////////////////////////////////////
       // return here and forget about the following code
       return ret;
       


       //////////////////////////////////////////////////////////
       // OLD version with some handling for local minima
       // (not used right now)
       /////////////////////////////////////////////////////////


         bool needToRebuild = true; // try to avoid rebuilding if possible

         if (fLastData == &data) // simple pointer comparison for now (note NLL makes COPY of data)
            needToRebuild = false;
         else fLastData = &data; // keep a copy of pointer to original data

         // pointer comparison causing problems.  See multiple datasets with same value of pointer
         // but actually a new dataset
         needToRebuild = true;

         // check mem leak in NLL or Profile. Should remove.
         // if(fProfile) needToRebuild = false;


         if (needToRebuild) {
            if (fProfile) delete fProfile;
            if (fNll) delete fNll;

            /*
             RooNLLVar* nll = new RooNLLVar("nll","",*fPdf,data, RooFit::Extended());
             fNll = nll;
             fProfile = new RooProfileLL("pll","",*nll, paramsOfInterest);
             */
            RooArgSet* constrainedParams = fPdf->getParameters(data);
            RemoveConstantParameters(constrainedParams);
            //cout << "cons: " << endl;
            //constrainedParams->Print("v");

            RooNLLVar * nll2 = (RooNLLVar*) fPdf->createNLL(
               data, RooFit::CloneData(kFALSE), RooFit::Constrain(*constrainedParams)
            );
            fNll = nll2;
            fProfile = (RooProfileLL*) nll2->createProfile(paramsOfInterest);
            delete constrainedParams;

            //	 paramsOfInterest.Print("v");

            // set parameters to previous best fit params, to speed convergence
            // and to avoid local minima
            if (fCachedBestFitParams) {
               // store original values, since minimization will change them.
               RooArgSet* origParamVals = (RooArgSet*) paramsOfInterest.snapshot();

               // these parameters are not guaranteed to be the best for this data
               SetParameters(fCachedBestFitParams, fProfile->getParameters(data));
               // now evaluate to force this profile to evaluate and store
               // best fit parameters for this data
               fProfile->getVal();

               // possibly store last MLE for reference
               //	 Double mle = fNll->getVal();

               // restore parameters
               SetParameters(origParamVals, &paramsOfInterest);

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
               RooMinuit minuit(*fNll);
               minuit.setPrintLevel(-999);
               minuit.setNoWarn();
               minuit.migrad();

               // store the best fit values for future use
               fCachedBestFitParams = (RooArgSet*) (fNll->getParameters(data)->snapshot());

               // restore parameters
               SetParameters(origParamVals, &paramsOfInterest);

               // evaluate to force this profile to evaluate and store
               // best fit parameters for this data
               fProfile->getVal();

               // cleanup
               delete origParamVals;

            }

         }
         // issue warning if problems
         if (!fProfile) {
            cout << "problem making profile" << endl;
         }

         // set parameters to point being requested
         SetParameters(&paramsOfInterest, fProfile->getParameters(data));

         Double_t value = fProfile->getVal();

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
         if (value < 0) {
            //	 cout << "ProfileLikelihoodTestStat: problem that profileLL = " << value
            //	      << " < 0, indicates false min.  Try again."<<endl;
            delete fNll;
            delete fProfile;
            /*
             RooNLLVar* nll = new RooNLLVar("nll","",*fPdf,data, RooFit::Extended());
             fNll = nll;
             fProfile = new RooProfileLL("pll","",*nll, paramsOfInterest);
             */

            RooArgSet* constrainedParams = fPdf->getParameters(data);
            RemoveConstantParameters(constrainedParams);

            RooNLLVar * nll2 = (RooNLLVar*) fPdf->createNLL(data, RooFit::CloneData(kFALSE), RooFit::Constrain(
               *constrainedParams));
            fNll = nll2;
            fProfile = (RooProfileLL*) nll2->createProfile(paramsOfInterest);
            delete constrainedParams;

            // set parameters to point being requested
            SetParameters(&paramsOfInterest, fProfile->getParameters(data));

            value = fProfile->getVal();
            //cout << "now profileLL = " << value << endl;
         }
         //       cout << "now profileLL = " << value << endl;
         RooMsgService::instance().setGlobalKillBelow(RooFit::DEBUG);
         return value;


       
      }

    
      virtual const TString GetVarName() const {return "Profile Likelihood Ratio";}
      
      //      const bool PValueIsRightTail(void) { return false; } // overwrites default


   private:
      RooProfileLL* fProfile; //!
      RooAbsPdf* fPdf;
      RooNLLVar* fNll; //!
      const RooArgSet* fCachedBestFitParams;
      RooAbsData* fLastData;
      //      Double_t fLastMLE;
      Bool_t fOneSided;

      static Bool_t fReuseNll ;

   protected:
      ClassDef(ProfileLikelihoodTestStat,3)   // implements the profile likelihood ratio as a test statistic to be used with several tools
   };
}


#endif
