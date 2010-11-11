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
     }
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
       if(fProfile) delete fProfile;
       if(fNll) delete fNll;
       if(fCachedBestFitParams) delete fCachedBestFitParams;
     }
    
     // Main interface to evaluate the test statistic on a dataset
     virtual Double_t Evaluate(RooAbsData& data, RooArgSet& paramsOfInterest) {
       if (!&data) {
	 cout << "problem with data" << endl;
	 return 0 ;
       }
       
       RooFit::MsgLevel msglevel = RooMsgService::instance().globalKillBelow();
       RooMsgService::instance().setGlobalKillBelow(RooFit::FATAL);


       // simple
       RooAbsReal* nll = fPdf->createNLL(data, RooFit::CloneData(kFALSE));
       RooAbsReal* profile = nll->createProfile(paramsOfInterest);
       // make sure we set the variables attached to this nll
       RooArgSet* attachedSet = nll->getVariables();
       *attachedSet = paramsOfInterest;
       double ret = profile->getVal();
       //       paramsOfInterest.Print("v");
       delete attachedSet;
       delete nll;
       nll = 0; 
       delete profile;
       RooMsgService::instance().setGlobalKillBelow(msglevel);
       //       cout << "ret = " << ret << endl;


       // return here and forget about the following code
       return ret;

       // OLD version with some handling for local minima
       // (not used right now)

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
               RooMinuit minuit(*nll);
               minuit.setPrintLevel(-999);
               minuit.setNoWarn();
               minuit.migrad();

               // store the best fit values for future use
               fCachedBestFitParams = (RooArgSet*) (nll->getParameters(data)->snapshot());

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
      RooProfileLL* fProfile;
      RooAbsPdf* fPdf;
      RooNLLVar* fNll;
      const RooArgSet* fCachedBestFitParams;
      RooAbsData* fLastData;
      //      Double_t fLastMLE;

   protected:
      ClassDef(ProfileLikelihoodTestStat,1)   // implements the profile likelihood ratio as a test statistic to be used with several tools
   };
}


#endif
