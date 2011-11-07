// @(#)root/roostats:$Id$
// Author: Kyle Cranmer, Lorenzo Moneta, Gregory Schott, Wouter Verkerke
// Additional Contributions: Giovanni Petrucciani 
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
#include "RooMsgService.h"

#include "RooMinuit.h"
#include "RooMinimizer.h"
#include "Math/MinimizerOptions.h"
#include "TStopwatch.h"

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
        fReuseNll = false;
	fMinimizer=::ROOT::Math::MinimizerOptions::DefaultMinimizerType().c_str();
	fStrategy=::ROOT::Math::MinimizerOptions::DefaultStrategy();
	fPrintLevel=::ROOT::Math::MinimizerOptions::DefaultPrintLevel();

     }
     ProfileLikelihoodTestStat(RooAbsPdf& pdf) {
       fPdf = &pdf;
       fProfile = 0;
       fNll = 0;
       fCachedBestFitParams = 0;
       fLastData = 0;
       fOneSided = false;
       fReuseNll = false;
       fMinimizer=::ROOT::Math::MinimizerOptions::DefaultMinimizerType().c_str();
       fStrategy=::ROOT::Math::MinimizerOptions::DefaultStrategy();
       fPrintLevel=::ROOT::Math::MinimizerOptions::DefaultPrintLevel();
     }
     virtual ~ProfileLikelihoodTestStat() {
       //       delete fRand;
       //       delete fTestStatistic;
       if(fProfile) delete fProfile;
       if(fNll) delete fNll;
       if(fCachedBestFitParams) delete fCachedBestFitParams;
     }
     void SetOneSided(Bool_t flag=true) {fOneSided = flag;}

     static void SetAlwaysReuseNLL(Bool_t flag) { fgAlwaysReuseNll = flag ; }
     void SetReuseNLL(Bool_t flag) { fReuseNll = flag ; }

     void SetMinimizer(const char* minimizer){ fMinimizer=minimizer;}
     void SetStrategy(Int_t strategy){fStrategy=strategy;}
     void SetPrintLevel(Int_t printlevel){fPrintLevel=printlevel;}
    
     // Main interface to evaluate the test statistic on a dataset
     virtual Double_t Evaluate(RooAbsData& data, RooArgSet& paramsOfInterest) {
       if (!&data) {
	 cout << "problem with data" << endl;
	 return 0 ;
       }

       //data.Print("V");
       
       TStopwatch tsw; 
       tsw.Start();

       RooRealVar* firstPOI = (RooRealVar*) paramsOfInterest.first();
       double initial_mu_value  = firstPOI->getVal();
       //paramsOfInterest.getRealValue(firstPOI->GetName());

       RooFit::MsgLevel msglevel = RooMsgService::instance().globalKillBelow();

       if (fPrintLevel < 3) RooMsgService::instance().setGlobalKillBelow(RooFit::FATAL);

       // simple
       Bool_t reuse=(fReuseNll || fgAlwaysReuseNll) ;
       
       Bool_t created(kFALSE) ;
       if (!reuse || fNll==0) {
          RooArgSet* allParams = fPdf->getParameters(data);
          RooStats::RemoveConstantParameters(allParams);

          // need to call constrain for RooSimultaneous until stripDisconnected problem fixed
          fNll = (RooNLLVar*) fPdf->createNLL(data, RooFit::CloneData(kFALSE),RooFit::Constrain(*allParams));

          //	 fNll = (RooNLLVar*) fPdf->createNLL(data, RooFit::CloneData(kFALSE));
          //	 fProfile = (RooProfileLL*) fNll->createProfile(paramsOfInterest);
          created = kTRUE ;
          delete allParams;
          //cout << "creating profile LL " << fNll << " " << fProfile << " data = " << &data << endl ;
       }
       if (reuse && !created) {
          //cout << "reusing profile LL " << fNll << " new data = " << &data << endl ;
          fNll->setData(data,kFALSE) ;
          // 	 if (fProfile) delete fProfile ; 
          // 	 fProfile = (RooProfileLL*) fNll->createProfile(paramsOfInterest) ; 
          //fProfile->clearAbsMin() ;
       }


       // make sure we set the variables attached to this nll
       RooArgSet* attachedSet = fNll->getVariables();

       *attachedSet = paramsOfInterest;
       RooArgSet* origAttachedSet = (RooArgSet*) attachedSet->snapshot();

       ///////////////////////////////////////////////////////////////////////
       // Main profiling version as of 5.30
       //       fPdf->setEvalErrorLoggingMode(RooAbsReal::CountErrors);
       //       profile->setEvalErrorLoggingMode(RooAbsReal::CountErrors);
       //       ((RooProfileLL*)profile)->nll().setEvalErrorLoggingMode(RooAbsReal::CountErrors);
       //       nll->setEvalErrorLoggingMode(RooAbsReal::CountErrors);
       //cout << "evaluating profile LL" << endl ;

       //      double ret = fProfile->getVal();  // previous main evaluation

       //       cout << "profile value = " << ret << endl ;
       //       cout <<"eval errors pdf = "<<fPdf->numEvalErrors() << endl;
       //       cout <<"eval errors profile = "<<profile->numEvalErrors() << endl;
       //       cout <<"eval errors profile->nll = "<<((RooProfileLL*)profile)->nll().numEvalErrors() << endl;
       //       cout <<"eval errors nll = "<<nll->numEvalErrors() << endl;
       //       if(profile->numEvalErrors()>0)
       //       	 cout <<"eval errors = "<<profile->numEvalErrors() << endl;
       //       paramsOfInterest.Print("v");
       //       cout << "ret = " << ret << endl;
       ///////////////////////////////////////////////////////////////////////


       ///////////////////////////////////////////////////////////////////////
       // New profiling based on RooMinimizer (allows for Minuit2)
       // based on major speed increases seen by CMS for complex problems

       /*
       // set the parameters of interest to be fixed for the conditional MLE
       TIterator* it = paramsOfInterest.createIterator();
       RooRealVar* tmpPar = NULL, *tmpParA=NULL;
       while((tmpPar = (RooRealVar*)it->Next())){
	 tmpParA =  ((RooRealVar*)attachedSet->find(tmpPar->GetName()));
	 tmpParA->setConstant();
       }
       
       //       cout <<"using Minimizer: "<< fMinimizer<< " with strategy = " << fStrategy << endl;

       // get the numerator
       int statusN;
       double condML = GetMinNLL(statusN);


       // set the parameters of interest back to floating
       it->Reset();
       while((tmpPar = (RooRealVar*)it->Next())){
	 tmpParA =  ((RooRealVar*)attachedSet->find(tmpPar->GetName()));
	 tmpParA->setConstant(false);
       }
       delete it;

       // get the denominator
       int statusD;
       double uncondML = GetMinNLL(statusD);
       */


       // other order
       // get the numerator
       RooArgSet* snap =  (RooArgSet*)paramsOfInterest.snapshot();

       tsw.Stop(); 
       double createTime = tsw.CpuTime();
       tsw.Start();

       // get the denominator
       int statusD;
       double uncondML = GetMinNLL(statusD);

       tsw.Stop();
       double fitTime1 = tsw.CpuTime();
       tsw.Start();

       // get best fit value for one-sided interval 
       double fit_favored_mu = attachedSet->getRealValue(firstPOI->GetName()) ;

       double ret = 0; 
       int statusN = 0;

       // do the conditional ML (the numerator) only when fit value is smaller than test value
       if (!fOneSided || fit_favored_mu <= initial_mu_value) {  

          //       cout <<" reestablish snapshot"<<endl;
          *attachedSet = *snap;

          // in case no nuisance parameters are present
          // no need to minimize just evaluate the nll
          double condML = 0; 

          // set the POI to constant
          RooLinkedListIter it = paramsOfInterest.iterator();
          RooRealVar* tmpPar = NULL, *tmpParA=NULL;
          while((tmpPar = (RooRealVar*)it.Next())){
             tmpParA =  ((RooRealVar*)attachedSet->find(tmpPar->GetName()));
             tmpParA->setConstant();
          }

          // check if there are non-const parameters so it is worth to do the minimization
          RooArgSet allParams(*attachedSet); 
          RooStats::RemoveConstantParameters(&allParams);
          
          if (allParams.getSize() == 0 ) {
             condML = fNll->getVal(); 
          }
          else { 
             condML = GetMinNLL(statusN);
          }

          ret=condML-uncondML;

//           if (ret < 0) { 
//              std::cout << " mu hat = " << fit_favored_mu << " cond ML = " << condML << " uncond ML = " << uncondML << std::endl;
//           }
       }

       tsw.Stop();
       double fitTime2 = tsw.CpuTime();
       //std::cout 
       //ooccoutD((TObject*)0,Eval) 
       if (fPrintLevel > 0) 
          std::cout 
             << "ProfileLikelihoodTestStat:Evaluate - mu hat = " << fit_favored_mu 
             <<  " uncond ML = " << uncondML << " cond ML = " << ret+uncondML 
             << " pll =  " << ret 
             << " time (create/fit1/2) " << createTime << " , " << fitTime1 << " , " << fitTime2  
             << std::endl;



       // need to restore the values ?
       *attachedSet = *origAttachedSet;

       delete attachedSet;
       delete origAttachedSet;
       delete snap;

       if (!reuse) {
	 delete fNll;
	 fNll = 0; 
	 //	 delete fProfile;
	 fProfile = 0 ;
       }

       RooMsgService::instance().setGlobalKillBelow(msglevel);

       if(statusN!=0 || statusD!=0)
	 ret= -1; // indicate failed fit

       return ret;
             
     }     



    
      virtual const TString GetVarName() const {return "Profile Likelihood Ratio";}
      
      //      const bool PValueIsRightTail(void) { return false; } // overwrites default

  private:
      double GetMinNLL(int& status) {

	RooMinimizer minim(*fNll);
	minim.setStrategy(fStrategy);
        //LM: RooMinimizer.setPrintLevel has +1 offset - so subtruct  here -1 + an extra -1 
        int level = (fPrintLevel == 0) ? -1 : fPrintLevel -2;
	minim.setPrintLevel(level);
        // this cayses a memory leak
	minim.optimizeConst(true); 
	for (int tries = 0, maxtries = 4; tries <= maxtries; ++tries) {
	  //	 status = minim.minimize(fMinimizer, ROOT::Math::MinimizerOptions::DefaultMinimizerAlgo().c_str());
	  status = minim.minimize(fMinimizer, "Minimize");
	  if (status == 0) {  
            break;
	  } else {
	    if (tries > 1) {
	      printf("    ----> Doing a re-scan first\n");
	      minim.minimize(fMinimizer,"Scan");
	    }
	    if (tries > 2) {
	      printf("    ----> trying with strategy = 1\n");
	     minim.setStrategy(1);
	    }
	  }
	}

        double val =  fNll->getVal();
        //minim.optimizeConst(false); 

	return val;
      }

   private:
      RooProfileLL* fProfile; //!
      RooAbsPdf* fPdf;
      RooNLLVar* fNll; //!
      const RooArgSet* fCachedBestFitParams;
      RooAbsData* fLastData;
      //      Double_t fLastMLE;
      Bool_t fOneSided;

      static Bool_t fgAlwaysReuseNll ;
      Bool_t fReuseNll ;
      TString fMinimizer;
      Int_t fStrategy;
      Int_t fPrintLevel;

   protected:
      ClassDef(ProfileLikelihoodTestStat,5)   // implements the profile likelihood ratio as a test statistic to be used with several tools
   };
}


#endif
