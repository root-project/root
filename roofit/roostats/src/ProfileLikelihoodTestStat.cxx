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

#include "RooStats/ProfileLikelihoodTestStat.h"

Bool_t RooStats::ProfileLikelihoodTestStat::fgAlwaysReuseNll = kTRUE ;

Double_t RooStats::ProfileLikelihoodTestStat::EvaluateProfileLikelihood(int type, RooAbsData& data, RooArgSet& paramsOfInterest) {
        // interna function to evaluate test statistics
        // can do depending on type: 
        // type  = 0 standard evaluation, type = 1 find only unconditional NLL minimum, type = 2 conditional MLL

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
       }


       // make sure we set the variables attached to this nll
       RooArgSet* attachedSet = fNll->getVariables();

       *attachedSet = paramsOfInterest;
       RooArgSet* origAttachedSet = (RooArgSet*) attachedSet->snapshot();

       ///////////////////////////////////////////////////////////////////////
       // New profiling based on RooMinimizer (allows for Minuit2)
       // based on major speed increases seen by CMS for complex problems

 
       // other order
       // get the numerator
       RooArgSet* snap =  (RooArgSet*)paramsOfInterest.snapshot();

       tsw.Stop(); 
       double createTime = tsw.CpuTime();
       tsw.Start();

       // get the denominator
       double uncondML = 0;
       double fit_favored_mu = 0;
       int statusD = 0;
       if (type != 2) {
          uncondML = GetMinNLL(statusD);

          // get best fit value for one-sided interval 
          fit_favored_mu = attachedSet->getRealValue(firstPOI->GetName()) ;

       }
       tsw.Stop();
       double fitTime1  = tsw.CpuTime();
          
       double ret = 0; 
       int statusN = 0;
       tsw.Start();

       double condML = 0; 

       bool doConditionalFit = (type != 1); 

       // skip the conditional ML (the numerator) only when fit value is smaller than test value
       if (fOneSided &&  fit_favored_mu > initial_mu_value) { 
          doConditionalFit = false; 
          condML = uncondML;
       }

       if (doConditionalFit) {  


          //       cout <<" reestablish snapshot"<<endl;
          *attachedSet = *snap;

 
          // set the POI to constant
          RooLinkedListIter it = paramsOfInterest.iterator();
          RooRealVar* tmpPar = NULL, *tmpParA=NULL;
          while((tmpPar = (RooRealVar*)it.Next())){
             tmpParA =  dynamic_cast<RooRealVar*>( attachedSet->find(tmpPar->GetName()));
             if (tmpParA) tmpParA->setConstant();
          }


          // check if there are non-const parameters so it is worth to do the minimization
          RooArgSet allParams(*attachedSet); 
          RooStats::RemoveConstantParameters(&allParams);
          
          // in case no nuisance parameters are present
          // no need to minimize just evaluate the nll
          if (allParams.getSize() == 0 ) {
             condML = fNll->getVal(); 
          }
          else {              
             condML = GetMinNLL(statusN);
          }

       }

       tsw.Stop();
       double fitTime2 = tsw.CpuTime();

       if (fPrintLevel > 0) { 
          std::cout << "EvaluateProfileLikelihood - ";
          if (type <= 1)  
             std::cout << "mu hat = " << fit_favored_mu  <<  " uncond ML = " << uncondML; 
          if (type != 1) 
             std::cout << " cond ML = " << condML;
          if (type == 0)
             std::cout << " pll =  " << condML-uncondML; 
          std::cout << " time (create/fit1/2) " << createTime << " , " << fitTime1 << " , " << fitTime2  
                    << std::endl;
       }


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

       if (type == 1) return uncondML;
       if (type == 2) return condML;
       return condML-uncondML;
             
     }     

double RooStats::ProfileLikelihoodTestStat::GetMinNLL(int& status) {
   //find minimum of NLL using RooMinimizer

   RooMinimizer minim(*fNll);
   minim.setStrategy(fStrategy);
   //LM: RooMinimizer.setPrintLevel has +1 offset - so subtruct  here -1 + an extra -1 
   int level = (fPrintLevel == 0) ? -1 : fPrintLevel -2;
   minim.setPrintLevel(level);
   minim.setEps(fTolerance);
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
