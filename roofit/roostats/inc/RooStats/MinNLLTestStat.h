// @(#)root/roostats:$Id: MinNLLTestStat.h 43035 2012-02-16 16:48:57Z sven $
// Author: Kyle Cranmer, Lorenzo Moneta, Gregory Schott, Wouter Verkerke
// Additional Contributions: Giovanni Petrucciani 
/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOSTATS_MinNLLTestStat
#define ROOSTATS_MinNLLTestStat

//_________________________________________________
/*
BEGIN_HTML
<p>
MinNLLTestStat is an implementation of the TestStatistic interface that calculates the minimum value of the negative log likelihood
function and returns it as a test statistic.
Internaly it operates by delegating to a MinNLLTestStat object.
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
#include "ProfileLikelihoodTestStat.h"

namespace RooStats {

  class MinNLLTestStat : public TestStatistic{

   public:
     MinNLLTestStat() {
        // Proof constructor. Do not use.
	proflts = 0;
     }
     MinNLLTestStat(RooAbsPdf& pdf) {
	proflts = new ProfileLikelihoodTestStat(pdf);
     }

     virtual ~MinNLLTestStat() {
	delete proflts;
     }

     void SetOneSided(Bool_t flag=true) {proflts->SetOneSided(flag);}
     void SetOneSidedDiscovery(Bool_t flag=true) {proflts->SetOneSidedDiscovery(flag);}
     void SetReuseNLL(Bool_t flag) { proflts->SetReuseNLL(flag); }
     void SetMinimizer(const char* minimizer){ proflts->SetMinimizer(minimizer); }
     void SetStrategy(Int_t strategy){ proflts->SetStrategy(strategy); }
     void SetTolerance(double tol){ proflts->SetTolerance(tol); }
     void SetPrintLevel(Int_t printlevel){ proflts->SetPrintLevel(printlevel); }
    
     // Main interface to evaluate the test statistic on a dataset
     virtual Double_t Evaluate(RooAbsData& data, RooArgSet& paramsOfInterest) {
        return proflts->EvaluateProfileLikelihood(1, data, paramsOfInterest); //find unconditional NLL minimum
     }

     virtual void EnableDetailedOutput( bool e=true ) { proflts->EnableDetailedOutput(e); }

     virtual const RooArgSet* GetDetailedOutput(void) const {
	     // Returns detailed output. The value returned by this function is updated after each call to Evaluate().
	     // The returned RooArgSet contains the following:
	     // <ul>
	     // <li> the minimum nll, fitstatus and convergence quality for each fit </li> 
	     // <li> for all non-constant parameters their value, error and pull </li>
	     // </ul>
	     return proflts->GetDetailedOutput();
     }
    
     virtual void SetVarName(const char* name) { proflts->SetVarName(name); }

     virtual const TString GetVarName() const { return proflts->GetVarName(); }

   private:
     ProfileLikelihoodTestStat* proflts;

   protected:
      ClassDef(MinNLLTestStat,1)   // implements the minimum NLL as a test statistic to be used with several tools
   };
}


#endif
