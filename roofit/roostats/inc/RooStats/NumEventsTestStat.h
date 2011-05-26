// @(#)root/roostats:$Id$
// Author: Kyle Cranmer, Lorenzo Moneta, Gregory Schott, Wouter Verkerke
/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOSTATS_NumEventsTestStat
#define ROOSTATS_NumEventsTestStat

//_________________________________________________
/*
BEGIN_HTML
<p>
NumEventsTestStat is a simple implementation of the TestStatistic interface used for simple number counting.
It should probably support simple cuts as well.
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

namespace RooStats {

  class NumEventsTestStat : public TestStatistic{

   public:
     NumEventsTestStat() : fPdf(0) { }
     NumEventsTestStat(RooAbsPdf& pdf) {
       fPdf = &pdf;
     }
     virtual ~NumEventsTestStat() {
       //       delete fRand;
       //       delete fTestStatistic;
     }
    
     // Main interface to evaluate the test statistic on a dataset
     virtual Double_t Evaluate(RooAbsData& data, RooArgSet& /*paramsOfInterest*/)  {       
       if(!&data) { 
	 cout << "problem with data" << endl;
	 return 0 ;
       } else {
       
	 RooMsgService::instance().setGlobalKillBelow(RooFit::FATAL) ;	 
	 return data.numEntries();
       }
     }

      // Get the TestStatistic
      virtual const RooAbsArg* GetTestStatistic()  const {return fPdf;}  

      virtual const TString GetVarName() const {return "Number of events";}
    
      
   private:
      RooAbsPdf* fPdf;

   protected:
      ClassDef(NumEventsTestStat,1)   
   };

}


#endif
