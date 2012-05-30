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


#ifndef ROO_REAL_VAR
#include "RooRealVar.h"
#endif

#ifndef ROO_ABS_DATA
#include "RooAbsData.h"
#endif

#ifndef ROO_ABS_PDF
#include "RooAbsPdf.h"
#endif

#ifndef ROOSTATS_TestStatistic
#include "RooStats/TestStatistic.h"
#endif


//#include "RooStats/DistributionCreator.h"


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
            std::cout << "Data set reference is NULL" << std::endl;
            return 0;
         }

         if(data.isWeighted()) {
            return data.sumEntries();
         }
 
         // if no pdf is given in the constructor, we assume by default it can be extended
         if (!fPdf || fPdf->canBeExtended()) {
            return data.numEntries();
         } 
          
         // data is not weighted as pdf cannot be extended 
         if(data.numEntries() == 1) { 

            const RooArgSet *obsSet = data.get(0);
            RooLinkedListIter iter = obsSet->iterator();

            RooRealVar *obs = NULL; Double_t numEvents = 0.0;
            while((obs = (RooRealVar *)iter.Next()) != NULL) {
               numEvents += obs->getValV();
            }
            return numEvents;
         }

         std::cout << "Data set is invalid" << std::endl;
         return 0;
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
