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



#include "Rtypes.h"


#include "RooRealVar.h"

#include "RooAbsData.h"

#include "RooAbsPdf.h"

#include "RooStats/TestStatistic.h"


//#include "RooStats/DistributionCreator.h"


namespace RooStats {

   /**

      NumEventsTestStat is a simple implementation of the TestStatistic interface used for simple number counting.
      It should probably support simple cuts as well.

      \ingroup Roostats
   */

  class NumEventsTestStat : public TestStatistic{

   public:
     NumEventsTestStat() : fPdf(0) { }
     NumEventsTestStat(RooAbsPdf& pdf) {
       fPdf = &pdf;
     }
     ~NumEventsTestStat() override {
       //       delete fRand;
       //       delete fTestStatistic;
     }

     // Main interface to evaluate the test statistic on a dataset
     double Evaluate(RooAbsData& data, RooArgSet& /*paramsOfInterest*/) override  {

         if(data.isWeighted()) {
            return data.sumEntries();
         }

         // if no pdf is given in the constructor, we assume by default it can be extended
         if (!fPdf || fPdf->canBeExtended()) {
            return data.numEntries();
         }

         // data is not weighted as pdf cannot be extended
         if(data.numEntries() == 1) {
            double numEvents = 0.0;
            for (auto const *obs : static_range_cast<RooRealVar *>(*data.get(0))) {
               numEvents += obs->getValV();
            }
            return numEvents;
         }

         std::cout << "Data set is invalid" << std::endl;
         return 0;
     }

      // Get the TestStatistic
      virtual const RooAbsArg* GetTestStatistic()  const {return fPdf;}

      const TString GetVarName() const override {return "Number of events";}


   private:
      RooAbsPdf* fPdf;

   protected:
      ClassDefOverride(NumEventsTestStat,1)
   };

}


#endif
