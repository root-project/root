// @(#)root/roostats:$Id: TestStatistic.h 26805 2009-02-19 10:00:00 pellicci $
// Author: Kyle Cranmer, Lorenzo Moneta, Gregory Schott, Wouter Verkerke
/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOSTATS_TestStatistic
#define ROOSTATS_TestStatistic

//_________________________________________________
/*
BEGIN_HTML
<p>
TestStatistic is an interface class to provide a facility for construction test statistics
distributions to the NeymanConstruction class. All the actual samplers inherit from this class.
</p>
END_HTML
*/
//

#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif

namespace RooStats {

  class TestStatistic {

  public:
    // TestStatistic();
    virtual ~TestStatistic() {}

      // Main interface to evaluate the test statistic on a dataset
    virtual Double_t Evaluate(RooAbsData& data, RooArgSet& paramsOfInterest) = 0;

    virtual const RooAbsArg* GetTestStatistic()  const = 0;  

   protected:
      ClassDef(TestStatistic,1)   // Interface for tools setting limits (producing confidence intervals)
   };
}


#endif
