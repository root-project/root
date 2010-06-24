// @(#)root/roostats:$Id$
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
      //TestStatistic();
      virtual ~TestStatistic() {
      }

      // Main interface to evaluate the test statistic on a dataset given the
      // values for the Null Parameters Of Interest.
      virtual Double_t Evaluate(RooAbsData& data, RooArgSet& nullPOI) = 0;

      virtual const TString GetVarName() const = 0;

      // Defines the sign convention of the test statistic. Overwrite function if necessary.
      virtual  bool PValueIsRightTail(void) const { return true; }

   protected:
   ClassDef(TestStatistic,1) // Interface for a TestStatistic
};

} // end namespace RooStats


#endif
