// @(#)root/roostats:$Id$
// Author: Kyle Cranmer   28/07/2008

/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef RooStats_NumberCountingUtils
#define RooStats_NumberCountingUtils

/////////////////////////////////////////
// NumberCountingUtils
//
// Encapsulates common number counting utilities
/////////////////////////////////////////
#include "Rtypes.h"

namespace RooStats{

   namespace  NumberCountingUtils {

      ///////////////////////////////////
      // Standalone Functions.
      // Naming conventions:
      //  Exp = Expected
      //  Obs = Observed
      //  P   = p-value
      //  Z   = Z-value or significance in Sigma (one-sided convention)
      //////////////////////////////////

  
      Double_t BinomialExpZ(Double_t, Double_t, Double_t);
      Double_t BinomialWithTauExpZ(Double_t, Double_t, Double_t);   
      Double_t BinomialObsZ(Double_t, Double_t, Double_t);
      Double_t BinomialWithTauObsZ(Double_t, Double_t, Double_t);

      Double_t BinomialExpP(Double_t, Double_t, Double_t);
      Double_t BinomialWithTauExpP(Double_t, Double_t, Double_t);
      Double_t BinomialObsP(Double_t, Double_t, Double_t);
      Double_t BinomialWithTauObsP(Double_t, Double_t, Double_t);
      

   }
}

#endif
