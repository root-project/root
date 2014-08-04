// @(#)root/minuit2:$Id$
// Authors: M. Winkler, F. James, L. Moneta, A. Zsenei   2003-2005

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 LCG ROOT Math team,  CERN/PH-SFT                *
 *                                                                    *
 **********************************************************************/

#include "Minuit2/MinimumBuilder.h"

#if defined(DEBUG) || defined(WARNINGMSG)
#include "Minuit2/MnPrint.h"
#endif


namespace ROOT {

   namespace Minuit2 {

      MinimumBuilder::MinimumBuilder() :
         fPrintLevel(MnPrint::Level()),
         fStorageLevel(1),
         fTracer(0)
      {}


   }  // namespace Minuit2

}  // namespace ROOT
