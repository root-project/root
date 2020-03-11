// @(#)root/minuit2:$Id$
// Authors: M. Winkler, F. James, L. Moneta, A. Zsenei   2003-2005

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 LCG ROOT Math team,  CERN/PH-SFT                *
 *                                                                    *
 **********************************************************************/

#include "Minuit2/MinimumBuilder.h"
#include "Minuit2/MnPrint.h"


namespace ROOT {

   namespace Minuit2 {

      MinimumBuilder::MinimumBuilder() :
         fPrintLevel(MnPrint::Level()),
         fStorageLevel(1),
         fTracer(0)
      {}

      MinimumBuilder::BuilderPrintLevelConf::BuilderPrintLevelConf(int printLevel)
      {
         // set global printlevel to be same as local
         //std::cout << "set global print level to " << printLevel << std::endl;
         fPrevGlobLevel = MnPrint::SetLevel(printLevel);
      }
      MinimumBuilder::BuilderPrintLevelConf::~BuilderPrintLevelConf()
      {
         //std::cout << "reset global print level to " << fPrevGlobLevel << std::endl;
         MnPrint::SetLevel(fPrevGlobLevel);
      }

   }  // namespace Minuit2

}  // namespace ROOT
