// @(#)root/netx:$Name:  $:$Id: TNetFile.h,v 1.16 2004/08/09 17:43:07 rdm Exp $
// Author: Alvise Dorigo, Fabrizio Furano

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TXDebug                                                              //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TXDebug.h"
#include "TError.h"
#include "TSystem.h"

TXDebug *TXDebug::fgInstance = 0;

//_____________________________________________________________________________
TXDebug* TXDebug::Instance()
{
   // Create unique instance

   if (!fgInstance) {
      fgInstance = new TXDebug;
      if (!fgInstance) {
         Error("TXDebug::Instance", "Fatal ERROR *** Object creation with new"
               " failed ! Probable system resources exhausted.");
         gSystem->Abort();
      }
   }
   return fgInstance;
}

//_____________________________________________________________________________
TXDebug::TXDebug()
{
   // Constructor

   fDbgLevel = gEnv->GetValue("XNet.Debug", 0);
}

//_____________________________________________________________________________
TXDebug::~TXDebug()
{
   // Destructor

   SafeDelete(fgInstance);
}
