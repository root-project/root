// @(#)root/netx:$Name:  $:$Id: TNetFile.h,v 1.16 2004/08/09 17:43:07 rdm Exp $
// Author: Alvise Dorigo, Fabrizio Furano

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TXDebug
#define ROOT_TXDebug

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TXDebug                                                              //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TEnv
#include "TEnv.h"
#endif

#define DebugLevel() TXDebug::Instance()->GetDebugLevel()

class TXDebug {

protected:
   TXDebug();
   ~TXDebug();

public:

   enum {
      kNODEBUG   = 0,
      kUSERDEBUG = 1,
      kHIDEBUG   = 2,
      kDUMPDEBUG = 3
   };

   Short_t         fDbgLevel;
   static TXDebug *fgInstance;

   Short_t         GetDebugLevel(void) { return fDbgLevel; }
   static TXDebug* Instance();
};

#endif
