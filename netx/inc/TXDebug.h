// Author: Alvise Dorigo, Fabrizio Furano

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
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
// Authors: Alvise Dorigo, Fabrizio Furano                              //
//          INFN Padova, 2003                                           //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TEnv.h"

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
