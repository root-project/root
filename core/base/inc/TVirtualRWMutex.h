// @(#)root/base:$Id$
// Author: Philippe Canal, 2017

/*************************************************************************
 * Copyright (C) 1995-2017, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TVirtualRWMutex
#define ROOT_TVirtualRWMutex


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TVirtualRWMutex                                                      //
//                                                                      //
// This class implements a read-write mutex interface. The actual work  //
// is done via TRWSpinLock which is available as soon as the thread     //
// library is loaded.                                                   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TVirtualMutex.h"


class TVirtualRWMutex : public TVirtualMutex  {

public:
   virtual void ReadLock() = 0;
   virtual void ReadUnLock() = 0;
   virtual void WriteLock() = 0;
   virtual void WriteUnLock() = 0;

   Int_t Lock() override { WriteLock(); return 1; }
   Int_t TryLock() override { WriteLock(); return 1; }
   Int_t UnLock() override { WriteUnLock(); return 1; }
   Int_t CleanUp() override { WriteUnLock(); return 1; }

   TVirtualRWMutex *Factory(Bool_t /*recursive*/ = kFALSE) override = 0;

   ClassDefOverride(TVirtualRWMutex, 0)  // Virtual mutex lock class
};



#endif
