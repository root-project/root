// @(#)root/base:$Id$
// Author: Philippe Canal   09/30/2011

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TThreadSlots
#define ROOT_TThreadSlots

#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif

namespace ROOT {

   enum EThreadSlotReservation {
      // Describe the thread local storage array in TThread::Tsd
      //
      // Slot 0 through 4 are reserved for the global system
      // Slot 5 through 24 can be used for user application
      
      // Slot reserved by ROOT's packages.
      kDirectoryThreadSlot = 0,
      kPadThreadSlot       = 1,
      kClassThreadSlot     = 2,
      kFileThreadSlot      = 3,
      kPerfStatsThreadSlot = 4,

      kMaxThreadSlot       = 25,  // Size of the array of thread local slots in TThread
      kMinUserThreadSlot   = 5,
      kMaxUserThreadSlot   = kMaxThreadSlot
   };
}

// This macro assumes that the first position in the TLS array corresponds
// to the current directory and initialises it to gROOT.
// The rest of the pointers in the array are initialised to null
#define TTHREAD_INIT_TLS_ARRAY {gROOT}

#ifndef __CINT__
R__EXTERN void **(*gThreadTsd)(void*,Int_t);
#endif

#endif // ROOT_TThreadSlots
