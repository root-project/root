// @(#)root/unix:$Id$
// Author: Zhe Zhang   10/03/16

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TUnixSigHandling
#define ROOT_TUnixSigHandling


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TUnixSigHandling                                                     //
//                                                                      //
// Class providing an interface to the UNIX Operating System.           //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TSigHandling
#include "TSigHandling.h"
#endif
#ifndef ROOT_TSysEvtHandler
#include "TSysEvtHandler.h"
#endif
#ifndef ROOT_TTimer
#include "TTimer.h"
#endif

typedef void (*SigHandler_t)(ESignals);


class TUnixSigHandling : public TSigHandling {

protected:
   //---- Unix signal interface functions ----------------------
   static void         UnixSignal(ESignals sig, SigHandler_t h);
   static const char  *UnixSigname(ESignals sig);
   static void         UnixSigAlarmInterruptsSyscalls(Bool_t set);
   static void         UnixResetSignal(ESignals sig);
   static void         UnixResetSignals();
   static void         UnixIgnoreSignal(ESignals sig, Bool_t ignore);
   static void         UnixSetDefaultSignals();

   //---- Unix stack trace helper functions ---------------------
   static void         StackTraceHelperInit();
   static void         StackTraceMonitorThread();
   static void         StackTraceTriggerThread();
   static void         StackTraceForkThread();
   static int          StackTraceExecScript(void *);
   
public:
   TUnixSigHandling();
   virtual ~TUnixSigHandling();

   //---- Misc -------------------------------------------------
   void               Init();
   
   //---- Handling of system events ----------------------------
   Bool_t             CheckSignals(Bool_t sync);
   Bool_t             HaveTrappedSignal(Bool_t pendingOnly);
   void               DispatchSignals(ESignals sig);
   void               AddSignalHandler(TSignalHandler *sh);
   TSignalHandler    *RemoveSignalHandler(TSignalHandler *sh);
   void               ResetSignal(ESignals sig, Bool_t reset = kTRUE);
   void               ResetSignals();
   void               IgnoreSignal(ESignals sig, Bool_t ignore = kTRUE);
   void               SigAlarmInterruptsSyscalls(Bool_t set);

   //---- Processes --------------------------------------------
   void               StackTrace();

   ClassDef(TUnixSigHandling,0)  //Interface to Unix Signal Handling
};

#endif
