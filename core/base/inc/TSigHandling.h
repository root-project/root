// @(#)root/base:$Id$
// Author: Fons Rademakers   15/09/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TSigHandling
#define ROOT_TSigHandling


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TSigHandling                                                         //
//                                                                      //
// Abstract base class defining a generic interface to the underlying   //
// Operating System.                                                    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <ctype.h>
#include <fcntl.h>
#ifndef WIN32
#include <unistd.h>
#endif

#include "TInetAddress.h"
#include "TNamed.h"
#include "ThreadLocalStorage.h"
#include "TString.h"
#include "TTimer.h"

class TSeqCollection;
class TFdSet;
class TVirtualMutex;

typedef void ((*Func_t)());

R__EXTERN const char  *gRootDir;
R__EXTERN TVirtualMutex *gSystemMutex;

class TSigHandling : public TNamed {

protected:
   TFdSet          *fSignals;          //!Signals that were trapped
   Int_t            fSigcnt;           //Number of pending signals
   TSeqCollection  *fSignalHandler;    //List of signal handlers

public:
   TSigHandling(const char *name = "Generic", const char *title = "Generic Signal Handling");
   virtual ~TSigHandling();

   //---- Misc
   virtual void            Init();
   virtual void            StackTrace();

   //---- Handling of system signals
   virtual Bool_t          HaveTrappedSignal(Bool_t pendingOnly);
   virtual void            DispatchSignals(ESignals sig); 
   virtual void            AddSignalHandler(TSignalHandler *sh);
   virtual TSignalHandler *RemoveSignalHandler(TSignalHandler *sh);
   virtual void            ResetSignal(ESignals sig, Bool_t reset = kTRUE);
   virtual void            ResetSignals();
   virtual void            IgnoreSignal(ESignals sig, Bool_t ignore = kTRUE);
   virtual void            IgnoreInterrupt(Bool_t ignore = kTRUE);
   virtual TSeqCollection *GetListOfSignalHandlers();
   virtual void            SigAlarmInterruptsSyscalls(Bool_t) { }

   ClassDef(TSigHandling,0)
};

R__EXTERN TSigHandling *gSigHandling;

#endif
