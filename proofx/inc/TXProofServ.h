// @(#)root/proofx:$Name:  $:$Id: TXProofServ.h,v 1.10 2006/11/20 10:31:48 rdm Exp $
// Author: G. Ganis Oct 2005

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TXProofServ
#define ROOT_TXProofServ

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TXProofServ                                                          //
//                                                                      //
// TXProofServ is the XRD version of the PROOF server. It differs from  //
// TProofServ only for the underlying connection technology             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TProofServ
#include "TProofServ.h"
#endif
#ifndef ROOT_TXHandler
#include "TXHandler.h"
#endif

class TXProofServInterruptHandler;
class TXSocketHandler;

class TXProofServ : public TProofServ, public TXHandler {

private:
   TXProofServInterruptHandler *fInterruptHandler;
   TXSocketHandler             *fInputHandler;
   TString                      fSockPath;

   Bool_t        fTerminated; //true if Terminate() has been already called

   Int_t         LockSession(const char *sessiontag, TProofLockPath **lck);

   void          SendLogFile(Int_t status = 0, Int_t start = -1, Int_t end = -1);
   void          SetShutdownTimer(Bool_t on = kTRUE, Int_t delay = 0);
   Int_t         Setup();

public:
   TXProofServ(Int_t *argc, char **argv, FILE *flog = 0);
   virtual ~TXProofServ();

   Int_t         CreateServer();

   // Disable / Enable read timeout
   void          DisableTimeout();
   void          EnableTimeout();

   EQueryAction  GetWorkers(TList *workers, Int_t &prioritychange);

   Bool_t        HandleError(const void *in = 0); // Error Handler
   Bool_t        HandleInput(const void *in = 0); // Input handler

   void          HandleUrgentData();
   void          HandleSigPipe();
   void          HandleTermination();

   void          Terminate(Int_t status);

   ClassDef(TXProofServ,0)  //XRD PROOF Server Application Interface
};

#endif
