// @(#)root/proofx:$Id$
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

#include "TProofServ.h"
#include "TXHandler.h"

class TXProofServInterruptHandler;
class TXSocketHandler;

class TXProofServ : public TProofServ, public TXHandler {

private:
   TXProofServInterruptHandler *fInterruptHandler;
   TXSocketHandler             *fInputHandler;
   TString                      fSockPath;

   Bool_t        fTerminated; //true if Terminate() has been already called

   Int_t         LockSession(const char *sessiontag, TProofLockPath **lck);

   Int_t         Setup() override;

public:
   TXProofServ(Int_t *argc, char **argv, FILE *flog = 0);
   ~TXProofServ() override;

   Int_t         CreateServer() override;

   // Disable / Enable read timeout
   void          DisableTimeout() override;
   void          EnableTimeout() override;

   EQueryAction  GetWorkers(TList *workers, Int_t &prioritychange,
                            Bool_t resume = kFALSE) override;

   Bool_t        HandleError(const void *in = 0) override; // Error Handler
   Bool_t        HandleInput(const void *in = 0) override; // Input handler

   void          HandleUrgentData() override;
   void          HandleSigPipe() override;
   void          HandleTermination() override;

   void          ReleaseWorker(const char *ord) override;
   void          Terminate(Int_t status) override;

   ClassDefOverride(TXProofServ,0)  //XRD PROOF Server Application Interface
};

#endif
