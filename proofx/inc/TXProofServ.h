// @(#)root/proofx:$Name:  $:$Id: TXProofServ.h,v 1.3 2006/04/19 10:57:44 rdm Exp $
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

   Bool_t        fTerminated; //true if Terminate() has been already called

   void          SendLogFile(Int_t status = 0, Int_t start = -1, Int_t end = -1);
   void          Setup();

public:
   TXProofServ(Int_t *argc, char **argv) : TProofServ(argc, argv)
                 { fInterruptHandler = 0; fInputHandler = 0; fTerminated = kFALSE;}
   virtual ~TXProofServ();

   void          CreateServer();

   // Disable / Enable read timeout
   void          DisableTimeout();
   void          EnableTimeout();

   EQueryAction  GetWorkers(TList *workers, Int_t &prioritychange);

   Bool_t        HandleError(); // Error Handler
   Bool_t        HandleInput(); // Input handler

   void          HandleUrgentData();
   void          HandleSigPipe();
   void          HandleTermination();

   void          Terminate(Int_t status);

   ClassDef(TXProofServ,0)  //XRD PROOF Server Application Interface
};

#endif
