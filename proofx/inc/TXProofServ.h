// @(#)root/proofx:$Name:  $:$Id: TXProofServ.h,v 1.2 2006/02/26 16:09:57 rdm Exp $
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

class TXProofServ : public TProofServ, public TXHandler {

private:
   void          SendLogFile(Int_t status = 0, Int_t start = -1, Int_t end = -1);
   void          Setup();

public:
   TXProofServ(Int_t *argc, char **argv);
   virtual ~TXProofServ();

   void          CreateServer();

   EQueryAction  GetWorkers(TList *workers, Int_t &prioritychange);

   Bool_t        HandleError(); // Error Handler
   Bool_t        HandleInput(); // Input handler

   void          HandleUrgentData();
   void          HandleSigPipe();

   ClassDef(TXProofServ,0)  //XRD PROOF Server Application Interface
};

#endif
