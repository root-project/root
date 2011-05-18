// @(#)root/proofx:$Id$
// Author: G. Ganis Oct 2005

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TXSlave
#define ROOT_TXSlave


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TXSlave                                                              //
//                                                                      //
// This is the version of TSlave for slave servers based on XRD.        //
// See TSlave for details.                                              //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TSlave
#include "TSlave.h"
#endif
#ifndef ROOT_TXHandler
#include "TXHandler.h"
#endif

class TObjString;
class TSocket;
class TSignalHandler;

class TXSlave : public TSlave, public TXHandler {

friend class TProof;
friend class TXProofMgr;

private:
   Bool_t   fValid;
   TSignalHandler *fIntHandler;     //interrupt signal handler (ctrl-c)

   void  Init(const char *host, Int_t stype);

   // Static methods
   static Int_t GetProofdProtocol(TSocket *s);

protected:
   void     FlushSocket();
   void     Interrupt(Int_t type);
   Int_t    Ping();
   TObjString *SendCoordinator(Int_t kind, const char *msg = 0, Int_t int2 = 0);
   Int_t    SendGroupPriority(const char *grp, Int_t priority);
   void     SetAlias(const char *alias);
   void     StopProcess(Bool_t abort, Int_t timeout);

public:
   TXSlave(const char *url, const char *ord, Int_t perf,
           const char *image, TProof *proof, Int_t stype,
           const char *workdir, const char *msd);
   virtual ~TXSlave();

   void   Close(Option_t *opt = "");
   void   DoError(int level, const char *location, const char *fmt,
                  va_list va) const;

   Bool_t HandleError(const void *in = 0); // Error Handler
   Bool_t HandleInput(const void *in = 0); // Input handler

   void   SetInterruptHandler(Bool_t on = kTRUE);

   Int_t  SetupServ(Int_t stype, const char *conffile);

   void   Touch();

   ClassDef(TXSlave,0)  //Xrd PROOF slave server
};

#endif
