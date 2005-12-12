// Author: G. Ganis Oct 2005

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
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

class TObjString;
class TSocket;

class TXSlave : public TSlave {

friend class TProof;

private:
   void  Init(const char *host, Int_t stype);

   // Static methods
   static Int_t GetProofdProtocol(TSocket *s);

protected:
   void     Interrupt(Int_t type);
   Int_t    Ping();
   TObjString *SendCoordinator(Int_t kind, const char *msg = 0);
   void     SetAlias(const char *alias);

public:
   TXSlave(const char *url, const char *ord, Int_t perf,
           const char *image, TProof *proof, Int_t stype,
           const char *workdir, const char *msd);
   virtual ~TXSlave();

   void   Close(Option_t *opt = "");
   void   SetupServ(Int_t stype, const char *conffile);

   ClassDef(TXSlave,0)  //Xrd PROOF slave server
};

#endif
