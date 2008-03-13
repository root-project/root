// @(#)root/clarens:$Id$
// Author: Maarten Ballintijn    21/10/2004
// Author: Kris Gulbrandsen      21/10/2004

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TLM
#define ROOT_TLM

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TLM                                                                  //
//                                                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TObject
#include "TObject.h"
#endif

#ifndef ROOT_TClProxy
#include "TClProxy.h"
#endif

#ifndef ROOT_TString
#include "TString.h"
#endif


class TList;
class TXmlRpc;


class TLM : public TClProxy {
public:
   TLM(TXmlRpc *rpc);
   virtual ~TLM() { }

   Bool_t   GetVersion(TString &version);
   Bool_t   StartSession(const Char_t *sessionid, TList *&config, Int_t &hbf);
   Bool_t   DataReady(const Char_t *sessionid, Long64_t &bytesready,
                      Long64_t &totalbytes);
   Bool_t   Heartbeat(const Char_t *sessionid);
   Bool_t   EndSession(const Char_t *sessionid);

   struct TSlaveParams : public TObject {
      TString  fNode;
      Int_t    fPerfidx;
      TString  fImg;
      TString  fAuth;
      TString  fAccount;
      TString  fType;

      void     Print(Option_t *option="") const;

      ClassDef(TLM::TSlaveParams, 0);  // PEAC Slave config
   };

   ClassDef(TLM,0);  // PEAC Local Manager proxy
};

#endif
