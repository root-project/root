// @(#)root/clarens:$Id$
// Author: Maarten Ballintijn    25/10/2004

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TXmlRpc
#define ROOT_TXmlRpc

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TXmlRpc                                                              //
//                                                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#if !defined(__CINT__)
#include "xmlrpc.h"
#include "xmlrpc_client.h"
#else
struct xmlrpc_env;
struct xmlrpc_server_info;
struct xmlrpc_value;
#endif

#ifndef ROOT_TObject
#include "TObject.h"
#endif

#ifndef ROOT_TClSession
#include "TClSession.h"
#endif

class TXmlRpc : public TObject {
private:
   TClSession          *fSession;     //Clarens session info
   xmlrpc_env          *fEnv;         //call enviroment
   TString              fService;     //our service

public:
   TXmlRpc(TClSession *session);
   virtual ~TXmlRpc();

   void                 SetService(const Char_t *svc) {fService = svc;}
   const Char_t        *GetService() const {return fService;}

   xmlrpc_env          *GetEnv() {xmlrpc_env_clean(fEnv);
                                  xmlrpc_env_init(fEnv);
                                  return fEnv;}
   xmlrpc_server_info  *GetServerInfo() {return fSession->GetServerInfo();}
   const Char_t        *GetServer() {return fSession->GetServer();}
   xmlrpc_value        *Call(const Char_t *method, xmlrpc_value *arg);
   Bool_t               RpcFailed(const Char_t *where, const Char_t *what);
   void                 PrintValue(xmlrpc_value *val);

   ClassDef(TXmlRpc,0);  // XMLRPC interface class
};

#endif
