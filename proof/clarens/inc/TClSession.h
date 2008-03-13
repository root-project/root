// @(#)root/clarens:$Id$
// Author: Maarten Ballintijn    25/10/2004

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TClSession
#define ROOT_TClSession

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TClSession                                                           //
//                                                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#if !defined(__CINT__)
#include "xmlrpc.h"
#include "xmlrpc_client.h"
#else
struct xmlrpc_server_info;
#endif

#ifndef ROOT_TObject
#include "TObject.h"
#endif

#ifndef ROOT_TUrl
#include "TUrl.h"
#endif


class TUrl;


class TClSession : public TObject {
private:
   TUrl                 fUrl;          //server we are connected to
   TString              fUser;         //SHA1 string
   TString              fPassword;     //SHA1 string
   xmlrpc_server_info  *fServerInfo;   //per server data
   void                *fServerPubRSA; //(X509*)

   // single client certificate for the moment
   static void         *fgPrivRSA;
   static void         *fgPubRSA;
   static TString       fgUserCert;
   static Bool_t        fgInitialized;

   TClSession(const Char_t *url, const Char_t *user, const Char_t *pw,
              xmlrpc_server_info *info, void *serverPubRSA);

   static Bool_t        InitAuthentication();

public:
   virtual ~TClSession() { }

   xmlrpc_server_info  *GetServerInfo() {return fServerInfo;}
   const Char_t        *GetServer() {return fUrl.GetUrl();}

   static TClSession   *Create(const Char_t *url);

   ClassDef(TClSession,0);  // Clarens Session
};

#endif
