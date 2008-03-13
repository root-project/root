// @(#)root/clarens:$Id$
// Author: Maarten Ballintijn   21/10/2004

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TClarens                                                             //
//                                                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TClarens.h"


#include "TClSession.h"
#include "TEcho.h"
#include "TGM.h"
#include "THashList.h"
#include "TLM.h"
#include "TSAM.h"
#include "TXmlRpc.h"


namespace {

//------------------------------------------------------------------------------

struct TClarensStartup {
   TClarensStartup() {TClarens::Init();}
} Clarenstartup;

};


//------------------------------------------------------------------------------

TClarens *gClarens = 0;

ClassImp(TClarens)


//______________________________________________________________________________
TClarens::TClarens()
   : fTimeout(10000), fSessions(new THashList)
{
   xmlrpc_client_init(XMLRPC_CLIENT_NO_FLAGS, "ROOT Clarens client", "1.0");
}


//______________________________________________________________________________
TClarens::~TClarens()
{
   delete fSessions;
   xmlrpc_client_cleanup();
}


//______________________________________________________________________________
TClSession *TClarens::Connect(const Char_t * url)
{
   TClSession *session = (TClSession*) fSessions->FindObject(url);

   if (session == 0) {
      session = TClSession::Create(url);
      if (session != 0) fSessions->Add(session);
   }

   return session;
}


//______________________________________________________________________________
TEcho *TClarens::CreateEcho(const Char_t * echoUrl)
{
   TClSession *session = Connect(echoUrl);

   if (session == 0) return 0;

   return new TEcho(new TXmlRpc(session));
}


//______________________________________________________________________________
TGM *TClarens::CreateGM(const Char_t * gmUrl)
{
   TClSession *session = Connect(gmUrl);

   if (session == 0) return 0;

   return new TGM(new TXmlRpc(session));
}


//______________________________________________________________________________
TLM *TClarens::CreateLM(const Char_t * lmUrl)
{
   TClSession *session = Connect(lmUrl);

   if (session == 0) return 0;

   return new TLM(new TXmlRpc(session));
}


//______________________________________________________________________________
TSAM *TClarens::CreateSAM(const Char_t * samUrl)
{
   TClSession *session = Connect(samUrl);

   if (session == 0) return 0;

   return new TSAM(new TXmlRpc(session));
}


//______________________________________________________________________________
void TClarens::Init()
{
   if (gClarens == 0) {
      gClarens = new TClarens;
   }
}
