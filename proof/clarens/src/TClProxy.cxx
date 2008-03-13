// @(#)root/clarens:$Id$
// Author: Maarten Ballintijn   26/10/2004

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TClProxy                                                             //
//                                                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TClProxy.h"


#include "xmlrpc.h"
#include "xmlrpc_client.h"


#include "Riostream.h"
#include "TClass.h"
#include "TXmlRpc.h"


ClassImp(TClProxy)


//______________________________________________________________________________
TClProxy::TClProxy(const Char_t *service, TXmlRpc *rpc)
   : fRpc(rpc)
{
   fRpc->SetService(service);
}

//______________________________________________________________________________
void TClProxy::Print(Option_t *) const
{
   cout << IsA()->GetName()
      << ": service " << fRpc->GetService() << " @ "
      << fRpc->GetServer() << endl;
}


//______________________________________________________________________________
Bool_t TClProxy::RpcFailed(const Char_t *member, const Char_t *what)
{
   // Test the environment for an error and report
   TString where(this->ClassName());
   where += "::";
   where += member;

   return fRpc->RpcFailed(where, what);

}
