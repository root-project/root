// @(#)root/clarens:$Id$
// Author: Maarten Ballintijn   25/10/2004

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TEcho                                                                //
//                                                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TEcho.h"


#include "xmlrpc.h"
#include "xmlrpc_client.h"


#include "TString.h"
#include "TXmlRpc.h"
#include "TStopwatch.h"
#include "Riostream.h"


ClassImp(TEcho)


//______________________________________________________________________________
TEcho::TEcho(TXmlRpc *rpc)
   : TClProxy("echo", rpc)
{
}


//______________________________________________________________________________
Bool_t TEcho::Echo(const Char_t *in, TString &out)
{
   xmlrpc_env *env = fRpc->GetEnv();

   xmlrpc_value *arg = xmlrpc_build_value(env, "(s)", in);
   if (RpcFailed("Echo", "encode argument")) return kFALSE;

   xmlrpc_value *retval = fRpc->Call("echo", arg);
   if (RpcFailed("Echo", "call")) return kFALSE;

   if (gDebug>1) fRpc->PrintValue(retval);

   char *o;
   xmlrpc_parse_value(env, retval, "(s)", &o);
   if (RpcFailed("Echo", "decode")) return kFALSE;

   out = o;

   xmlrpc_DECREF (arg);
   xmlrpc_DECREF (retval);

      return kTRUE;
}


//______________________________________________________________________________
Bool_t TEcho::Hostname(TString &name, TString &ip)
{
   xmlrpc_env *env = fRpc->GetEnv();

   xmlrpc_value *retval = fRpc->Call("hostname", 0);
   if (RpcFailed("Hostname", "call")) return kFALSE;

   if (gDebug>1) fRpc->PrintValue(retval);

   char *n, *i;
   xmlrpc_parse_value(env, retval, "(ss)", &n, &i);
   if (RpcFailed("Hostname", "decode")) return kFALSE;

   name = n;
   ip = i;

   xmlrpc_DECREF (retval);

   return kTRUE;
}


//______________________________________________________________________________
void TEcho::Benchmark(Int_t iterations)
{
   TStopwatch timer;
   TString out;

   for(Int_t i = 0; i < iterations ; i++) {
      Echo("Dummy test string", out);
   }

   timer.Stop();

   cout <<
      "Elapsed time is " << timer.RealTime() << " s, " <<
      iterations / timer.RealTime() << " calls/s for " <<
      iterations << " calls" << endl;
}

