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
// TLM                                                                  //
//                                                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TLM.h"

#include "memory"
#include <stdlib.h>
#include "xmlrpc.h"
#include "xmlrpc_client.h"


#include "Riostream.h"
#include "TClass.h"
#include "TList.h"
#include "TString.h"
#include "TXmlRpc.h"


ClassImp(TLM)
ClassImp(TLM::TSlaveParams)


//______________________________________________________________________________
TLM::TLM(TXmlRpc *rpc)
   : TClProxy("LM", rpc)
{
}


//______________________________________________________________________________
Bool_t TLM::GetVersion(TString &version)
{
   xmlrpc_env *env = fRpc->GetEnv();

   xmlrpc_value *retval = fRpc->Call("version", 0);
   if (RpcFailed("GetVersion", "call")) return kFALSE;

   if (gDebug>0) fRpc->PrintValue(retval);

   char *rc;
   xmlrpc_value *val;
   xmlrpc_parse_value(env, retval, "(sV)", &rc, &val);
   if (RpcFailed("GetVersion", "decode")) return kFALSE;

   if (strcmp(rc, "SUCCESS") != 0) {
      char *errmsg;
      xmlrpc_parse_value(env, val, "s", &errmsg);
      if (RpcFailed("GetVersion", "decode errmsg")) return kFALSE;

      Error("GetVersion", "%s", errmsg);
      return kFALSE;
   }

   char *v;
   xmlrpc_parse_value(env, val, "s", &v);
   if (RpcFailed("GetVersion", "decode version")) return kFALSE;

   version = v;

   xmlrpc_DECREF (retval);
   return kTRUE;
}


//______________________________________________________________________________
Bool_t TLM::StartSession(const Char_t *sessionid, TList *&config, Int_t &hbf)
{
   xmlrpc_env *env = fRpc->GetEnv();

   xmlrpc_value *arg = xmlrpc_build_value(env, "(s)", sessionid);
   if (RpcFailed("StartSession", "encode argument")) return kFALSE;

   xmlrpc_value *retval = fRpc->Call("get_config", arg);
   if (RpcFailed("StartSession", "call")) return kFALSE;

   char *rc;
   xmlrpc_value *val;
   xmlrpc_parse_value(env, retval, "(sV)", &rc, &val);
   if (RpcFailed("StartSession", "decode reply")) return kFALSE;

   if (strcmp(rc, "SUCCESS") != 0) {
      char *errmsg;
      xmlrpc_parse_value(env, val, "s", &errmsg);
      if (RpcFailed("StartSession", "decode errmsg")) return kFALSE;

      Error("StartSession", "%s", errmsg);
      return kFALSE;
   }

   char *hbfs;
   xmlrpc_value *nodearray;
   xmlrpc_parse_value(env, val, "(sA)", &hbfs, &nodearray);
   if (RpcFailed("StartSession", "decode results")) return kFALSE;

   hbf = atol(hbfs);
   int n = xmlrpc_array_size (env, nodearray);
   if (RpcFailed("StartSession", "array size")) return kFALSE;

   std::auto_ptr<TList> ntemp(new TList);
   ntemp->SetOwner();
   // skip entry zero with the labels
   for(int i=1; i < n; i++) {
      xmlrpc_value *entry = xmlrpc_array_get_item(env, nodearray, i);
      if (RpcFailed("StartSession", "get entry")) return kFALSE;

//fields=['name','cpu_speed_Ghz','specint','image','start_mechanism',
//   'authentication_type','maximum_slaves']

      char *name, *img, *startup, *auth, *max;
      double mhz;
      int specint;
      xmlrpc_parse_value(env, entry, "(sdisssi)", &name, &mhz, &specint,
                         &img, &startup, &auth, &max);
      if (RpcFailed("StartSession", "decode entry")) return kFALSE;

      TSlaveParams *sl = new TSlaveParams;

      sl->fNode      = name;
      sl->fPerfidx   = specint;
      sl->fImg       = img;
      sl->fAuth      = auth;
      sl->fAccount   = "nobody";
      sl->fType      = startup;

      ntemp->Add(sl);
   }

   config = ntemp.release();
   xmlrpc_DECREF (arg);
   xmlrpc_DECREF (retval);

   return kTRUE;
}


//______________________________________________________________________________
Bool_t TLM::Heartbeat(const Char_t *sessionid)
{
   xmlrpc_env *env = fRpc->GetEnv();

   xmlrpc_value *arg = xmlrpc_build_value(env, "(s)", sessionid);
   if (RpcFailed("Heartbeat", "encode argument")) return kFALSE;

   xmlrpc_value *retval = fRpc->Call("alive", arg);
   if (RpcFailed("Heartbeat", "call")) return kFALSE;

   char *rc;
   xmlrpc_value *val;
   xmlrpc_parse_value(env, retval, "(sV)", &rc, &val);
   if (RpcFailed("Heartbeat", "decode reply")) return kFALSE;

   if (strcmp(rc, "SUCCESS") != 0) {
      char *errmsg;
      xmlrpc_parse_value(env, val, "s", &errmsg);
      if (RpcFailed("Heartbeat", "decode errmsg")) return kFALSE;

      Error("Heartbeat", "%s", errmsg);
      return kFALSE;
   }

   return kTRUE;
}


//______________________________________________________________________________
Bool_t TLM::DataReady(const Char_t *sessionid, Long64_t & bytesready,
                      Long64_t & bytestotal)
{
   xmlrpc_env *env = fRpc->GetEnv();

   xmlrpc_value *arg = xmlrpc_build_value(env, "(s)", sessionid);
   if (RpcFailed("DataReady", "encode argument")) return kFALSE;

   xmlrpc_value *retval = fRpc->Call("data_ready", arg);
   if (RpcFailed("DataReady", "call")) return kFALSE;

   char *rc;
   xmlrpc_value *val;
   xmlrpc_parse_value(env, retval, "(sV)", &rc, &val);
   if (RpcFailed("DataReady", "decode reply")) return kFALSE;

   if (strcmp(rc, "SUCCESS") != 0) {
      char *errmsg;
      xmlrpc_parse_value(env, val, "s", &errmsg);
      if (RpcFailed("DataReady", "decode errmsg")) return kFALSE;

      Error("DataReady", "%s", errmsg);
      return kFALSE;
   }

   Int_t isReady;
   Double_t ready, total;
   xmlrpc_parse_value(env, val, "(bdd)", &isReady, &ready, &total);
   if (RpcFailed("DataReady", "decode results")) return kFALSE;

   bytesready = (Long64_t) ready;
   bytestotal = (Long64_t) total;
   return isReady;
}


//______________________________________________________________________________
Bool_t TLM::EndSession(const Char_t *sessionid)
{
   xmlrpc_env *env = fRpc->GetEnv();

   xmlrpc_value *arg = xmlrpc_build_value(env, "(s)", sessionid);
   if (RpcFailed("EndSession", "encode argument")) return kFALSE;

   xmlrpc_value *retval = fRpc->Call("destroy_job", arg);
   if (RpcFailed("EndSession", "call")) return kFALSE;

   char *rc;
   xmlrpc_value *val;
   xmlrpc_parse_value(env, retval, "(sV)", &rc, &val);
   if (RpcFailed("EndSession", "decode reply")) return kFALSE;

   if (strcmp(rc, "SUCCESS") != 0) {
      char *errmsg;
      xmlrpc_parse_value(env, val, "s", &errmsg);
      if (RpcFailed("EndSession", "decode errmsg")) return kFALSE;

      Error("EndSession", "%s", errmsg);
      return kFALSE;
   }
   return kTRUE;
}



//______________________________________________________________________________
void TLM::TSlaveParams::Print(Option_t * /*option*/) const
{
   cout << IsA()->GetName()
      << ":  " << fNode
      << ", " << fPerfidx
      << ", " << fImg
      << ", " << fAuth
      << ", " << fAccount
      << ", " << fType
      << endl;
}
