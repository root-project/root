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
// TSAM                                                                 //
//                                                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TSAM.h"

#include "memory"
#include <stdlib.h>
#include "xmlrpc.h"
#include "xmlrpc_client.h"


#include "TString.h"
#include "TObjString.h"
#include "TList.h"
#include "TXmlRpc.h"
#include "TGM.h"


ClassImp(TSAM)


//______________________________________________________________________________
TSAM::TSAM(TXmlRpc *rpc)
   : TClProxy("SAM", rpc)
{
}


//______________________________________________________________________________
Bool_t TSAM::GetVersion(TString &version)
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
Bool_t TSAM::GetDatasets(TList *&datasets)
{
   xmlrpc_env *env = fRpc->GetEnv();

   xmlrpc_value *retval = fRpc->Call("list_datasets", 0);
   if (RpcFailed("GetDatasets", "call")) return kFALSE;

   char *rc;
   xmlrpc_value *val;
   xmlrpc_parse_value(env, retval, "(sV)", &rc, &val);
   if (RpcFailed("GetDatasets", "decode reply")) return kFALSE;

   if (strcmp(rc, "SUCCESS") != 0) {
      char *errmsg;
      xmlrpc_parse_value(env, val, "s", &errmsg);
      if (RpcFailed("GetDatasets", "decode errmsg")) return kFALSE;

      Error("GetDatasets", "%s", errmsg);
      return kFALSE;
   }

   int n = xmlrpc_array_size (env, val);
   if (RpcFailed("GetDatasets", "array size")) return kFALSE;

   std::auto_ptr<TList> temp(new TList);
   temp->SetOwner();
   for(int i=0; i < n; i++) {
      xmlrpc_value *entry = xmlrpc_array_get_item(env, val, i);
      if (RpcFailed("GetDatasets", "get entry")) return kFALSE;

      char *ds;
      xmlrpc_parse_value(env, entry, "s", &ds);
      if (RpcFailed("GetDatasets", "decode entry")) return kFALSE;

      temp->Add(new TObjString(ds));
   }

   xmlrpc_DECREF (retval);

      datasets = temp.release();
   return kTRUE;
}


//______________________________________________________________________________
Bool_t TSAM::GetDSetLocations(const Char_t *dataset, TList *&lmUrls)
{
   xmlrpc_env *env = fRpc->GetEnv();

   xmlrpc_value *arg = xmlrpc_build_value(env, "(s)", dataset);
   if (RpcFailed("GetDSetLocations", "encode argument")) return kFALSE;

   xmlrpc_value *retval = fRpc->Call("dataset_locations", arg);
   if (RpcFailed("GetDSetLocations", "call")) return kFALSE;

   char *rc;
   xmlrpc_value *val;
   xmlrpc_parse_value(env, retval, "(sV)", &rc, &val);
   if (RpcFailed("GetDSetLocations", "decode reply")) return kFALSE;

   if (strcmp(rc, "SUCCESS") != 0) {
      char *errmsg;
      xmlrpc_parse_value(env, val, "s", &errmsg);
      if (RpcFailed("GetDSetLocations", "decode errmsg")) return kFALSE;

      Error("GetDSetLocations", "%s", errmsg);
      return kFALSE;
   }

   int n = xmlrpc_array_size (env, val);
   if (RpcFailed("GetDSetLocations", "array size")) return kFALSE;

   std::auto_ptr<TList> temp(new TList);
   temp->SetOwner();
   for(int i=0; i < n; i++) {
      xmlrpc_value *entry = xmlrpc_array_get_item(env, val, i);
      if (RpcFailed("GetDSetLocations", "get entry")) return kFALSE;

      char *ds;
      xmlrpc_parse_value(env, entry, "s", &ds);
      if (RpcFailed("GetDSetLocations", "decode entry")) return kFALSE;

      temp->Add(new TObjString(ds));
   }

   xmlrpc_DECREF (arg);
   xmlrpc_DECREF (retval);

   lmUrls = temp.release();
   return kTRUE;
}


//______________________________________________________________________________
Bool_t TSAM::GetDSetFiles(const Char_t *dataset, const Char_t *lmUrl, TList *&files)
{
   xmlrpc_env *env = fRpc->GetEnv();

   xmlrpc_value *arg = xmlrpc_build_value(env, "(ss)", dataset, lmUrl);
   if (RpcFailed("GetDSetFiles", "encode argument")) return kFALSE;

   xmlrpc_value *retval = fRpc->Call("dataset_files", arg);
   if (RpcFailed("GetDSetFiles", "call")) return kFALSE;

   char *rc;
   xmlrpc_value *val;
   xmlrpc_parse_value(env, retval, "(sV)", &rc, &val);
   if (RpcFailed("GetDSetFiles", "decode reply")) return kFALSE;

   if (strcmp(rc, "SUCCESS") != 0) {
      char *errmsg;
      xmlrpc_parse_value(env, val, "s", &errmsg);
      if (RpcFailed("GetDSetetFiles", "decode errmsg")) return kFALSE;

      Error("GetDSetFiles", "%s", errmsg);
      return kFALSE;
   }

   int n = xmlrpc_array_size (env, val);
   if (RpcFailed("GetDSetFiles", "array size")) return kFALSE;

   std::auto_ptr<TList> temp(new TList);
   temp->SetOwner();
   for(int i=0; i < n; i++) {
      xmlrpc_value *entry = xmlrpc_array_get_item(env, val, i);
      if (RpcFailed("GetDSetFiles", "get entry")) return kFALSE;

      char *dummy, *file, *cls, *name, *dir;
      Int_t size, first, n;
      xmlrpc_parse_value(env, entry, "((ss)(si)(ss)(ss)(si)(si)(ss))",
         &dummy, &file, &dummy, &size, &dummy, &cls, &dummy, &name,
         &dummy, &first, &dummy, &n, &dummy, &dir);
      if (RpcFailed("GetDSetFiles", "decode entry")) return kFALSE;

      temp->Add(new TGM::TFileParams(file, cls, name, dir, first, n));
   }

   xmlrpc_DECREF (arg);
   xmlrpc_DECREF (retval);

   files = temp.release();
   return kTRUE;
}


//______________________________________________________________________________
Bool_t TSAM::GetDSetSize(const Char_t *dataset, Long64_t &size)
{
   xmlrpc_env *env = fRpc->GetEnv();

   xmlrpc_value *arg = xmlrpc_build_value(env, "(s)", dataset);
   if (RpcFailed("GetDSetSize", "encode argument")) return kFALSE;

   xmlrpc_value *retval = fRpc->Call("dataset_size", arg);
   if (RpcFailed("GetDSetSize", "call")) return kFALSE;

   char *rc;
   xmlrpc_value *val;
   xmlrpc_parse_value(env, retval, "(sV)", &rc, &val);
   if (RpcFailed("GetDSetSize", "decode reply")) return kFALSE;

   if (strcmp(rc, "SUCCESS") != 0) {
      char *errmsg;
      xmlrpc_parse_value(env, val, "s", &errmsg);
      if (RpcFailed("GetDSetSize", "decode errmsg")) return kFALSE;

      Error("GetDSetSize", "%s", errmsg);
      return kFALSE;
   }

   double d;
   xmlrpc_parse_value(env, val, "d", &d);
   if (RpcFailed("GetDSetSize", "decode version")) return kFALSE;

   xmlrpc_DECREF (arg);
   xmlrpc_DECREF (retval);

   size = static_cast<Long64_t>(d);

   return kTRUE;
}
