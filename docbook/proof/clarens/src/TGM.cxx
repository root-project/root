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
// TGM                                                                  //
//                                                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TGM.h"


#include "memory"
#include "xmlrpc.h"
#include "xmlrpc_client.h"


#include "Riostream.h"
#include "TClass.h"
#include "TList.h"
#include "TObjString.h"
#include "TUrl.h"
#include "TXmlRpc.h"


ClassImp(TGM)
ClassImp(TGM::TFileParams)


//______________________________________________________________________________
TGM::TGM(TXmlRpc *rpc)
   : TClProxy("GM", rpc)
{
}


//______________________________________________________________________________
Bool_t TGM::GetVersion(TString &version)
{
   xmlrpc_env *env = fRpc->GetEnv();

   xmlrpc_value *retval = fRpc->Call("version", 0);
   if (RpcFailed("GetVersion", "call")) return kFALSE;

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
Bool_t TGM::CreateSession(const Char_t *dataset,
                          TString &sessionid,
                          TList   *&files,
                          TUrl    &proofUrl)
{
   xmlrpc_env *env = fRpc->GetEnv();

   xmlrpc_value *arg = xmlrpc_build_value(env, "(s)", dataset);
   if (RpcFailed("CreateSession", "encode argument")) return kFALSE;

   xmlrpc_value *retval = fRpc->Call("create_session",arg);
   if (RpcFailed("CreateSession", "call")) return kFALSE;

   char *rc;
   xmlrpc_value *val;
   xmlrpc_parse_value(env, retval, "(sV)", &rc, &val);
   if (RpcFailed("CreateSession", "decode")) return kFALSE;

   if (strcmp(rc, "SUCCESS") != 0) {
      char *errmsg;
      xmlrpc_parse_value(env, val, "s", &errmsg);
      if (RpcFailed("CreateSession", "decode errmsg")) return kFALSE;

      Error("CreateSession", "%s", errmsg);
      return kFALSE;
   }

   char *id, *url;
   xmlrpc_value *filearray;
   xmlrpc_parse_value(env, val, "(ssA)",&id, &url, &filearray);
   if (RpcFailed("CreateSession", "decode results")) return kFALSE;

   sessionid = id;
   proofUrl = url;

   int n = xmlrpc_array_size (env, filearray);
   if (RpcFailed("CreateSession", "array size")) return kFALSE;

   std::auto_ptr<TList> ftemp(new TList);
   ftemp->SetOwner();
   for(int i=0; i < n; i++) {
      xmlrpc_value *entry = xmlrpc_array_get_item(env, filearray, i);
      if (RpcFailed("CreateSession", "get entry")) return kFALSE;

      char *dummy, *file, *cls, *name, *dir;
      Int_t size, first, n;
      xmlrpc_parse_value(env, entry, "((ss)(si)(ss)(ss)(si)(si)(ss))",
         &dummy, &file, &dummy, &size, &dummy, &cls, &dummy, &name,
         &dummy, &first, &dummy, &n, &dummy, &dir);
      if (RpcFailed("CreateSession", "decode entry")) return kFALSE;

      ftemp->Add(new TFileParams(file, cls, name, dir, first, n));
   }

   files = ftemp.release();
   xmlrpc_DECREF (arg);
   xmlrpc_DECREF (retval);

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGM::DestroySession(const Char_t *sessionid)
{
   xmlrpc_env *env = fRpc->GetEnv();

   xmlrpc_value *arg = xmlrpc_build_value(env, "(s)", sessionid);
   if (RpcFailed("DestroySession", "encode argument")) return kFALSE;

   xmlrpc_value *retval = fRpc->Call("destroy_job", arg);
   if (RpcFailed("DestroySession", "call")) return kFALSE;

   char *rc;
   xmlrpc_value *val;
   xmlrpc_parse_value(env, retval, "(sV)", &rc, &val);
   if (RpcFailed("DestroySession", "decode reply")) return kFALSE;

   if (strcmp(rc, "SUCCESS") != 0) {
      char *errmsg;
      xmlrpc_parse_value(env, val, "s", &errmsg);
      if (RpcFailed("DestroySession", "decode errmsg")) return kFALSE;

      Error("DestroySession", "%s", errmsg);
      return kFALSE;
   }

   xmlrpc_DECREF (arg);
   xmlrpc_DECREF (retval);

   return kTRUE;
}


//______________________________________________________________________________
TGM::TFileParams::TFileParams(const Char_t *file, const Char_t *cl, const Char_t *name,
                              const Char_t *dir, Int_t first,  Int_t n)
   : fFileName(file), fObjClass(cl), fObjName(name), fDir(dir), fFirst(first), fNum(n)
{
}


//______________________________________________________________________________
void TGM::TFileParams::Print(Option_t *) const
{
   cout << IsA()->GetName()
      << ":  '" << fFileName << "'"
      << "  " << fObjClass
      << " " << fObjName
      << " (" << fDir
      << ") [" << fFirst << ", " << fNum << "]"
      << endl;
}

