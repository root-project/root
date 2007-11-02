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
// TXmlRpc                                                              //
//                                                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TXmlRpc.h"


#include "TClSession.h"
#include "TError.h"




namespace {

void print_values(int level, xmlrpc_env *env, xmlrpc_value *in);

Bool_t report_error(xmlrpc_env *env, const char *what)
{
   if (env->fault_occurred) {
      printf("Error decoding %s: %s (%d)", what, env->fault_string,
         env->fault_code);
   }
   return env->fault_occurred;
}


int get_int(xmlrpc_env *env, xmlrpc_value *in)
{
  int i = 0;
  xmlrpc_parse_value(env, in, "i", &i);
  report_error(env,"int");
  return(i);
}


int get_boolean(xmlrpc_env *env, xmlrpc_value *in)
{
  int i = 0;
  xmlrpc_parse_value(env, in, "b", &i);
  report_error(env,"bool");
  return(i);
}


double get_double(xmlrpc_env *env, xmlrpc_value *in)
{
  double d = 0;
  xmlrpc_parse_value(env, in, "d", &d);
  report_error(env,"double");
  return(d);
}


char* get_timestamp(xmlrpc_env *env, xmlrpc_value *in)
{
  char *s = 0;

  xmlrpc_parse_value(env, in, "8", &s);
  report_error(env,"timestamp");
  return(s);
}


char* get_string(xmlrpc_env *env, xmlrpc_value *in){
  char* s = 0;

  xmlrpc_parse_value(env, in, "s", &s);
  report_error(env,"string");
  return(s);
}


char* get_base64(xmlrpc_env *env, xmlrpc_value *in)
{
  char *s = 0;

  xmlrpc_parse_value(env, in, "6", &s);
  report_error(env,"base64");
  return(s);
}


void get_struct(int level, xmlrpc_env *env, xmlrpc_value *in){
  int i, size = 0;
  xmlrpc_value *key, *value;
  TString space(' ', level * 3);

  size = xmlrpc_struct_size(env, in);
  if (report_error(env,"struct")) return;

  for(i=0; i < size; i++){
    xmlrpc_struct_get_key_and_value(env, in, i, &key, &value);
    if (report_error(env,"struct member")) return;

    char *keystr = get_string(env, key);
    if (env->fault_occurred) return;

    printf("%s%s:\n", space.Data(), keystr);
    print_values(level+1, env, value);
    if (env->fault_occurred) return;
  }
}


void get_array(int level, xmlrpc_env *env, xmlrpc_value *in)
{
  int i, size = 0;
  xmlrpc_value *el;

  size = xmlrpc_array_size(env, in);
  if (report_error(env,"array")) return;

  for(i=0; i < size; i++){
    el = xmlrpc_array_get_item( env, in, i);
    if (report_error(env,"array element")) return;

    print_values(level, env, el);
    if (env->fault_occurred) return;
  }
}


void print_values(int level, xmlrpc_env *env, xmlrpc_value *in)
{
   TString space(' ', level * 3);

   printf("%s", space.Data());

   /* What did we get back? */
   switch (xmlrpc_value_type(in)) {
   case (XMLRPC_TYPE_INT):
     printf("int       %d\n", get_int(env, in));
     break;
   case (XMLRPC_TYPE_BOOL):
     printf("bool      %s\n", get_boolean(env, in) ? "true" : "false" );
     break;
   case (XMLRPC_TYPE_DOUBLE):
     printf("double    %g\n", get_double(env, in));
     break;
   case (XMLRPC_TYPE_DATETIME):
     printf("timestamp %s\n",
           get_timestamp(env, in));
     break;
   case (XMLRPC_TYPE_STRING):
     printf("string   '%s'\n", get_string(env, in));
     break;
   case (XMLRPC_TYPE_BASE64):
     printf("base64    %s\n",
           get_base64(env, in));
     break;
   case (XMLRPC_TYPE_ARRAY):
     printf("(\n");
     get_array(level+1, env, in);
     printf("%s)\n", space.Data());
     break;
   case (XMLRPC_TYPE_STRUCT):
     printf("{\n");
     get_struct(level+1, env, in);
     printf("%s}\n", space.Data());
     break;
   case (XMLRPC_TYPE_C_PTR):
     printf("Got a C pointer?!\n");
     break;
   case (XMLRPC_TYPE_DEAD):
     printf("Got a 0xDEADr?!\n");
     break;
   default:
     printf("UNKNOWN XML-RPC DATATYPE\n");
   }
}


};


ClassImp(TXmlRpc)

//______________________________________________________________________________
TXmlRpc::TXmlRpc(TClSession *session)
   : fSession(session)
{
   fEnv = new xmlrpc_env;
   xmlrpc_env_init(fEnv);
}


//______________________________________________________________________________
TXmlRpc::~TXmlRpc()
{
   delete fEnv;
}


//______________________________________________________________________________
xmlrpc_value *TXmlRpc::Call(const Char_t *method, xmlrpc_value *arg)
{
   // Make an XMLRPC call

   TString m = fService + "." + method;

   xmlrpc_value *retval = xmlrpc_client_call_server( fEnv,
                     fSession->GetServerInfo(), (char *) m.Data(),
                     (char *) (arg == 0 ? "()" : "V"), arg);

   if (gDebug > 1) {
      if (retval != 0) {
         Info("Call", "%s: returns:", m.Data());
         PrintValue(retval);
      } else {
         Info("Call", "%s: no return value", m.Data());
      }
   }

   return retval;
}


//______________________________________________________________________________
Bool_t TXmlRpc::RpcFailed(const Char_t *where, const Char_t *what)
{
   // Test the environment for an error and report

   if (fEnv->fault_occurred) {
      ::Error(where, "%s: %s (%d)", what,
            fEnv->fault_string, fEnv->fault_code);
      return kTRUE;
   }

   return kFALSE;
}


//______________________________________________________________________________
void TXmlRpc::PrintValue(xmlrpc_value *val)
{
   // Pretty print a, possibly complex, xmlrpc value

   xmlrpc_env *env = new xmlrpc_env; xmlrpc_env_init(env);

   print_values(0, env, val);

   xmlrpc_env_clean(env);
}


