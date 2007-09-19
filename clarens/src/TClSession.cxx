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
// TClSession                                                           //
//                                                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TClSession.h"


#include <openssl/x509.h>
#include <openssl/pem.h>
#include <openssl/bio.h>
#include <openssl/err.h>
#include <openssl/evp.h>
#include <openssl/bn.h>


#include "Riostream.h"
#include "TEnv.h"
#include "TError.h"
#include "TList.h"
#include "TRandom.h"
#include "TString.h"
#include "TSystem.h"
#include "TTimeStamp.h"
#include "TUrl.h"


ClassImp(TClSession)


Bool_t   TClSession::fgInitialized  = kFALSE;
void    *TClSession::fgPrivRSA      = 0;
void    *TClSession::fgPubRSA       = 0;
TString  TClSession::fgUserCert;

//______________________________________________________________________________
unsigned char *B64Encode(xmlrpc_env *env,unsigned char *instring,int len)
{
   // Utility function that encodes instring in base64, and returns a new
   // string with its own memory reference. Free this memory upon
   // deconstruction.

    xmlrpc_mem_block *mem;
    mem=xmlrpc_base64_encode (env,instring,len);
    if (env->fault_occurred) {
       cerr<<"XML-RPC Fault: "<<env->fault_string<<"("<< env->fault_code<<")"<<endl;
       if (mem) xmlrpc_mem_block_free (mem);
       return 0;
    }

    if (!mem) return 0;
    int olen=xmlrpc_mem_block_size (mem);

    unsigned char *outstring=new unsigned char[olen+1];
    memcpy((void *) outstring,(void *)xmlrpc_mem_block_contents(mem),olen);
    outstring[olen]='\0';// Make a proper null-terminated string

    xmlrpc_mem_block_free (mem);
    return outstring;
}

//______________________________________________________________________________
unsigned char *B64Decode(xmlrpc_env *env, unsigned char *instring,int *len)
{
   // Utility function that encodes instring in base64, and returns a new
   // string with its own memory reference. Free this memory upon
   // deconstruction.

    xmlrpc_mem_block *mem;
    mem=xmlrpc_base64_decode (env,(char *)instring,strlen((const char *)instring));
    if (env->fault_occurred) {
       cerr<<"XML-RPC Fault: "<<env->fault_string<<"("<< env->fault_code<<")"<<endl;
       if (mem) xmlrpc_mem_block_free (mem);
       return 0;
    }

    if (!mem) return 0;
    int olen=xmlrpc_mem_block_size (mem);
    unsigned char *outstring=new unsigned char[olen+1];
    memcpy((void *) outstring,(void *)xmlrpc_mem_block_contents(mem),olen);
    outstring[olen]='\0';// Make a proper null-terminated string
    *len=olen;
    xmlrpc_mem_block_free (mem);
    return outstring;
}

//______________________________________________________________________________
TClSession::TClSession(const Char_t *url, const Char_t *user, const Char_t *pw,
                       xmlrpc_server_info *info, void *serverPubRSA)
   : fUrl(url), fUser(user), fPassword(pw), fServerInfo(info), fServerPubRSA(serverPubRSA)
{
}

//______________________________________________________________________________
TClSession *TClSession::Create(const Char_t *url)
{
   if (!InitAuthentication()) return 0;

   if (TString(url).EndsWith("/") == kFALSE) {
      ::Error("TClSession::Create", "URL must end with \"/\" (%s)", url);
      return 0;
   }

   // Construct user nonce value
   unsigned char nonce[SHA_DIGEST_LENGTH];
   Long_t ns = TTimeStamp().GetNanoSec();
   TRandom rndm(ns);
   SHA1((UChar_t*) Form("%x_%lx_%lx", gSystem->GetPid(), ns, (Long_t)rndm.Uniform(1e8)),
        22, nonce);

   xmlrpc_env env;
   xmlrpc_env_init(&env);

   TString user = (char *) B64Encode(&env, nonce, SHA_DIGEST_LENGTH);

   xmlrpc_server_info *info = xmlrpc_server_info_new(&env, (char*)url);
   if (env.fault_occurred) {
      ::Error("TClSession::Create", "creating server info: %s (%d)",
            env.fault_string, env.fault_code);
      return 0;
   }

   xmlrpc_server_info_set_basic_auth (&env, info, (char*)user.Data(), (char*)fgUserCert.Data());
   if (env.fault_occurred) {
      ::Error("TClSession::Create", "setting basic auth: %s (%d)",
            env.fault_string, env.fault_code);
      return 0;
   }

   xmlrpc_value *val = xmlrpc_client_call_server (&env, info, "system.auth", "()");
   if (env.fault_occurred) {
      ::Error("TClSession::Create", "call system.auth(): %s (%d)",
            env.fault_string, env.fault_code);
      return 0;
   }

   char *cert;
   unsigned char *cryptServerNonce64, *cryptUserNonce64;
   xmlrpc_parse_value(&env, val, "(sss)", &cert, &cryptServerNonce64, &cryptUserNonce64);
   if (env.fault_occurred) {
      ::Error("TClSession::Create", "parsing result: %s (%d)",
            env.fault_string, env.fault_code);
      return 0;
   }

   BIO *b = BIO_new_mem_buf(cert,strlen(cert));
   X509 *serverCert = PEM_read_bio_X509(b, 0, 0, 0);
   BIO_free(b);
   if (serverCert == 0) {
      ::Error("TClSession::Create", "reading cert from server response: %s",
            ERR_reason_error_string(ERR_get_error()));
      return 0;
   }

   EVP_PKEY *serverPubKey = X509_get_pubkey(serverCert);
   if (serverPubKey == 0) {
      ::Error("TClSession::Create", "extracting cert from server response: %s",
            ERR_reason_error_string(ERR_get_error()));
      return 0;
   }

   void* serverPubRSA = EVP_PKEY_get1_RSA(serverPubKey);
   if (serverPubRSA == 0) {
      ::Error("TClSession::Create", "extracting pub key from cert: %s",
            ERR_reason_error_string(ERR_get_error()));
      return 0;
   }

   // user nonce first
   int len;

   unsigned char *cryptNonce = B64Decode(&env, cryptUserNonce64, &len);
   unsigned char *serverUstring = new unsigned char[RSA_size((RSA*)serverPubRSA)];
   len = RSA_public_decrypt(len, cryptNonce, serverUstring,
                            (RSA*) serverPubRSA, RSA_PKCS1_PADDING);
   if (len == -1) {
      ::Error("TClSession::Create", "recovering digest: %s",
            ERR_reason_error_string(ERR_get_error()));
      delete [] cryptNonce;
      return 0;
   }
   serverUstring[len] = '\0';
   delete [] cryptNonce;

   // server nonce next
   cryptNonce = B64Decode(&env, cryptServerNonce64, &len);
   unsigned char *serverNonce = new unsigned char[RSA_size((RSA*)fgPrivRSA)];

   len = RSA_private_decrypt(len, cryptNonce, serverNonce,
                             (RSA*) fgPrivRSA, RSA_PKCS1_PADDING);
   if (len == -1) {
      ::Error("TClSession::Create", "decoding server nonce: %s",
            ERR_reason_error_string(ERR_get_error()));
      delete [] cryptNonce;
      return 0;
   }
   serverNonce[len] = '\0';
   delete [] cryptNonce;
   xmlrpc_DECREF (val);

   // calculate hash of server nonce

   SHA1(serverNonce, len, nonce);
   TString password = (char *) B64Encode(&env, nonce, SHA_DIGEST_LENGTH);

   xmlrpc_server_info_set_basic_auth (&env, info, (char*)user.Data(), (char*)password.Data());

   return new TClSession(url, user, password, info, serverPubRSA);
}

//______________________________________________________________________________
Bool_t TClSession::InitAuthentication()
{
   if (fgInitialized) return kTRUE;

   // Initialize SSL
   OpenSSL_add_all_algorithms();
   OpenSSL_add_all_ciphers();
   OpenSSL_add_all_digests();
   ERR_load_crypto_strings();

   // Load user certificate and public key


   BIO *bio = 0;
   TString certFile(gEnv->GetValue("Clarens.CertFile", ""));
   if (certFile.Length() > 0) {
      bio = BIO_new_file(certFile,"r");
   } else {
      certFile = Form("/tmp/x509up_u%d", gSystem->GetUid());
      if (gSystem->AccessPathName(certFile) == kFALSE) {
         // yes, file exists
         bio = BIO_new_file(certFile,"r");
      } else {
         certFile = gSystem->HomeDirectory();
         certFile += "/.globus/usercert.pem";
         bio = BIO_new_file(certFile,"r");
      }
   }

   if (bio == 0) {
      ::Error("TClSession::InitAuthentication", "cannot open '%s' (%s)",
              certFile.Data(), gSystem->GetError());
      return kFALSE;
   }

   if (gDebug > 0) ::Info("TClSession::InitAuthentication",
      "using public key: '%s'", certFile.Data());

   X509* userCert = PEM_read_bio_X509(bio, 0,0,0);
   if (!userCert) {
      ::Error("TClSession::InitAuthentication", "reading user public key: %s (%ld)",
         ERR_reason_error_string(ERR_get_error()), ERR_get_error());
      BIO_free(bio);
      return kFALSE;
   }

   BIO_free(bio);

   TString line;
   Bool_t incert=kFALSE;
   fgUserCert = "";
   ifstream fin(certFile);
   while (!fin.eof()) {
      line.ReadToDelim(fin,'\n');

      if (line.Contains("-----BEGIN CERTIFICATE-----") ||
          line.Contains("-----BEGIN X509 CERTIFICATE-----")) incert=kTRUE;

      if (incert) fgUserCert += line + "\n";

      if (line.Contains("-----END CERTIFICATE-----") ||
          line.Contains("-----END X509 CERTIFICATE-----")) incert=kFALSE;
   }
   fin.close();

   EVP_PKEY *userPubKey = X509_get_pubkey(userCert);
   if (userPubKey == 0) {
      ::Error("TClSession::InitAuthentication", "extracting user public key: %s (%ld)",
         ERR_reason_error_string(ERR_get_error()), ERR_get_error());
      X509_free(userCert);
      return kFALSE;
   }

   X509_free(userCert);
   fgPubRSA = EVP_PKEY_get1_RSA(userPubKey);
   if (fgPubRSA == 0) {
      ::Error("TClSession::InitAuthentication", "extracting RSA structure from user public key: %s (%ld)",
         ERR_reason_error_string(ERR_get_error()), ERR_get_error());
      EVP_PKEY_free(userPubKey);
      return kFALSE;
   }

   EVP_PKEY_free(userPubKey);

   // Load private key

   TString privfile(gEnv->GetValue("Clarens.KeyFile", ""));
   if (privfile.Length() > 0) {
      bio = BIO_new_file(privfile,"r");
   } else {
      privfile = Form("/tmp/x509up_u%d", gSystem->GetUid());
      if (gSystem->AccessPathName(privfile) == kFALSE) {
         // yes, file exists
         bio = BIO_new_file(privfile,"r");
      } else {
         privfile = gSystem->HomeDirectory();
         privfile += "/.globus/userkey.pem";
         bio = BIO_new_file(privfile,"r");
      }
   }

   if (bio == 0) {
      ::Error("TClSession::InitAuthentication", "cannot open '%s' (%s)",
              privfile.Data(), gSystem->GetError());
      RSA_free((RSA*)fgPubRSA); fgPubRSA = 0;
      return kFALSE;
   }

   if (gDebug > 0) ::Info("TClSession::InitAuthentication",
      "using private key: '%s'", privfile.Data());

   fgPrivRSA = PEM_read_bio_RSAPrivateKey(bio, 0, 0, 0);
   BIO_free(bio);

   if (fgPrivRSA == 0) {
      ::Error("TClSession::InitAuthentication", "extracting RSA structure from user private key: %s (%ld)",
         ERR_reason_error_string(ERR_get_error()), ERR_get_error());
      RSA_free((RSA*)fgPubRSA); fgPubRSA = 0;
      return kFALSE;
   }

   fgInitialized = kTRUE;
   return kTRUE;
}
