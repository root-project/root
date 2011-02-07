/* This code is extracted from mod_gridsite removing references to apache methods */

/*
   Copyright (c) 2003-7, Andrew McNab, Shiv Kaushal, Joseph Dada,
   and Yibiao Li, University of Manchester. All rights reserved.

   Redistribution and use in source and binary forms, with or
   without modification, are permitted provided that the following
   conditions are met:

     o Redistributions of source code must retain the above
       copyright notice, this list of conditions and the following
       disclaimer. 
     o Redistributions in binary form must reproduce the above
       copyright notice, this list of conditions and the following
       disclaimer in the documentation and/or other materials
       provided with the distribution. 

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
   CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
   INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
   MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
   DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS
   BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
   TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
   ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
   OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
   OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
   POSSIBILITY OF SUCH DAMAGE.


   This program includes code from dav_parse_range() from Apache mod_dav.c,
   and associated code contributed by  David O Callaghan
   
   Copyright 2000-2005 The Apache Software Foundation or its licensors, as
   applicable.
   
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0
   
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

#include "grst_verifycallback.h"
#include "gridsite.h"
#include <openssl/x509v3.h>
#include <string.h>

X509_STORE* grst_store=NULL;
int grst_verify=0;
int grst_depth=0;
char* grst_cadir   = "/etc/grid-certificates/certificates";
char* grst_vomsdir = "/etc/grid-certificates/vomsdir";

// APACHE mod_ssl functions
int ssl_callback_SSLVerify(int ok, X509_STORE_CTX *ctx);
int ssl_callback_SSLVerify_CRL(int ok, X509_STORE_CTX *ctx);

// GRIDSITE functions
int GRST_X509_check_issued_wrapper(X509_STORE_CTX *ctx, X509 *x, X509 *issuer)
/* We change the default callback to use our wrapper and discard errors
   due to GSI proxy chains (ie where users certs act as CAs) */
{
    int ret;
    ret = X509_check_issued(issuer, x);
    if (ret == X509_V_OK)
                return 1;
         
    /* Non self-signed certs without signing are ok if they passed
           the other checks inside X509_check_issued. Is this enough? */
    if ((ret == X509_V_ERR_KEYUSAGE_NO_CERTSIGN) &&
        (X509_NAME_cmp(X509_get_subject_name(issuer),
                           X509_get_subject_name(x)) != 0)) return 1;
 
    /* If we haven't asked for issuer errors don't set ctx */
#if OPENSSL_VERSION_NUMBER < 0x00908000
    if (!(ctx->flags & X509_V_FLAG_CB_ISSUER_CHECK)) return 0;
#else
    if (!(ctx->param->flags & X509_V_FLAG_CB_ISSUER_CHECK)) return 0;
#endif 
  
    ctx->error = ret;
    ctx->current_cert = x;
    ctx->current_issuer = issuer;
    return ctx->verify_cb(0, ctx);
}

/* Later OpenSSL versions add a second pointer ... */
int GRST_verify_cert_wrapper(X509_STORE_CTX *ctx, void *p)

/* Earlier ones have a single argument ... */
// int GRST_verify_cert_wrapper(X509_STORE_CTX *ctx)

/* Before 0.9.7 we cannot change the check_issued callback directly in
   the X509_STORE, so we must insert it in another callback that gets
   called early enough */
{
   ctx->check_issued = GRST_X509_check_issued_wrapper;
#ifdef R__SSL_GE_098
   X509_STORE_CTX_set_flags(ctx, X509_V_FLAG_ALLOW_PROXY_CERTS);
#endif
   return X509_verify_cert(ctx);
}

int GRST_callback_SSLVerify_wrapper(int ok, X509_STORE_CTX *ctx)
{
   SSL *ssl            = (SSL *) X509_STORE_CTX_get_app_data(ctx);
   int errnum          = X509_STORE_CTX_get_error(ctx);
   int errdepth        = X509_STORE_CTX_get_error_depth(ctx);
   int returned_ok;
   STACK_OF(X509) *certstack;
   GRSTx509Chain *grst_chain;
   GRSTx509Chain *grst_old_chain;

   /*
    * GSI Proxy user-cert-as-CA handling:
    * we skip Invalid CA errors at this stage, since we will check this
    * again at errdepth=0 for the full chain using GRSTx509ChainLoadCheck
    */
   if (errnum == X509_V_ERR_INVALID_CA)
     {
       GRSTerrorLog(GRST_LOG_DEBUG,"Skip invalid CA error, since we will check again at errdepth=0");
       ok = TRUE;
       errnum = X509_V_OK;
        X509_STORE_CTX_set_error(ctx, errnum);
     }

   /*
    * New style GSI Proxy handling, with critical ProxyCertInfo
    * extension: we use GRSTx509KnownCriticalExts() to check this
    */
#ifndef X509_V_ERR_UNHANDLED_CRITICAL_EXTENSION
#define X509_V_ERR_UNHANDLED_CRITICAL_EXTENSION 34
#endif
   if (errnum == X509_V_ERR_UNHANDLED_CRITICAL_EXTENSION)
     {
       if (GRSTx509KnownCriticalExts(X509_STORE_CTX_get_current_cert(ctx))
                                                              == GRST_RET_OK)
         {
	   GRSTerrorLog(GRST_LOG_DEBUG,"GRSTx509KnownCriticalExts() accepts previously unhandled Critical Extension(GSI Proxy?)");
	   ok = TRUE;
	   errnum = X509_V_OK;
	   X509_STORE_CTX_set_error(ctx, errnum);
         }
     }

   if (errnum == X509_V_ERR_INVALID_PURPOSE ) {
     GRSTerrorLog(GRST_LOG_DEBUG,"GRSTx509 invalid purpose error ignored ");
     ok =  TRUE;
     errnum = X509_V_OK;
     X509_STORE_CTX_set_error(ctx, errnum);
   }

   returned_ok = ssl_callback_SSLVerify(ok, ctx);

   /* in case ssl_callback_SSLVerify changed it */
   errnum = X509_STORE_CTX_get_error(ctx); 

   if ((errdepth == 0) && (errnum == X509_V_OK))
   /*
    * We've now got the last certificate - the identity being used for
    * this connection. At this point we check the whole chain for valid
    * CAs or, failing that, GSI-proxy validity using GRSTx509CheckChain.
    */
     {
        certstack = (STACK_OF(X509) *) X509_STORE_CTX_get_chain(ctx);

        errnum = GRSTx509ChainLoadCheck(&grst_chain, certstack, NULL,
                                        grst_cadir, 
                                        grst_vomsdir);

        if (errnum != X509_V_OK)
          {
	    GRSTerrorLog(GRST_LOG_ERR,"Invalid certificate chain reported by GRSTx509CheckChain() %s\n", X509_verify_cert_error_string(errnum));
            ok = FALSE;
          }
        else 
          {
	    GRSTerrorLog(GRST_LOG_DEBUG,"Valid certificate chain reported by GRSTx509ChainLoadCheck()\n");
          }

	// we don't free it but rather put it into the SSL context application data
	if ((grst_old_chain = SSL_get_app_data(ssl))) {
	  SSL_set_app_data(ssl,grst_chain);
	  GRSTerrorLog(GRST_LOG_INFO,"Free Chain %llx", grst_old_chain);
	  GRSTx509ChainFree(grst_old_chain);
	} else {
	  SSL_set_app_data(ssl,grst_chain);
	}
     }

   return returned_ok;
}

/*
    Save result validity info from chain into connection notes,
    and write out in an SSL session creds file.
*/

void GRST_print_ssl_creds(void *in_chain)
{
   int                       i= 0;
   int lowest_voms_delegation = 65535;
   GRSTx509Chain *grst_chain = (GRSTx509Chain*) in_chain;
   GRSTx509Cert  *grst_cert = NULL;

   /* check if already done */

   for (grst_cert = grst_chain->firstcert;
        grst_cert != NULL; grst_cert = grst_cert->next)
      {
        if (grst_cert->type == GRST_CERT_TYPE_VOMS)
          {
            /* want to record the delegation level 
               of the last proxy with VOMS attributes */
	    GRSTerrorLog(GRST_LOG_DEBUG,"Recording VOMS delegation %d\n",grst_cert->delegation);
            lowest_voms_delegation = grst_cert->delegation;
          }
        else if ((grst_cert->type == GRST_CERT_TYPE_EEC) ||
                 (grst_cert->type == GRST_CERT_TYPE_PROXY))
          {
	    GRSTerrorLog(GRST_LOG_INFO,"(%d) dn: %s\n",i,grst_cert->dn);
	    GRSTerrorLog(GRST_LOG_INFO,"notbefore=%ld notafter=%ld delegation=%d nist-loa=%d\n",grst_cert->notbefore,grst_cert->notafter,grst_cert->delegation);
	    ++i;
          }
      }
   
   for (grst_cert = grst_chain->firstcert; 
        grst_cert != NULL; grst_cert = grst_cert->next)
      {
        if ((grst_cert->type == GRST_CERT_TYPE_VOMS) &&
            (grst_cert->delegation == lowest_voms_delegation))
          {
            /* only export attributes from the last proxy to contain them */
	    GRSTerrorLog(GRST_LOG_INFO,"fqan:%s\n",grst_cert->value);
	    GRSTerrorLog(GRST_LOG_INFO,"notbefore=%ld notafter=%ld delegation=%d nist-loa=%d\n",grst_cert->notbefore,grst_cert->notafter,grst_cert->delegation);
            ++i;
          }
      }
}


char* GRST_get_voms_roles_and_free(void* in_chain)
{
  int          i, lowest_voms_delegation = 65535;
  GRSTx509Chain *grst_chain = (GRSTx509Chain*) in_chain;
  GRSTx509Cert  *grst_cert = NULL;

  char* voms_roles = (char*) malloc(16384);
  voms_roles[0] =0;

  if (!voms_roles) {
    return NULL;
  }

  /* check if already done */
  
  for (grst_cert = grst_chain->firstcert;
       grst_cert != NULL; grst_cert = grst_cert->next)
    {
      if (grst_cert->type == GRST_CERT_TYPE_VOMS)
	{
	  /* want to record the delegation level 
	     of the last proxy with VOMS attributes */
	  lowest_voms_delegation = grst_cert->delegation;
	}
      else if ((grst_cert->type == GRST_CERT_TYPE_EEC) ||
	       (grst_cert->type == GRST_CERT_TYPE_PROXY))
	{
	  ++i;
	}
    }
  
  for (grst_cert = grst_chain->firstcert; 
       grst_cert != NULL; grst_cert = grst_cert->next)
    {
      if ((grst_cert->type == GRST_CERT_TYPE_VOMS) &&
	  (grst_cert->delegation == lowest_voms_delegation))
	{
	  /* only export attributes from the last proxy to contain them */
	  GRSTerrorLog(GRST_LOG_DEBUG,"fqan:%s\n",grst_cert->value);

	  // filter the faulty roles out
	  //	  if (((strstr(grst_cert->value,"Role=NULL")))) {
	  //		//|| (strstr(grst_cert->value,"Capability=NULL"))))) {
          // filter out - don't concatenate
	  //	  } else {
	  
	  // don't filter, but leave it to the application to interpret the FQANs
	  strcat(voms_roles,grst_cert->value);
	  strcat(voms_roles,":");
	  //	  }
	  GRSTerrorLog(GRST_LOG_DEBUG,"notbefore=%ld notafter=%ld delegation=%d nist-loa=%d\n",grst_cert->notbefore,grst_cert->notafter,grst_cert->delegation);
	  ++i;
	}
    }
  
  if (strlen(voms_roles)) {
    // remove last :
    voms_roles[strlen(voms_roles)-1] = 0;
  }


  if (grst_chain) {
    GRSTerrorLog(GRST_LOG_INFO,"Free Chain %llx", grst_chain);
    GRSTx509ChainFree(grst_chain);
  }
  return voms_roles;
}

void GRST_free_chain(void* in_chain) {
  GRSTx509Chain *grst_chain = (GRSTx509Chain*) in_chain;
  if (grst_chain) 
    GRSTx509ChainFree(grst_chain);
}


/*                      _             _
**  _ __ ___   ___   __| |    ___ ___| |  mod_ssl
** | '_ ` _ \ / _ \ / _` |   / __/ __| |  Apache Interface to OpenSSL
** | | | | | | (_) | (_| |   \__ \__ \ |  www.modssl.org
** |_| |_| |_|\___/ \__,_|___|___/___/_|  ftp.modssl.org
**                      |_____|
**  ssl_engine_kernel.c
**  The SSL engine kernel
*/

/*
 * This OpenSSL callback function is called when OpenSSL
 * does client authentication and verifies the certificate chain.
 */
int ssl_callback_SSLVerify(int ok, X509_STORE_CTX *ctx)
{
  X509 *xs;
  int errnum;
  int errdepth;
  char *cp;
  char *cp2;
  int depth;

  /*
   * Get context back through OpenSSL context
   */
  SSL *ssl            = (SSL *) X509_STORE_CTX_get_app_data(ctx);
  
  /*
   * Get verify ingredients
   */
  xs       = X509_STORE_CTX_get_current_cert(ctx);
  errnum   = X509_STORE_CTX_get_error(ctx);
  errdepth = X509_STORE_CTX_get_error_depth(ctx);
  
  /*
   * Log verification information
   */
  cp  = X509_NAME_oneline(X509_get_subject_name(xs), NULL, 0);
  cp2 = X509_NAME_oneline(X509_get_issuer_name(xs),  NULL, 0);
  
  GRSTerrorLog(GRST_LOG_DEBUG,
	  "Certificate Verification: depth: %d, subject: %s, issuer: %s\n",
	  errdepth, cp != NULL ? cp : "-unknown-",
	  cp2 != NULL ? cp2 : "-unknown");
  

  if (cp)
    OPENSSL_free(cp);
  if (cp2)
    OPENSSL_free(cp2);
  
  /*
   * Check for optionally acceptable non-verifiable issuer situation
   */
  
  if (   (   errnum == X509_V_ERR_DEPTH_ZERO_SELF_SIGNED_CERT
	     || errnum == X509_V_ERR_SELF_SIGNED_CERT_IN_CHAIN
	     || errnum == X509_V_ERR_UNABLE_TO_GET_ISSUER_CERT_LOCALLY
#if SSL_LIBRARY_VERSION >= 0x00905000
	     || errnum == X509_V_ERR_CERT_UNTRUSTED
#endif
	     || errnum == X509_V_ERR_UNABLE_TO_VERIFY_LEAF_SIGNATURE  )
	 && grst_verify == GRST_VERIFY_OPTIONAL_NO_CA ) {
    GRSTerrorLog(GRST_LOG_ERR,
		 "Certificate Verification: Verifiable Issuer is configured as "
		 "optional, therefore we're accepting the certificate\n");
    SSL_set_verify_result(ssl, X509_V_OK);
    ok = TRUE;
  }

  /*
   * Additionally perform CRL-based revocation checks
   */
  
  if (ok) {
    ok = ssl_callback_SSLVerify_CRL(ok, ctx);
    if (!ok)
      errnum = X509_STORE_CTX_get_error(ctx);
  }
  
    /*
     * If we already know it's not ok, log the real reason
     */
  if (!ok) {
    GRSTerrorLog(GRST_LOG_ERR,"Certificate Verification: Error (%d): %s\n",
		 errnum, X509_verify_cert_error_string(errnum));
  }
  
  /*
   * Finally check the depth of the certificate verification
   */
  
  depth = grst_depth;
  
  if (errdepth > depth) {
    GRSTerrorLog(GRST_LOG_ERR,
		 "Certificate Verification: Certificate Chain too long "
		 "(chain has %d certificates, but maximum allowed are only %d)\n",
		 errdepth, depth);
    ok = FALSE;
  }
  
  /*
   * And finally signal OpenSSL the (perhaps changed) state
   */
  return (ok);
}

int ssl_callback_SSLVerify_CRL(
			       int ok, X509_STORE_CTX *ctx)
{
  X509_OBJECT obj;
  X509_NAME *subject;
  X509_NAME *issuer;
  X509 *xs;
  X509_CRL *crl;
  X509_REVOKED *revoked;
  EVP_PKEY *pubkey;
  long serial;
  int i, n, rc;
  char *cp;
  ASN1_TIME *t;
  
  /*
   * Determine certificate ingredients in advance
   */

  GRSTerrorLog(GRST_LOG_DEBUG,"Checking certificate revocation lists\n");


  xs      = X509_STORE_CTX_get_current_cert(ctx);
  subject = X509_get_subject_name(xs);
  issuer  = X509_get_issuer_name(xs);
  

  /*
   * OpenSSL provides the general mechanism to deal with CRLs but does not
   * use them automatically when verifying certificates, so we do it
   * explicitly here. We will check the CRL for the currently checked
   * certificate, if there is such a CRL in the store.
   *
   * We come through this procedure for each certificate in the certificate
   * chain, starting with the root-CA's certificate. At each step we've to
   * both verify the signature on the CRL (to make sure it's a valid CRL)
   * and it's revocation list (to make sure the current certificate isn't
   * revoked).  But because to check the signature on the CRL we need the
   * public key of the issuing CA certificate (which was already processed
   * one round before), we've a little problem. But we can both solve it and
   * at the same time optimize the processing by using the following
   * verification scheme (idea and code snippets borrowed from the GLOBUS
   * project):
   *
   * 1. We'll check the signature of a CRL in each step when we find a CRL
   *    through the _subject_ name of the current certificate. This CRL
   *    itself will be needed the first time in the next round, of course.
   *    But we do the signature processing one round before this where the
   * public key of the issuing CA certificate (which was already processed
   * one round before), we've a little problem. But we can both solve it and
   * at the same time optimize the processing by using the following
   * verification scheme (idea and code snippets borrowed from the GLOBUS
   * project):
   *
   * 1. We'll check the signature of a CRL in each step when we find a CRL
   *    through the _subject_ name of the current certificate. This CRL
   *    itself will be needed the first time in the next round, of course.
   *    But we do the signature processing one round before this where the
   *    public key of the CA is available.
   *
   * 2. We'll check the revocation list of a CRL in each step when
   *    we find a CRL through the _issuer_ name of the current certificate.
   *    This CRLs signature was then already verified one round before.
   *
   * This verification scheme allows a CA to revoke its own certificate as
   * well, of course.

   */

  if (!grst_store) {
    return 1;
  }


  /*
   * Try to retrieve a CRL corresponding to the _subject_ of
   * the current certificate in order to verify it's integrity.
   */
  memset((char *)&obj, 0, sizeof(obj));
  
  rc = SSL_X509_STORE_lookup(grst_store,X509_LU_CRL, subject, &obj);

  crl = obj.data.crl;
  if (rc > 0 && crl != NULL) {
    GRSTerrorLog(GRST_LOG_DEBUG,"CRL lookup ...");
    /*
     * Verify the signature on this CRL
     */
    pubkey = X509_get_pubkey(xs);
    if (X509_CRL_verify(crl, pubkey) <= 0) {
    GRSTerrorLog(GRST_LOG_ERR,"Invalid signature on CRL\n");
    X509_STORE_CTX_set_error(ctx, X509_V_ERR_CRL_SIGNATURE_FAILURE);
    X509_OBJECT_free_contents(&obj);
    if (pubkey != NULL)
      EVP_PKEY_free(pubkey);
    return 0;
    }
    
    if (pubkey != NULL)
      EVP_PKEY_free(pubkey);
    
    /*
     * Check date of CRL to make sure it's not expired
     */
    if ((t = X509_CRL_get_nextUpdate(crl)) == NULL) {
      GRSTerrorLog(GRST_LOG_ERR,"Found CRL has invalid enxtUpdate field\n");
      X509_STORE_CTX_set_error(ctx, X509_V_ERR_ERROR_IN_CRL_NEXT_UPDATE_FIELD);
      X509_OBJECT_free_contents(&obj);
      return 0;
    }
    if (X509_cmp_current_time(t) < 0) {
      GRSTerrorLog(GRST_LOG_ERR,"Found CRL is expired - "
		   "revoking all certificates until you get updated CRL\n");
      X509_STORE_CTX_set_error(ctx, X509_V_ERR_CRL_HAS_EXPIRED);
      X509_OBJECT_free_contents(&obj);
      return 0;
    }
    X509_OBJECT_free_contents(&obj);
  }
  
  /*
   * Try to retrieve a CRL corresponding to the _issuer_ of
   * the current certificate in order to check for revocation.
   */
  
  memset((char *)&obj, 0, sizeof(obj));
  rc = SSL_X509_STORE_lookup(grst_store,X509_LU_CRL, issuer, &obj);
  crl = obj.data.crl;
  if (rc > 0 && crl != NULL) {
    /*
     * Check if the current certificate is revoked by this CRL
     */
#if SSL_LIBRARY_VERSION < 0x00904000
    n = sk_num(X509_CRL_get_REVOKED(crl));
#else
    n = sk_X509_REVOKED_num(X509_CRL_get_REVOKED(crl));
#endif
    for (i = 0; i < n; i++) {
#if SSL_LIBRARY_VERSION < 0x00904000
      revoked = (X509_REVOKED *)sk_value(X509_CRL_get_REVOKED(crl), i);
#else
      revoked = sk_X509_REVOKED_value(X509_CRL_get_REVOKED(crl), i);
#endif
      if (ASN1_INTEGER_cmp(revoked->serialNumber, X509_get_serialNumber(xs)) == 0) {
	serial = ASN1_INTEGER_get(revoked->serialNumber);
	cp = X509_NAME_oneline(issuer, NULL, 0);
	GRSTerrorLog(GRST_LOG_ERR,
		     "Certificate with serial %ld (0x%lX) "
		     "revoked per CRL from issuer %s\n",
		     serial, serial, cp);
	OPENSSL_free(cp);
	
	X509_STORE_CTX_set_error(ctx, X509_V_ERR_CERT_REVOKED);
	X509_OBJECT_free_contents(&obj);
	return 0;
      }
    }
    X509_OBJECT_free_contents(&obj);
  }
  return 1;
}

/*  _________________________________________________________________
**
**  Certificate Revocation List (CRL) Storage
**  _________________________________________________________________
*/

X509_STORE *SSL_X509_STORE_create(char *cpFile, char *cpPath)
{
  X509_STORE *pStore;
  X509_LOOKUP *pLookup;
  
  if (cpFile == NULL && cpPath == NULL)
    return NULL;
  if ((pStore = X509_STORE_new()) == NULL)
    return NULL;
  if (cpFile != NULL) {
    if ((pLookup = X509_STORE_add_lookup(pStore, X509_LOOKUP_file())) == NULL) {
      X509_STORE_free(pStore);
      return NULL;
    }
    X509_LOOKUP_load_file(pLookup, cpFile, X509_FILETYPE_PEM);
  }
  if (cpPath != NULL) {
    if ((pLookup = X509_STORE_add_lookup(pStore, X509_LOOKUP_hash_dir())) == NULL) {
      X509_STORE_free(pStore);
      return NULL;
    }
    X509_LOOKUP_add_dir(pLookup, cpPath, X509_FILETYPE_PEM);
  }
  return pStore;
}

int SSL_X509_STORE_lookup(X509_STORE *pStore, int nType,
                          X509_NAME *pName, X509_OBJECT *pObj)
{
  X509_STORE_CTX pStoreCtx;
  int rc;
  
  X509_STORE_CTX_init(&pStoreCtx, pStore, NULL, NULL);
  rc = X509_STORE_get_by_subject(&pStoreCtx, nType, pName, pObj);
  X509_STORE_CTX_cleanup(&pStoreCtx);
  return rc;
}
