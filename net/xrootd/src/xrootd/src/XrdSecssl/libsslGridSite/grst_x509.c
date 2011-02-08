/*
   Copyright (c) 2002-7, Andrew McNab, University of Manchester
   All rights reserved.

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

   ---------------------------------------------------------------
    For more information about GridSite: http://www.gridsite.org/
   ---------------------------------------------------------------
*/ 

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <stdio.h>
#include <unistd.h>       
#include <stdlib.h>
#include <stdarg.h>
#include <time.h>
#include <stdarg.h>
#include <dirent.h>
#include <string.h>
#include <strings.h>
#include <pwd.h>
#include <errno.h>
#include <getopt.h>
#include <pthread.h>

#include <sys/types.h>
#include <sys/stat.h>

#ifndef GRST_NO_OPENSSL
#include <openssl/x509.h>
#include <openssl/x509v3.h>
#include <openssl/err.h>
#include <openssl/pem.h>

#include <openssl/rsa.h>
#include <openssl/pem.h>
#include <openssl/err.h>
#include <openssl/evp.h>
#include <openssl/bio.h>    
#include <openssl/des.h>    
#include <openssl/rand.h>
#endif

#include "gridsite.h"

#define GRST_KEYSIZE		512
#define GRST_PROXYCACHE		"/../proxycache/"
#define GRST_MAX_CHAIN_LEN	9
#define GRST_BACKDATE_SECONDS	300

pthread_mutex_t vomsmutex;

/// Compare X509 Distinguished Name strings
int GRSTx509NameCmp(char *a, char *b)
///
/// This function attempts to do with string representations what
/// would ideally be done with OIDs/values. In particular, we equate
/// "/Email=" == "/emailAddress=" to deal with this important change
/// between OpenSSL 0.9.6 and 0.9.7. 
/// Other than that, it is currently the same as ordinary strcasecmp(3)
/// (for consistency with EDG/LCG/EGEE gridmapdir case insensitivity.)
{
   int   ret;
   char *aa, *bb, *p;

   if ((a == NULL) || (b == NULL)) return 1; /* NULL never matches */

   aa = strdup(a);
   while ((p = strstr(aa, "/emailAddress=")) != NULL)
        {
          memmove(&p[6], &p[13], strlen(&p[13]) + 1);
          p[1] = 'E';
        }

   bb = strdup(b);
   while ((p = strstr(bb, "/emailAddress=")) != NULL)
        {
          memmove(&p[6], &p[13], strlen(&p[13]) + 1);
          p[1] = 'E';
        }

   ret = strcasecmp(aa, bb);

   free(aa);
   free(bb);
                                                                                
   return ret;
}


/// Check critical extensions
int GRSTx509KnownCriticalExts(X509 *cert)
///
/// Returning GRST_RET_OK if all of extensions are known to us or 
/// OpenSSL; GRST_REF_FAILED otherwise.   
///
/// Since this function relies on functionality (X509_supported_extension)
/// introduced in 0.9.7, then we do nothing and report an error 
/// (GRST_RET_FAILED) if one of the associated defines 
/// (X509_V_ERR_UNHANDLED_CRITICAL_EXTENSION) is absent.
{
   int  i;
   char s[80];
   X509_EXTENSION *ex;
   
#ifdef X509_V_ERR_UNHANDLED_CRITICAL_EXTENSION        
   for (i = 0; i < X509_get_ext_count(cert); ++i)
      {
        ex = X509_get_ext(cert, i);

        if (X509_EXTENSION_get_critical(ex) &&
                                 !X509_supported_extension(ex))
          {
            OBJ_obj2txt(s, sizeof(s), X509_EXTENSION_get_object(ex), 1);
            if ( (strcmp(s, GRST_PROXYCERTINFO_OID) != 0) && 
		 (strcmp(s, GRST_PROXYCERTNEWINFO_OID) != 0) )
	      return GRST_RET_FAILED;
          }
      }

   return GRST_RET_OK;
#else
   return GRST_RET_FAILED;
#endif
}

/// Check if certificate can be used as a CA to sign standard X509 certs
int GRSTx509IsCA(X509 *cert)
///
/// Return GRST_RET_OK if true; GRST_RET_FAILED if not.
{
   int purpose_id;

   purpose_id = X509_PURPOSE_get_by_sname("sslclient");

   /* final argument to X509_check_purpose() is whether to check for CAness */   
   if (X509_check_purpose(cert, purpose_id + X509_PURPOSE_MIN, 1))
        return GRST_RET_OK;
   else return GRST_RET_FAILED;
}   

int GRSTx509ChainFree(GRSTx509Chain *chain)
{
   GRSTx509Cert *grst_cert, *next_grst_cert;

   if (chain == NULL) return GRST_RET_OK;

   next_grst_cert = chain->firstcert; 
   
   while (next_grst_cert != NULL)
      {
        grst_cert = next_grst_cert;
        
        if (grst_cert->issuer != NULL) free(grst_cert->issuer);
        if (grst_cert->dn     != NULL) free(grst_cert->dn);
        if (grst_cert->ocsp   != NULL) free(grst_cert->ocsp);
        
        next_grst_cert = grst_cert->next;
        free(grst_cert);        
      }

   free(chain);

   return GRST_RET_OK;
}

/// Check a specific signature against a specific (VOMS) cert
static int GRSTx509VerifySig(time_t *time1_time, time_t *time2_time,
                             unsigned char *txt, int txt_len,
                             unsigned char *sig, int sig_len, 
                             X509 *cert)
///
/// Returns GRST_RET_OK if signature is ok, other values if not.
{   
   int            ret;
   EVP_PKEY      *prvkey;
   EVP_MD_CTX     ctx;
   time_t         voms_service_time1, voms_service_time2;

   prvkey = X509_extract_key(cert);
   if (prvkey == NULL) {
     return GRST_RET_FAILED;
   }
            
   OpenSSL_add_all_digests();
#if OPENSSL_VERSION_NUMBER >= 0x0090701fL
   EVP_MD_CTX_init(&ctx);
   EVP_VerifyInit_ex(&ctx, EVP_md5(), NULL);
#else
   EVP_VerifyInit(&ctx, EVP_md5());
#endif
          
   EVP_VerifyUpdate(&ctx, txt, txt_len);

   ret = EVP_VerifyFinal(&ctx, sig, sig_len, prvkey);

#if OPENSSL_VERSION_NUMBER >= 0x0090701fL
   EVP_MD_CTX_cleanup(&ctx);      
#endif
   EVP_PKEY_free(prvkey);

   if (ret != 1) {
     return GRST_RET_FAILED;
   }

   voms_service_time1 =
           GRSTasn1TimeToTimeT(ASN1_STRING_data(X509_get_notBefore(cert)),0);
          if (voms_service_time1 > *time1_time) 
                             *time1_time = voms_service_time1; 
           
   voms_service_time2 =
           GRSTasn1TimeToTimeT(ASN1_STRING_data(X509_get_notAfter(cert)),0);
          if (voms_service_time2 < *time1_time) 
                             *time2_time = voms_service_time2;

   return GRST_RET_OK ; /* verified */
}

/// Check the signature of the VOMS attributes
static int GRSTx509VerifyVomsSig(time_t *time1_time, time_t *time2_time,
                                 unsigned char *asn1string, 
                                 struct GRSTasn1TagList taglist[], 
                                 int lasttag,
                                 char *vomsdir, int acnumber)
///
/// Returns GRST_RET_OK if signature is ok, other values if not.
{   
#define GRST_ASN1_COORDS_VOMS_DN   "-1-1-%d-1-3-1-1-1-%%d-1-%%d"
#define GRST_ASN1_COORDS_VOMS_INFO "-1-1-%d-1"
#define GRST_ASN1_COORDS_VOMS_SIG  "-1-1-%d-3"
   int            isig, iinfo;
   char           certpath[16384], certpath2[16384], acvomsdn[200], dn_coords[200],
                  info_coords[200], sig_coords[200];

   DIR           *vomsDIR, *vomsDIR2;
   struct dirent *vomsdirent, *vomsdirent2;
   X509          *cert;

   FILE          *fp;
   struct stat    statbuf;
   static int    voms_mutex_init=0;

   if (!voms_mutex_init) {
     // initialize a protection mutex
     voms_mutex_init = 1;
     pthread_mutex_init(&vomsmutex,NULL);
   }

   pthread_mutex_lock(&vomsmutex);

   if ((vomsdir == NULL) || (vomsdir[0] == '\0')) {
     pthread_mutex_unlock(&vomsmutex);
     return GRST_RET_FAILED;
   }

   snprintf(dn_coords, sizeof(dn_coords),
            GRST_ASN1_COORDS_VOMS_DN, acnumber);
   
   if (GRSTasn1GetX509Name(acvomsdn, sizeof(acvomsdn), dn_coords,
			   (char*)asn1string, taglist, lasttag) != GRST_RET_OK) {
     pthread_mutex_unlock(&vomsmutex);
     return GRST_RET_FAILED;
   }
         
   snprintf(info_coords, sizeof(info_coords),
            GRST_ASN1_COORDS_VOMS_INFO, acnumber);
   iinfo = GRSTasn1SearchTaglist(taglist, lasttag, info_coords);

   snprintf(sig_coords, sizeof(sig_coords), 
            GRST_ASN1_COORDS_VOMS_SIG, acnumber);
   isig  = GRSTasn1SearchTaglist(taglist, lasttag, sig_coords);

   if ((iinfo < 0) || (isig < 0)) {
     pthread_mutex_unlock(&vomsmutex);
     return GRST_RET_FAILED;
   }

   vomsDIR = opendir(vomsdir);
   if (vomsDIR == NULL) {
     pthread_mutex_unlock(&vomsmutex);
     return GRST_RET_FAILED;
   }

   while ((vomsdirent = readdir(vomsDIR)) != NULL)
        {
          if (vomsdirent->d_name[0] == '.') continue;
        
          sprintf(certpath, "%s/%s", vomsdir, vomsdirent->d_name);
          stat(certpath, &statbuf);
          
          if (S_ISDIR(statbuf.st_mode))
            {
              vomsDIR2 = opendir(certpath);
              GRSTerrorLog(GRST_LOG_DEBUG, 
                           "Descend VOMS subdirectory %s", certpath);
              
              if (vomsDIR2 == NULL) continue;
                

              while ((vomsdirent2 = readdir(vomsDIR2)) != NULL)
                {
                  if (vomsdirent2->d_name[0] == '.') continue;
                  
                  sprintf(certpath2, "%s/%s/%s", 
                           vomsdir, vomsdirent->d_name, vomsdirent2->d_name);
                
                  fp = fopen(certpath2, "r");
                  GRSTerrorLog(GRST_LOG_DEBUG, 
                               "Examine VOMS cert %s", certpath2);
                  if (fp == NULL) continue;

                  cert = PEM_read_X509(fp, NULL, NULL, NULL);
                  fclose(fp);
                  if (cert == NULL) continue;

                  if (GRSTx509VerifySig(time1_time, time2_time,
                            &asn1string[taglist[iinfo].start], 
                            taglist[iinfo].length+taglist[iinfo].headerlength,
                            &asn1string[taglist[isig].start+
                                                taglist[isig].headerlength+1],
                            taglist[isig].length - 1,
                            cert) == GRST_RET_OK)
                    {
                      GRSTerrorLog(GRST_LOG_DEBUG, " VOMS cert signature match");
                      X509_free(cert);
                      closedir(vomsDIR2);
                      closedir(vomsDIR);
		      pthread_mutex_unlock(&vomsmutex);
		      return GRST_RET_OK ; /* verified */
                    }
            
                  X509_free(cert);
                }
                
              closedir(vomsDIR2);
            }
          else
            {
              fp = fopen(certpath, "r");
              GRSTerrorLog(GRST_LOG_DEBUG, "Examine VOMS cert %s", certpath);
              if (fp == NULL) continue;

              cert = PEM_read_X509(fp, NULL, NULL, NULL);
              fclose(fp);
              if (cert == NULL) continue;

              if (GRSTx509VerifySig(time1_time, time2_time,
                            &asn1string[taglist[iinfo].start], 
                            taglist[iinfo].length+taglist[iinfo].headerlength,
                            &asn1string[taglist[isig].start+
                                                taglist[isig].headerlength+1],
                            taglist[isig].length - 1,
                            cert) == GRST_RET_OK)
                {
                  X509_free(cert);
                  closedir(vomsDIR);
		  pthread_mutex_unlock(&vomsmutex);
                  return GRST_RET_OK ; /* verified */
                }
            
              X509_free(cert);
            }
        }

   closedir(vomsDIR);   
   pthread_mutex_unlock(&vomsmutex);
   return GRST_RET_FAILED;
}

/// Get the VOMS attributes in the given extension
static int GRSTx509ChainVomsAdd(GRSTx509Cert **grst_cert, 
                         time_t time1_time, time_t time2_time,
                         X509_EXTENSION *ex, 
                         char *ucuserdn, char *vomsdir)
///
/// Add any VOMS credentials found into the chain. Always returns GRST_RET_OK
/// - even for invalid credentials, which are flagged in errors field
{
#define MAXTAG 500
#define GRST_ASN1_COORDS_FQAN    "-1-1-%d-1-7-1-2-1-2-%d"
#define GRST_ASN1_COORDS_USER_DN "-1-1-%d-1-2-1-1-1-1-%%d-1-%%d"
#define GRST_ASN1_COORDS_VOMS_DN "-1-1-%d-1-3-1-1-1-%%d-1-%%d"
#define GRST_ASN1_COORDS_TIME1   "-1-1-%d-1-6-1"
#define GRST_ASN1_COORDS_TIME2   "-1-1-%d-1-6-2"
   ASN1_OCTET_STRING *asn1data;
   char              *asn1string, acuserdn[200], acvomsdn[200],
                      dn_coords[200], fqan_coords[200], time1_coords[200],
                      time2_coords[200];
   long               asn1length;
   int                lasttag=-1, itag, i, acnumber = 1, chain_errors = 0;
   struct GRSTasn1TagList taglist[MAXTAG+1];
   time_t             actime1 = 0, actime2 = 0, time_now;

   asn1data   = X509_EXTENSION_get_data(ex);
   asn1string = (char*)ASN1_STRING_data(asn1data);
   asn1length = ASN1_STRING_length(asn1data);

   GRSTasn1ParseDump(NULL, (unsigned char*) asn1string, asn1length, taglist, MAXTAG, &lasttag);

   for (acnumber = 1; ; ++acnumber) /* go through ACs one by one */
      {
        chain_errors = 0;
      
        snprintf(dn_coords, sizeof(dn_coords), GRST_ASN1_COORDS_USER_DN, acnumber);
        if (GRSTasn1GetX509Name(acuserdn, sizeof(acuserdn), dn_coords,
                                asn1string, taglist, lasttag) != GRST_RET_OK)
                             break;

        snprintf(dn_coords, sizeof(dn_coords), GRST_ASN1_COORDS_VOMS_DN, acnumber);
        if (GRSTasn1GetX509Name(acvomsdn, sizeof(acvomsdn), dn_coords,
                                asn1string, taglist, lasttag) != GRST_RET_OK)
                             break;

        if (GRSTx509NameCmp(ucuserdn, acuserdn) != 0)
                             chain_errors |= GRST_CERT_BAD_CHAIN;

        if (GRSTx509VerifyVomsSig(&time1_time, &time2_time,
                              (unsigned char*)asn1string, taglist, lasttag, vomsdir, acnumber)
                              != GRST_RET_OK)
                             chain_errors |= GRST_CERT_BAD_SIG; 

        snprintf(time1_coords, sizeof(time1_coords), GRST_ASN1_COORDS_TIME1, acnumber);
        itag = GRSTasn1SearchTaglist(taglist, lasttag, time1_coords);
        
        if (itag > -1) actime1 = GRSTasn1TimeToTimeT((unsigned char*)
                                   &asn1string[taglist[itag].start+
                                               taglist[itag].headerlength],
                                   taglist[itag].length);
        else actime1 = 0;
        
        snprintf(time2_coords, sizeof(time2_coords), GRST_ASN1_COORDS_TIME2, acnumber);
        itag = GRSTasn1SearchTaglist(taglist, lasttag, time2_coords);
        
        if (itag > -1) actime2 = GRSTasn1TimeToTimeT((unsigned char*)
                                   &asn1string[taglist[itag].start+
                                               taglist[itag].headerlength],
                                   taglist[itag].length);
        else actime2 = 0;
        
        if (actime1 > time1_time) time1_time = actime1;
        if (actime2 < time2_time) time2_time = actime2;

        time(&time_now);
        if ((time1_time > time_now + 300) || (time2_time < time_now))
               chain_errors |= GRST_CERT_BAD_TIME;

        for (i=1; ; ++i) /* now go through FQANs */
           {
             snprintf(fqan_coords, sizeof(fqan_coords), GRST_ASN1_COORDS_FQAN, acnumber, i);
             itag = GRSTasn1SearchTaglist(taglist, lasttag, fqan_coords);

             if (itag > -1)
               {
                 (*grst_cert)->next = malloc(sizeof(GRSTx509Cert));
                 *grst_cert = (*grst_cert)->next;
                 bzero(*grst_cert, sizeof(GRSTx509Cert));
               
                 (*grst_cert)->notbefore = time1_time;
                 (*grst_cert)->notafter  = time2_time;
                 sprintf(((*grst_cert)->value), "%.*s",
                          taglist[itag].length,
                          &asn1string[taglist[itag].start+
                                      taglist[itag].headerlength]);
                                      
                 (*grst_cert)->errors = chain_errors; /* ie may be invalid */
                 (*grst_cert)->type = GRST_CERT_TYPE_VOMS;
                 (*grst_cert)->issuer = strdup(acvomsdn);
                 (*grst_cert)->dn = strdup(acuserdn);
               }
             else break;
           }
      }
      
   return GRST_RET_OK;
}

/// Check certificate chain for GSI proxy acceptability.
int GRSTx509ChainLoadCheck(GRSTx509Chain **chain, 
                           STACK_OF(X509) *certstack, X509 *lastcert,
                           char *capath, char *vomsdir)
///
/// Returns GRST_RET_OK if valid; OpenSSL X509 errors otherwise.
///
/// The GridSite version handles old and new style Globus proxies, and
/// proxies derived from user certificates issued with "X509v3 Basic
/// Constraints: CA:FALSE" (eg UK e-Science CA)
///
/// TODO: we do not yet check ProxyCertInfo and ProxyCertPolicy extensions
///       (although via GRSTx509KnownCriticalExts() we can accept them.)
{
   X509 *cert;                  /* Points to the current cert in the loop */
   X509 *cacert = NULL;         /* The CA root cert */
   int depth = 0;               /* Depth of cert chain */
   int chain_errors = 0;	/* records previous errors */
   int first_non_ca;		/* number of the EEC issued to user by CA */
   char *ucuserdn = NULL;	/* DN of EEC issued to user by CA */
   size_t len,len2;             /* Lengths of issuer and cert DN */
   int IsCA;                    /* Holds whether cert is allowed to sign */
   int prevIsCA;                /* Holds whether previous cert in chain is 
                                   allowed to sign */
   int prevIsLimited;		/* previous cert was proxy and limited */
   int i,j,ret;                 /* Iteration/temp variables */
   char *proxy_part_DN;         /* Pointer to end part of current-cert-in-chain
                                   maybe eg "/CN=proxy" */
   char s[80];
   char cacertpath[16384];
   unsigned long subjecthash = 0;	/* hash of the name of first cert */
   unsigned long issuerhash = 0;	/* hash of issuer name of first cert */
   FILE *fp;
   X509_EXTENSION *ex;
   time_t now;
   GRSTx509Cert *grst_cert, *new_grst_cert;
   grst_cert = NULL;
   new_grst_cert = NULL;
   GRSTerrorLog(GRST_LOG_DEBUG, "GRSTx509ChainLoadCheck() starts");

   time(&now);

   first_non_ca = 0; /* set to something predictable if things fail */
 
   /* Set necessary preliminary values */
   IsCA          = TRUE;           /* =prevIsCA - start from a CA */
   prevIsLimited = 0;
 
   /* Get the client cert chain */
   if (certstack != NULL) 
     depth = sk_X509_num(certstack); /* How deep is that chain? */
   
   if ((depth == 0) && (lastcert == NULL)) 
     {
       *chain = NULL;
       return GRST_RET_FAILED;
     }

   cert = sk_X509_value(certstack, depth - 1);
   subjecthash = X509_NAME_hash(X509_get_subject_name(cert));
   issuerhash = X509_NAME_hash(X509_get_issuer_name(cert));
   sprintf(cacertpath, "%s/%.8x.0", capath, (int)issuerhash);
   
   GRSTerrorLog(GRST_LOG_DEBUG, "Look for CA root file %s", cacertpath);

   fp = fopen(cacertpath, "r");

   if (fp == NULL) chain_errors |= GRST_CERT_BAD_CHAIN;
   else
     {
       cacert = PEM_read_X509(fp, NULL, NULL, NULL);
       fclose(fp);
       if (cacert != NULL) { 
	 GRSTerrorLog(GRST_LOG_DEBUG, "Loaded CA root cert from file");
       } else {
	 GRSTerrorLog(GRST_LOG_DEBUG, "Failed to load CA root cert file");
       }
     }

   //   *chain = (GRSTx509Chain*) malloc(sizeof(GRSTx509Chain));
   *chain = malloc(sizeof(GRSTx509Chain));
   bzero(*chain, sizeof(GRSTx509Chain));

   /* Check the client chain */
   for (i = depth - ((subjecthash == issuerhash) ? 1 : 0);
        i >= ((lastcert == NULL) ? 0 : -1); 
        --i) 
      /* loop through client-presented chain starting at CA end */
      {
        GRSTerrorLog(GRST_LOG_DEBUG, "Process cert at depth %d in chain", i);

        prevIsCA=IsCA;

        new_grst_cert = malloc(sizeof(GRSTx509Cert));
        bzero(new_grst_cert, sizeof(GRSTx509Cert));
        new_grst_cert->errors = chain_errors;
        
        if ((*chain)->firstcert == NULL)
          {
            GRSTerrorLog(GRST_LOG_DEBUG, "Initialise chain");
            (*chain)->firstcert = new_grst_cert;
          }
        else grst_cert->next = new_grst_cert;

        grst_cert = new_grst_cert;

        /* Choose X509 certificate and point to it with 'cert' */
        if (i < 0) cert = lastcert;
        else if (i == depth)
             cert = cacert; /* the self-signed CA from the store*/
        else if ((i == depth - 1) && (subjecthash == issuerhash))
             cert = cacert; /* ie claims to be a copy of a self-signed CA */
        else cert = sk_X509_value(certstack, i);

        if (cert != NULL)
          {
            if ((i == depth - 1) && (subjecthash != issuerhash))
              {
                /* if first cert does not claim to be a self-signed copy 
                   of a CA root cert in the store, we check the signature */

                if (cacert == NULL)
                  {
                    chain_errors |= GRST_CERT_BAD_CHAIN;
                    ret = X509_V_ERR_UNABLE_TO_GET_ISSUER_CERT;
                  }
                else 
                  {
                    ret = X509_check_issued(cacert, cert);

                    GRSTerrorLog(GRST_LOG_DEBUG, 
                             "Cert sig check %d returns %d", i, ret);

                    if (ret != X509_V_OK) 
                             new_grst_cert->errors |= GRST_CERT_BAD_SIG;
                  }
              }
            else if ((i == depth - 2) && (subjecthash == issuerhash))
              {
                /* first cert claimed to be a self-signed copy of a CA root
                cert in the store, we check the signature of the second
                cert, using OUR copy of the CA cert DIRECT from the store */

                if (cacert == NULL)
                  {
                    chain_errors |= GRST_CERT_BAD_CHAIN;
                    ret = X509_V_ERR_UNABLE_TO_GET_ISSUER_CERT;
                  }
                else 
                  {
                    ret = X509_check_issued(cacert, cert);
                
                    GRSTerrorLog(GRST_LOG_DEBUG, 
                             "Cert sig check %d returns %d", i, ret);
                
                    if (ret != X509_V_OK)
                             new_grst_cert->errors |= GRST_CERT_BAD_SIG;
                  }
              }
            else if (i < depth - 1)
              {
                /* otherwise a normal part of the chain: note that if the
                   first cert claims to be a self-signed copy of a CA root
                   cert in the store, we never use it for sig checking */
              
                ret = X509_check_issued(sk_X509_value(certstack, i + 1), cert);
                
                GRSTerrorLog(GRST_LOG_DEBUG, 
                             "Cert sig check %d returns %d", i, ret);

                if ((ret != X509_V_OK) &&
                    (ret != X509_V_ERR_KEYUSAGE_NO_CERTSIGN))                
                          new_grst_cert->errors |= GRST_CERT_BAD_SIG;
                          
                /* NO_CERTSIGN can still be ok due to Proxy Certificates */
              }

            new_grst_cert->serial = (int) ASN1_INTEGER_get(
                               X509_get_serialNumber(cert));
            new_grst_cert->notbefore = GRSTasn1TimeToTimeT(
                               ASN1_STRING_data(X509_get_notBefore(cert)), 0);
            new_grst_cert->notafter  = GRSTasn1TimeToTimeT(
                               ASN1_STRING_data(X509_get_notAfter(cert)), 0);
          
            /* we check times and record if invalid */
          
            if (now < new_grst_cert->notbefore)
                 new_grst_cert->errors |= GRST_CERT_BAD_TIME;

            if (now > new_grst_cert->notafter)
                 new_grst_cert->errors |= GRST_CERT_BAD_TIME;

            new_grst_cert->dn = X509_NAME_oneline(X509_get_subject_name(cert),NULL,0);
            new_grst_cert->issuer = X509_NAME_oneline(X509_get_issuer_name(cert),NULL,0);
            len       = strlen(new_grst_cert->dn);
            len2      = strlen(new_grst_cert->issuer);

            /* always treat a first cert from the CA files as a 
               CA: this is really for lousy CAs that dont create 
               proper v3 root certificates */
                
            if (i == depth) 
	      IsCA = TRUE;
            else 
	      IsCA = (GRSTx509IsCA(cert) == GRST_RET_OK);

            /* If any forebear certificate is not allowed to sign we must 
               assume all decendents are proxies and cannot sign either */
            if (prevIsCA)
              {
                if (IsCA)
                  {               
                    new_grst_cert->type = GRST_CERT_TYPE_CA;
                  }
                else 
                  {
                    new_grst_cert->type = GRST_CERT_TYPE_EEC;
                    first_non_ca = i;
                    ucuserdn = new_grst_cert->dn;
                    new_grst_cert->delegation 
                       = (lastcert == NULL) ? i : i + 1;
                  }
              } 
            else 
              {
                new_grst_cert->type = GRST_CERT_TYPE_PROXY;

                IsCA = FALSE;
                /* Force proxy check next iteration. Important because I can
                   sign any CA I create! */

                new_grst_cert->delegation = (lastcert == NULL) ? i : i + 1;
              }

            if (!prevIsCA)
              {
                /* issuer didn't have CA status, so this is (at best) a proxy:
                   check for bad proxy extension*/

                if (prevIsLimited) /* we reject proxies of limited proxies! */
                  {
                    new_grst_cert->errors |= GRST_CERT_BAD_CHAIN;
                    chain_errors |= GRST_CERT_BAD_CHAIN;
                  }
              
                /* User not allowed to sign shortened DN */
                if (len2 > len) 
                  {
                    new_grst_cert->errors |= GRST_CERT_BAD_CHAIN;
                    chain_errors |= GRST_CERT_BAD_CHAIN;
                  }
                  
                /* Proxy subject must begin with issuer. */
                if (strncmp((const char*)new_grst_cert->dn, new_grst_cert->issuer, len2) != 0) 
                  {
                    new_grst_cert->errors  |= GRST_CERT_BAD_CHAIN;
                    chain_errors |= GRST_CERT_BAD_CHAIN;
                  }

                /* Set pointer to end of base DN in cert_DN */
                proxy_part_DN = &(new_grst_cert->dn[len2]);

                /* First attempt at support for Old and New style GSI
                   proxies: /CN=anything is ok for now */
                if (strncmp((const char*)proxy_part_DN, "/CN=", 4) != 0)
                  {
                    new_grst_cert->errors  |= GRST_CERT_BAD_CHAIN;
                    chain_errors |= GRST_CERT_BAD_CHAIN;
                  }

                if (strncmp((const char*)proxy_part_DN, "/CN=limited proxy", 17) == 0)
                        prevIsLimited = 1; /* ready for next cert ... */

                for (j=0; j < X509_get_ext_count(cert); ++j)
                   {
                     ex = X509_get_ext(cert, j);
                     OBJ_obj2txt(s,sizeof(s),X509_EXTENSION_get_object(ex),1);

                     if (strcmp(s, GRST_VOMS_OID) == 0) /* a VOMS extension */
                       {
                         GRSTx509ChainVomsAdd(&grst_cert, 
                                              new_grst_cert->notbefore,
                                              new_grst_cert->notafter,                                              
                                              ex,
                                              ucuserdn, 
                                              vomsdir);
                         grst_cert->delegation = (lastcert == NULL) ? i : i+1;
                       }
                   }
                        
              } 
          }
          

      } /* end of for loop */

   if (cacert != NULL) X509_free(cacert);
 
   return GRST_RET_OK;
}

/// Check certificate chain for GSI proxy acceptability.
int GRSTx509CheckChain(int *first_non_ca, X509_STORE_CTX *ctx)
///
/// Returns X509_V_OK/GRST_RET_OK if valid; OpenSSL X509 errors otherwise.
///
/// Inspired by GSIcheck written by Mike Jones, SVE, Manchester Computing,
/// The University of Manchester.
///
/// The GridSite version handles old and new style Globus proxies, and
/// proxies derived from user certificates issued with "X509v3 Basic
/// Constraints: CA:FALSE" (eg UK e-Science CA)
///
/// We do not check chain links between certs here: this is done by
/// GRST_check_issued/X509_check_issued in mod_ssl's ssl_engine_init.c
///
/// TODO: we do not yet check ProxyCertInfo and ProxyCertPolicy extensions
///       (although via GRSTx509KnownCriticalExts() we can accept them.)
{
   STACK_OF(X509) *certstack;   /* Points to the client's cert chain */
   X509 *cert;                  /* Points to the client's cert */
   int depth;                   /* Depth of cert chain */
   size_t len,len2;             /* Lengths of issuer and cert DN */
   int IsCA;                    /* Holds whether cert is allowed to sign */
   int prevIsCA=TRUE;           /* Holds whether previous cert in chain is 
                                   allowed to sign */
   int prevIsLimited;		/* previous cert was proxy and limited */
   int i;                       /* Iteration variables */
   char *cert_DN;               /* Pointer to current-certificate-in-chain's 
                                   DN */
   char *issuer_DN;             /* Pointer to 
                                   issuer-of-current-cert-in-chain's DN */
   char *proxy_part_DN;         /* Pointer to end part of current-cert-in-chain
                                   maybe eg "/CN=proxy" */
   time_t now;
   
   time(&now);

   *first_non_ca = 0; /* set to something predictable if things fail */

   /* Check for context */
   if (!ctx) return X509_V_ERR_INVALID_CA; 
     /* Can't GSI-verify if there is no context. Here and throughout this
        function we report all errors as X509_V_ERR_INVALID_CA. */
 
   /* Set necessary preliminary values */
   IsCA          = TRUE;           /* =prevIsCA - start from a CA */
   prevIsLimited = 0;
 
   /* Get the client cert chain */
   certstack = X509_STORE_CTX_get_chain(ctx);     /* Get the client's chain  */
   depth     = sk_X509_num(certstack);            /* How deep is that chain? */
 
   /* Check the client chain */
   for (i=depth-1; i >= 0; --i) 
      /* loop through client-presented chain starting at CA end */
      {
        prevIsCA=IsCA;

        /* Check for X509 certificate and point to it with 'cert' */
        if ( (cert = sk_X509_value(certstack, i)) )
          {
            /* we check times and reject immediately if invalid */
          
            if (now <
           GRSTasn1TimeToTimeT(ASN1_STRING_data(X509_get_notBefore(cert)),0))
                  return X509_V_ERR_INVALID_CA;
                
            if (now > 
           GRSTasn1TimeToTimeT(ASN1_STRING_data(X509_get_notAfter(cert)),0))
                  return X509_V_ERR_INVALID_CA;

            /* If any forebear certificate is not allowed to sign we must 
               assume all decendents are proxies and cannot sign either */
            if (prevIsCA)
              {
                /* always treat the first cert (from the CA files) as a CA */
                if (i == depth-1) IsCA = TRUE;
                /* check if this cert is valid CA for signing certs */
                else IsCA = (GRSTx509IsCA(cert) == GRST_RET_OK);
                
                if (!IsCA) *first_non_ca = i;
              } 
            else 
              {
                IsCA = FALSE; 
                /* Force proxy check next iteration. Important because I can
                   sign any CA I create! */
              }
 
            cert_DN   = X509_NAME_oneline(X509_get_subject_name(cert),NULL,0);
            issuer_DN = X509_NAME_oneline(X509_get_issuer_name(cert),NULL,0);
            len       = strlen(cert_DN);
            len2      = strlen(issuer_DN);

            /* issuer didn't have CA status, so this is (at best) a proxy:
               check for bad proxy extension*/

            if (!prevIsCA)
              {
                if (prevIsLimited) /* we reject proxies of limited proxies! */
                                return X509_V_ERR_INVALID_CA;
              
                /* User not allowed to sign shortened DN */
                if (len2 > len) return X509_V_ERR_INVALID_CA;                           
                  
                /* Proxy subject must begin with issuer. */
                if (strncmp((const char*)cert_DN, issuer_DN, len2) != 0) 
                              return X509_V_ERR_INVALID_CA;

                /* Set pointer to end of base DN in cert_DN */
                proxy_part_DN = &cert_DN[len2];

                /* First attempt at support for Old and New style GSI
                   proxies: /CN=anything is ok for now */
                if (strncmp((const char*)proxy_part_DN, "/CN=", 4) != 0)
                                         return X509_V_ERR_INVALID_CA;
                                         
                if ((strncmp((const char*)proxy_part_DN, "/CN=limited proxy", 17) == 0) &&
                    (i > 0)) prevIsLimited = 1; /* ready for next cert ... */
              } 
          }
      }

   /* Check cert whose private key is being used by client. If previous in 
      chain is not allowed to be a CA then need to check this final cert for 
      valid proxy-icity too */
   if (!prevIsCA) 
     { 
       if (prevIsLimited) return X509_V_ERR_INVALID_CA;
        /* we do not accept proxies signed by limited proxies */
     
       if ( (cert = sk_X509_value(certstack, 0)) )
         {
           /* Load DN & length of DN and either its issuer or the
              first-bad-issuer-in-chain */
           cert_DN = X509_NAME_oneline(X509_get_subject_name(cert), NULL, 0);
           issuer_DN = X509_NAME_oneline(X509_get_issuer_name(cert),  NULL, 0);
           len = strlen(cert_DN);
           len2 = strlen(issuer_DN);
 
           /* issuer didn't have CA status, check for bad proxy extension */

           if (len2 > len) return X509_V_ERR_INVALID_CA;
             /* User not allowed to sign shortened DN */

           if (strncmp((const char*)cert_DN, issuer_DN, len2) != 0) 
                           return X509_V_ERR_INVALID_CA;
             /* Proxy subject must begin with issuer. */

           proxy_part_DN = &cert_DN[len2];                         
             /* Set pointer to end of DN base in cert_DN */
             
           /* Remander of subject must be either "/CN=proxy" or 
              "/CN=limited proxy" (or /CN=XYZ for New style GSI) */
              
           /* First attempt at support for Old and New style GSI
              proxies: /CN=anything is ok for now. */
           if (strncmp((const char*)proxy_part_DN, "/CN=", 4) != 0)
                                   return X509_V_ERR_INVALID_CA;
         }
     }
 
   return X509_V_OK; /* this is also GRST_RET_OK, of course - by choice */
}

/// Example VerifyCallback routine
int GRSTx509VerifyCallback (int ok, X509_STORE_CTX *ctx)
{
   int errnum   = X509_STORE_CTX_get_error(ctx);
   int errdepth = X509_STORE_CTX_get_error_depth(ctx);
   int first_non_ca;

#ifndef X509_V_ERR_UNHANDLED_CRITICAL_EXTENSION
#define X509_V_ERR_UNHANDLED_CRITICAL_EXTENSION 34
#endif

   if (errnum == X509_V_ERR_UNHANDLED_CRITICAL_EXTENSION)
     {
       if (GRSTx509KnownCriticalExts(X509_STORE_CTX_get_current_cert(ctx))
           == GRST_RET_OK)
         {
           ok = TRUE;
           errnum = X509_V_OK;
           X509_STORE_CTX_set_error(ctx, errnum);
         }                               
     }
   else if ((errdepth == 0)       && 
            (errnum == X509_V_OK) && 
            (GRSTx509CheckChain(&first_non_ca, ctx) != X509_V_OK)) ok = FALSE;
   
   
   return ok;
  
// check this 
   
//   if (ok) return GRST_RET_OK;
//   else    return GRST_RET_FAILED;
}

/// Get the VOMS attributes in the given extension
int GRSTx509ParseVomsExt(int *lastcred, int maxcreds, size_t credlen, 
                         char *creds, time_t time1_time, time_t time2_time,
                         X509_EXTENSION *ex, char *ucuserdn, char *vomsdir)
///
/// Puts any VOMS credentials found into the Compact Creds string array
/// starting at *creds. Always returns GRST_RET_OK - even for invalid
/// credentials, which are just ignored.
{
#define MAXTAG 500
#define GRST_ASN1_COORDS_FQAN    "-1-1-%d-1-7-1-2-1-2-%d"
#define GRST_ASN1_COORDS_USER_DN "-1-1-%d-1-2-1-1-1-1-%%d-1-%%d"
#define GRST_ASN1_COORDS_TIME1   "-1-1-%d-1-6-1"
#define GRST_ASN1_COORDS_TIME2   "-1-1-%d-1-6-2"
   ASN1_OCTET_STRING *asn1data;
   unsigned char     *asn1string;
   char               acuserdn[200],
                      dn_coords[200], fqan_coords[200], time1_coords[200],
                      time2_coords[200];
   long               asn1length;
   int                lasttag=-1, itag, i, acnumber = 1;
   struct GRSTasn1TagList taglist[MAXTAG+1];
   time_t             actime1, actime2, time_now;

   asn1data   = X509_EXTENSION_get_data(ex);
   asn1string = (unsigned char*) (ASN1_STRING_data((asn1data)));
   asn1length = (long)ASN1_STRING_length(asn1data);

   GRSTasn1ParseDump(NULL, asn1string, asn1length, taglist, MAXTAG, &lasttag);

   for (acnumber = 1; ; ++acnumber) /* go through ACs one by one */
      {
        snprintf(dn_coords, sizeof(dn_coords), GRST_ASN1_COORDS_USER_DN, acnumber);
        if (GRSTasn1GetX509Name(acuserdn, sizeof(acuserdn), dn_coords,
                       (char*)asn1string, taglist, lasttag) != GRST_RET_OK) break;

        if (GRSTx509NameCmp(ucuserdn, acuserdn) != 0) continue;

        if (GRSTx509VerifyVomsSig(&time1_time, &time2_time,
                             asn1string, taglist, lasttag, vomsdir, acnumber)
                             != GRST_RET_OK) continue;

        snprintf(time1_coords, sizeof(time1_coords), GRST_ASN1_COORDS_TIME1, acnumber);
        itag = GRSTasn1SearchTaglist(taglist, lasttag, time1_coords);
        actime1 = GRSTasn1TimeToTimeT(&asn1string[taglist[itag].start+
                                             taglist[itag].headerlength],
                                 taglist[itag].length);
        if (actime1 > time1_time) time1_time = actime1;

        snprintf(time2_coords, sizeof(time2_coords), GRST_ASN1_COORDS_TIME2, acnumber);
        itag = GRSTasn1SearchTaglist(taglist, lasttag, time2_coords);
        actime2 = GRSTasn1TimeToTimeT(&asn1string[taglist[itag].start+
                                             taglist[itag].headerlength],
                                             taglist[itag].length);
        if (actime2 < time2_time) time2_time = actime2;

        time(&time_now);
        if ((time1_time > time_now + 300) || (time2_time < time_now))
               continue; /* expiration isnt invalidity ...? */

        for (i=1; ; ++i)
           {
             snprintf(fqan_coords, sizeof(fqan_coords), GRST_ASN1_COORDS_FQAN, acnumber, i);
             itag = GRSTasn1SearchTaglist(taglist, lasttag, fqan_coords);

             if (itag > -1)
               {
                 if (*lastcred < maxcreds - 1)
                   {
                     ++(*lastcred);
                     snprintf(&creds[*lastcred * (credlen + 1)], credlen+1,
                           "VOMS %010lu %010lu 0 %.*s", 
                           time1_time, time2_time, 
                           taglist[itag].length,
                           &asn1string[taglist[itag].start+
                                       taglist[itag].headerlength]);
                   }            
               }
             else break;
           }
      }
      
   return GRST_RET_OK;
}

/// Get the VOMS attributes in the extensions to the given cert stack
int GRSTx509GetVomsCreds(int *lastcred, int maxcreds, size_t credlen, 
                         char *creds, X509 *usercert, STACK_OF(X509) *certstack,
                         char *vomsdir)
///
/// Puts any VOMS credentials found into the Compact Creds string array
/// starting at *creds. Always returns GRST_RET_OK.
{
   int  i, j;
   char s[80];
   unsigned char  *ucuser;
   X509_EXTENSION *ex;
   X509           *cert;
   time_t          time1_time = 0, time2_time = 0, uctime1_time, uctime2_time;

   uctime1_time = 
        GRSTasn1TimeToTimeT(ASN1_STRING_data(X509_get_notBefore(usercert)),0);
   uctime2_time =       
        GRSTasn1TimeToTimeT(ASN1_STRING_data(X509_get_notAfter(usercert)),0);
   ucuser = (unsigned char*)
        X509_NAME_oneline(X509_get_subject_name(usercert), NULL, 0);

   for (j=sk_X509_num(certstack)-1; j >= 0; --j)
    {
      cert = sk_X509_value(certstack, j);

      time1_time =
          GRSTasn1TimeToTimeT(ASN1_STRING_data(X509_get_notBefore(cert)),0);
      uctime1_time = (time1_time > uctime1_time) ? time1_time:uctime1_time;

      time2_time =
          GRSTasn1TimeToTimeT(ASN1_STRING_data(X509_get_notAfter(cert)),0);
      uctime2_time = (time2_time < uctime2_time) ? time2_time:uctime2_time;

      for (i=0; i < X509_get_ext_count(cert); ++i)
         {
           ex = X509_get_ext(cert, i);
           OBJ_obj2txt(s, sizeof(s), X509_EXTENSION_get_object(ex), 1);

           if (strcmp(s, GRST_VOMS_OID) == 0) /* a VOMS extension */
             {
               GRSTx509ParseVomsExt(lastcred, maxcreds, credlen, creds,
                                 uctime1_time, uctime2_time,
                                 ex, (char*)ucuser, vomsdir);
             }
         }
    }

   return GRST_RET_OK;
}

/// Turn a Compact Cred line into a GRSTgaclCred object
GRSTgaclCred *GRSTx509CompactToCred(char *grst_cred)
///
/// Returns pointer to created GRSTgaclCred or NULL or failure.
{
   int       delegation;
   char     *p, *encoded;
   time_t    now, notbefore, notafter;
   GRSTgaclCred *cred = NULL;

   time(&now);

   if (grst_cred == NULL) return NULL; /* just in case */

   if (strncmp((const char*)grst_cred, "X509USER ", 9) == 0)
     {
       if ((sscanf(grst_cred, "X509USER %lu %lu %d", 
                              &notbefore, &notafter, &delegation) == 3)
            && (now >= notbefore)
            && (now <= notafter)
            && (p = index(grst_cred, ' '))
            && (p = index(++p, ' '))
            && (p = index(++p, ' '))
            && (p = index(++p, ' ')))
         {
           encoded = GRSThttpUrlMildencode(&p[1]);
           cred = GRSTgaclCredCreate("dn:", encoded);
           free(encoded);
           GRSTgaclCredSetDelegation(cred, delegation);
         }

       return cred;
     }

   if (strncmp((const char*)grst_cred, "VOMS ", 5) == 0)
     {
       if ((sscanf(grst_cred, "VOMS %lu %lu %d",
                              &notbefore, &notafter, &delegation) == 3)
            && (now >= notbefore)
            && (now <= notafter)
            && (p = index(grst_cred, ' '))
            && (p = index(++p, ' '))
            && (p = index(++p, ' '))
            && (p = index(++p, ' ')))
         {
           /* include /VO/group/subgroup/Role=role/Capability=cap */

           if (p[1] != '/') return NULL; /* must begin with / */

           encoded = GRSThttpUrlMildencode(&p[1]);
           cred = GRSTgaclCredCreate("fqan:", encoded);
           free(encoded);
           GRSTgaclCredSetDelegation(cred, delegation);
         }

       return cred;
     }

   return NULL; /* dont recognise this credential type */
}

/// Get the credentials in an X509 cert/GSI proxy, including any VOMS
int GRSTx509CompactCreds(int *lastcred, int maxcreds, size_t credlen, 
                         char *creds, STACK_OF(X509) *certstack, char *vomsdir, 
                         X509 *peercert)
///
/// Credentials are placed in Compact Creds string array at *creds.
///
/// Function returns GRST_RET_OK on success, or GRST_RET_FAILED if
/// some inconsistency found in certificate.
{   
   int   i, delegation = 0;
   char  credtemp[credlen+1];
   X509 *cert, *usercert = NULL, *gsiproxycert = NULL;

   *lastcred = -1;

   for (i = sk_X509_num(certstack) - 1; i >= 0; --i) 
      {
         cert = sk_X509_value(certstack, i);

         if (usercert != NULL) 
           {           /* found a (GSI proxy) cert after the user cert */
             gsiproxycert = cert;
             ++delegation;
           }
           
         if ((usercert == NULL) && 
             (i < sk_X509_num(certstack) - 1) &&
             (GRSTx509IsCA(cert) != GRST_RET_OK)) usercert = cert;
                                          /* found the 1st non-CA cert */
      }

   if (peercert != NULL)
     {
       if (usercert != NULL) /* found a (GSI proxy) cert after user cert */
         {
           gsiproxycert = peercert;
           ++delegation;
         }

       if ((usercert == NULL) && 
           (GRSTx509IsCA(peercert) != GRST_RET_OK)) usercert = peercert;
                                          /* found the 1st non-CA cert */
     }

   if ((usercert == NULL) /* if no usercert ("EEC"), we're not interested */
       ||
       (snprintf(credtemp, credlen+1, "X509USER %010lu %010lu %d %s",
          GRSTasn1TimeToTimeT(ASN1_STRING_data(X509_get_notBefore(usercert)),0),
          GRSTasn1TimeToTimeT(ASN1_STRING_data(X509_get_notAfter(usercert)),0),
          delegation,
     X509_NAME_oneline(X509_get_subject_name(usercert), NULL, 0)) >= credlen+1)
       || 
       (*lastcred >= maxcreds-1))
     {
       *lastcred = -1;  /* just in case the caller looks at it */
       return GRST_RET_FAILED; /* tell caller that things didn't work out */
     }

   ++(*lastcred);
   strcpy(&creds[*lastcred * (credlen + 1)], credtemp);

   if ((gsiproxycert != NULL) 
       &&
       (snprintf(credtemp, credlen+1, "GSIPROXY %010lu %010lu %d %s",
     GRSTasn1TimeToTimeT(ASN1_STRING_data(X509_get_notBefore(gsiproxycert)),0), 
     GRSTasn1TimeToTimeT(ASN1_STRING_data(X509_get_notAfter(gsiproxycert)),0),
     delegation,
  X509_NAME_oneline(X509_get_subject_name(gsiproxycert), NULL, 0)) < credlen+1)
       &&
       (*lastcred < maxcreds-1))
     {
       ++(*lastcred);
       strcpy(&creds[*lastcred * (credlen + 1)], credtemp);
       GRSTx509GetVomsCreds(lastcred, maxcreds, credlen, creds, 
                            usercert, certstack, vomsdir);
     }
         
   return GRST_RET_OK;
}

/// Find proxy file name of the current user
char *GRSTx509FindProxyFileName(void)
///
/// Return a string with the proxy file name or NULL if not present.
/// This function does not check if the proxy has expired.
{
  char *p;
  
  p = getenv("X509_USER_PROXY");
  
  if (p != NULL) return strdup(p);
  
  p = malloc(sizeof("/tmp/x509up_uXYYYXXXYYY"));
  
  sprintf(p, "/tmp/x509up_u%d", getuid());  

  return p;
}

static void mpcerror(FILE *debugfp, char *msg)
{
  if (debugfp != NULL)
    {
      fputs(msg, debugfp);
      ERR_print_errors_fp(debugfp);
    }
}

/// Make a GSI Proxy chain from a request, certificate and private key
int GRSTx509MakeProxyCert(char **proxychain, FILE *debugfp, 
                          char *reqtxt, char *cert, char *key, int minutes)
///
/// The proxy chain is returned in *proxychain. If debugfp is non-NULL,
/// errors are output to that file pointer. The proxy will expired in
/// the given number of minutes starting from the current time.
{
  char *ptr, *certchain;
  int i, ncerts;
  long serial = 1234, ptrlen;
  EVP_PKEY *pkey, *CApkey;
  const EVP_MD *digest;
  X509 *certs[GRST_MAX_CHAIN_LEN];
  X509_REQ *req;
  X509_NAME *name, *CAsubject, *newsubject;
  X509_NAME_ENTRY *ent;
  FILE *fp;
  BIO *reqmem, *certmem;
  time_t notAfter;

  /* read in the request */
  reqmem = BIO_new(BIO_s_mem());
  BIO_puts(reqmem, reqtxt);
    
  if (!(req = PEM_read_bio_X509_REQ(reqmem, NULL, NULL, NULL)))
    {
      mpcerror(debugfp,
              "GRSTx509MakeProxyCert(): error reading request from BIO memory\n");
      BIO_free(reqmem);
      return GRST_RET_FAILED;
    }
    
  BIO_free(reqmem);

  /* verify signature on the request */
  if (!(pkey = X509_REQ_get_pubkey(req)))
    {
      mpcerror(debugfp,
              "GRSTx509MakeProxyCert(): error getting public key from request\n");
      
      X509_REQ_free(req);
      return GRST_RET_FAILED;
    }

  if (X509_REQ_verify(req, pkey) != 1)
    {
      mpcerror(debugfp,
            "GRSTx509MakeProxyCert(): error verifying signature on certificate\n");

      X509_REQ_free(req);
      return GRST_RET_FAILED;
    }
    
  /* read in the signing certificate */
  if (!(fp = fopen(cert, "r")))
    {
      mpcerror(debugfp,
            "GRSTx509MakeProxyCert(): error opening signing certificate file\n");

      X509_REQ_free(req);
      return GRST_RET_FAILED;
    }    

  for (ncerts = 1; ncerts < GRST_MAX_CHAIN_LEN; ++ncerts)
   if ((certs[ncerts] = PEM_read_X509(fp, NULL, NULL, NULL)) == NULL) break;

  if (ncerts == 1) /* zeroth cert with be new proxy cert */
    {
      mpcerror(debugfp,
            "GRSTx509MakeProxyCert(): error reading signing certificate file\n");

      X509_REQ_free(req);
      return GRST_RET_FAILED;
    }    

  fclose(fp);
  
  CAsubject = X509_get_subject_name(certs[1]);

  /* read in the CA private key */
  if (!(fp = fopen(key, "r")))
    {
      mpcerror(debugfp,
            "GRSTx509MakeProxyCert(): error reading signing private key file\n");

      X509_REQ_free(req);
      return GRST_RET_FAILED;
    }    

  if (!(CApkey = PEM_read_PrivateKey(fp, NULL, NULL, NULL)))
    {
      mpcerror(debugfp,
            "GRSTx509MakeProxyCert(): error reading signing private key in file\n");

      X509_REQ_free(req);
      return GRST_RET_FAILED;
    }    

  fclose(fp);
  
  /* get subject name */
  if (!(name = X509_REQ_get_subject_name(req)))
    {
      mpcerror(debugfp,
            "GRSTx509MakeProxyCert(): error getting subject name from request\n");

      X509_REQ_free(req);
      return GRST_RET_FAILED;
    }    

  /* create new certificate */
  if (!(certs[0] = X509_new()))
    {
      mpcerror(debugfp,
            "GRSTx509MakeProxyCert(): error creating X509 object\n");

      X509_REQ_free(req);
      return GRST_RET_FAILED;
    }    

  /* set version number for the certificate (X509v3) and the serial number   
     need 3 = v4 for GSI proxy?? */
  if (X509_set_version(certs[0], 3L) != 1)
    {
      mpcerror(debugfp,
            "GRSTx509MakeProxyCert(): error setting certificate version\n");

      X509_REQ_free(req);
      return GRST_RET_FAILED;
    }    

  ASN1_INTEGER_set(X509_get_serialNumber(certs[0]), serial++);

  if (!(name = X509_get_subject_name(certs[1])))
    {
      mpcerror(debugfp,
      "GRSTx509MakeProxyCert(): error getting subject name from CA certificate\n");

      X509_REQ_free(req);
      return GRST_RET_FAILED;
    }    

  if (X509_set_issuer_name(certs[0], name) != 1)
    {
      mpcerror(debugfp,
      "GRSTx509MakeProxyCert(): error setting issuer name of certificate\n");

      X509_REQ_free(req);
      return GRST_RET_FAILED;
    }    

  /* set issuer and subject name of the cert from the req and the CA */
  ent = X509_NAME_ENTRY_create_by_NID(NULL, OBJ_txt2nid("commonName"), 
                                      MBSTRING_ASC, (unsigned char*)"proxy", -1);

  newsubject = X509_NAME_dup(CAsubject);

  X509_NAME_add_entry(newsubject, ent, -1, 0);

  if (X509_set_subject_name(certs[0], newsubject) != 1)
    {
      mpcerror(debugfp,
      "GRSTx509MakeProxyCert(): error setting subject name of certificate\n");

      X509_REQ_free(req);
      return GRST_RET_FAILED;
    }    
    
  X509_NAME_free(newsubject);
  X509_NAME_ENTRY_free(ent);

  /* set public key in the certificate */
  if (X509_set_pubkey(certs[0], pkey) != 1)
    {
      mpcerror(debugfp,
      "GRSTx509MakeProxyCert(): error setting public key of the certificate\n");

      X509_REQ_free(req);
      return GRST_RET_FAILED;
    }    

  /* set duration for the certificate */
  if (!(X509_gmtime_adj(X509_get_notBefore(certs[0]), -GRST_BACKDATE_SECONDS)))
    {
      mpcerror(debugfp,
      "GRSTx509MakeProxyCert(): error setting beginning time of the certificate\n");

      X509_REQ_free(req);
      return GRST_RET_FAILED;
    }    

  if (!(X509_gmtime_adj(X509_get_notAfter(certs[0]), 60 * minutes)))
    {
      mpcerror(debugfp,
      "GRSTx509MakeProxyCert(): error setting ending time of the certificate\n");

      X509_REQ_free(req);
      return GRST_RET_FAILED;
    }
    
  /* go through chain making sure this proxy is not longer lived */

  notAfter = 
     GRSTasn1TimeToTimeT(ASN1_STRING_data(X509_get_notAfter(certs[0])), 0);

  for (i=1; i < ncerts; ++i)
       if (notAfter > 
           GRSTasn1TimeToTimeT(ASN1_STRING_data(X509_get_notAfter(certs[i])),
                               0))
         {
           notAfter = 
            GRSTasn1TimeToTimeT(ASN1_STRING_data(X509_get_notAfter(certs[i])),
                                0);
            
           ASN1_UTCTIME_set(X509_get_notAfter(certs[0]), notAfter);
         }

  /* sign the certificate with the signing private key */
  if (EVP_PKEY_type(CApkey->type) == EVP_PKEY_RSA)
    digest = EVP_md5();
  else
    {
      mpcerror(debugfp,
      "GRSTx509MakeProxyCert(): error checking signing private key for a valid digest\n");

      X509_REQ_free(req);
      return GRST_RET_FAILED;
    }    

  if (!(X509_sign(certs[0], CApkey, digest)))
    {
      mpcerror(debugfp,
      "GRSTx509MakeProxyCert(): error signing certificate\n");

      X509_REQ_free(req);
      return GRST_RET_FAILED;
    }    

  /* store the completed certificate chain */

  certchain = strdup("");

  for (i=0; i < ncerts; ++i)
     {
       certmem = BIO_new(BIO_s_mem());

       if (PEM_write_bio_X509(certmem, certs[i]) != 1)
         {
           mpcerror(debugfp,
            "GRSTx509MakeProxyCert(): error writing certificate to memory BIO\n");            

           X509_REQ_free(req);
           return GRST_RET_FAILED;
         }

       ptrlen = BIO_get_mem_data(certmem, &ptr);
  
       certchain = realloc(certchain, strlen(certchain) + ptrlen + 1);
       
       strncat(certchain, ptr, ptrlen);
    
       BIO_free(certmem);
       X509_free(certs[i]);
     }
  
  EVP_PKEY_free(pkey);
  EVP_PKEY_free(CApkey);
  X509_REQ_free(req);
      
  *proxychain = certchain;  
  return GRST_RET_OK;
}

/// Find a proxy file in the proxy cache
char *GRSTx509CachedProxyFind(char *proxydir, char *delegation_id, 
                              char *user_dn)
///
/// Returns the full path and file name of proxy file associated
/// with given delegation ID and user DN.
///
/// Return a pointer to a malloc'd string with the full path of the 
/// proxy file corresponding to the given delegation_id, or NULL
/// if not found.
{
  char *user_dn_enc;
  char* proxyfile = (char*)malloc(16384);
  struct stat statbuf;

  user_dn_enc = GRSThttpUrlEncode(user_dn);

  sprintf(proxyfile, "%s/%s/%s/userproxy.pem",
           proxydir, user_dn_enc, delegation_id);
           
  free(user_dn_enc);

  if ((stat(proxyfile, &statbuf) != 0) || !S_ISREG(statbuf.st_mode))
    {
      free(proxyfile);
      return NULL;
    }
    
  return proxyfile;
}

/// Find a temporary proxy private key file in the proxy cache
char *GRSTx509CachedProxyKeyFind(char *proxydir, char *delegation_id, 
                                 char *user_dn)
///
/// Returns the full path and file name of the private key file associated
/// with given delegation ID and user DN.
///
/// Return a pointer to a malloc'd string with the full path of the 
/// private proxy key corresponding to the given delegation_id, or NULL
/// if not found.
{
  char *user_dn_enc;
  char* prvkeyfile = (char*) malloc(16384);
  struct stat statbuf;

  user_dn_enc = GRSThttpUrlEncode(user_dn);

  sprintf(prvkeyfile, "%s/cache/%s/%s/userkey.pem",
           proxydir, user_dn_enc, delegation_id);
           
  free(user_dn_enc);

  if ((stat(prvkeyfile, &statbuf) != 0) || !S_ISREG(statbuf.st_mode))
    {
      free(prvkeyfile);
      return NULL;
    }
    
  return prvkeyfile;
}

static void mkdir_printf(mode_t mode, char *fmt, ...)
{
  int   ret;
  char path[16384];
  va_list ap;
  
  va_start(ap, fmt);
  vsprintf(path, fmt, ap);
  va_end(ap);

  ret = mkdir(path, mode);
}

/// Create a X.509 request for a GSI proxy and its private key
int GRSTx509CreateProxyRequest(char **reqtxt, char **keytxt, char *ocspurl)
///
/// Returns GRST_RET_OK on success, non-zero otherwise. Request string
/// and private key are PEM encoded strings
{
  char            *ptr;
  size_t           ptrlen;
  RSA             *keypair;
  X509_NAME       *subject;
  X509_NAME_ENTRY *ent;
  EVP_PKEY        *pkey;
  X509_REQ        *certreq;
  BIO             *reqmem, *keymem;
  const EVP_MD    *digest;

  /* create key pair and put it in a PEM string */

  if ((keypair = RSA_generate_key(GRST_KEYSIZE, 65537, NULL, NULL)) == NULL)
                                                               return 1;

  keymem = BIO_new(BIO_s_mem());
  if (!PEM_write_bio_RSAPrivateKey(keymem, keypair, NULL, NULL, 0, NULL, NULL))
    {
      BIO_free(keymem);
      return 3;
    }

  ptrlen = BIO_get_mem_data(keymem, &ptr);
  
  *keytxt = malloc(ptrlen + 1);
  memcpy(*keytxt, ptr, ptrlen);
  (*keytxt)[ptrlen] = '\0';

  BIO_free(keymem);
  
  /* now create the certificate request */

  certreq = X509_REQ_new();

  OpenSSL_add_all_algorithms();

  pkey = EVP_PKEY_new();
  EVP_PKEY_assign_RSA(pkey, keypair);

  X509_REQ_set_pubkey(certreq, pkey);
  
  subject = X509_NAME_new();
  ent = X509_NAME_ENTRY_create_by_NID(NULL, OBJ_txt2nid("organizationName"), 
                                      MBSTRING_ASC, (unsigned char*)"Dummy", -1);
  X509_NAME_add_entry (subject, ent, -1, 0);
  X509_REQ_set_subject_name (certreq, subject);
  
  digest = EVP_md5();
  X509_REQ_sign(certreq, pkey, digest);

  reqmem = BIO_new(BIO_s_mem());
  PEM_write_bio_X509_REQ(reqmem, certreq);
  ptrlen = BIO_get_mem_data(reqmem, &ptr);
  
  *reqtxt = malloc(ptrlen + 1);
  memcpy(*reqtxt, ptr, ptrlen);
  (*reqtxt)[ptrlen] = '\0';

  BIO_free(reqmem);

  X509_REQ_free(certreq);
  
  return 0;
}

/// Make and store a X.509 request for a GSI proxy
int GRSTx509MakeProxyRequest(char **reqtxt, char *proxydir, 
                             char *delegation_id, char *user_dn)
///
/// Returns GRST_RET_OK on success, non-zero otherwise. Request string
/// is PEM encoded, and the key is stored in the temporary cache under
/// proxydir
{
  char            prvkeyfile[16384], *ptr, *user_dn_enc;
  size_t           ptrlen;
  FILE            *fp;
  RSA             *keypair;
  X509_NAME       *subject;
  X509_NAME_ENTRY *ent;
  EVP_PKEY        *pkey;
  X509_REQ        *certreq;
  BIO             *reqmem;
  const EVP_MD    *digest;

  if (strcmp(user_dn, "cache") == 0) return GRST_RET_FAILED;
    
  user_dn_enc = GRSThttpUrlEncode(user_dn);

  /* create directories if necessary */

  mkdir_printf(S_IRUSR | S_IWUSR | S_IXUSR, 
               "%s/cache",       proxydir);
  mkdir_printf(S_IRUSR | S_IWUSR | S_IXUSR, 
               "%s/cache/%s",    proxydir, user_dn_enc);
  mkdir_printf(S_IRUSR | S_IWUSR | S_IXUSR, 
               "%s/cache/%s/%s", proxydir, user_dn_enc, delegation_id);

  /* make the new proxy private key */

  sprintf(prvkeyfile, "%s/cache/%s/%s/userkey.pem",
           proxydir, user_dn_enc, delegation_id);

  if (prvkeyfile == NULL)  
    {
      free(user_dn_enc);
      return GRST_RET_FAILED;
    }
        
  if ((keypair = RSA_generate_key(GRST_KEYSIZE, 65537, NULL, NULL)) == NULL)
                                                               return 1;
          
  if ((fp = fopen(prvkeyfile, "w")) == NULL) return 2;
  
  chmod(prvkeyfile, S_IRUSR | S_IWUSR);
  free(user_dn_enc);

  if (!PEM_write_RSAPrivateKey(fp, keypair, NULL, NULL, 0, NULL, NULL))
                               return 3;
  
  if (fclose(fp) != 0) return 4;
  
  /* now create the certificate request */

  certreq = X509_REQ_new();
  if (certreq == NULL) return 5;

  OpenSSL_add_all_algorithms();

  pkey = EVP_PKEY_new();
  EVP_PKEY_assign_RSA(pkey, keypair);

  X509_REQ_set_pubkey(certreq, pkey);
  
  subject = X509_NAME_new();
  ent = X509_NAME_ENTRY_create_by_NID(NULL, OBJ_txt2nid("organizationName"), 
                                      MBSTRING_ASC, (unsigned char*)"Dummy", -1);
  X509_NAME_add_entry (subject, ent, -1, 0);
  X509_REQ_set_subject_name (certreq, subject);
  
  digest = EVP_md5();
  X509_REQ_sign(certreq, pkey, digest);

  reqmem = BIO_new(BIO_s_mem());
  PEM_write_bio_X509_REQ(reqmem, certreq);
  ptrlen = BIO_get_mem_data(reqmem, &ptr);
  
  *reqtxt = malloc(ptrlen + 1);
  memcpy(*reqtxt, ptr, ptrlen);
  (*reqtxt)[ptrlen] = '\0';

  BIO_free(reqmem);

  X509_REQ_free(certreq);
  
  return 0;
}

/// Destroy stored GSI proxy files
int GRSTx509ProxyDestroy(char *proxydir, char *delegation_id, char *user_dn)
///
/// Returns GRST_RET_OK on success, non-zero otherwise.
/// (Including GRST_RET_NO_SUCH_FILE if the private key or cert chain
///  were not found.)
{
  int              ret = GRST_RET_OK;
  char            filename[16384], *user_dn_enc;

  if (strcmp(user_dn, "cache") == 0) return GRST_RET_FAILED;
    
  user_dn_enc = GRSThttpUrlEncode(user_dn);

  /* proxy file */
  
  sprintf(filename, "%s/%s/%s/userproxy.pem",
           proxydir, user_dn_enc, delegation_id);

  if (unlink(filename) != 0) ret = GRST_RET_NO_SUCH_FILE;  

  /* voms file */
  
  sprintf(filename, "%s/%s/%s/voms.attributes",
           proxydir, user_dn_enc, delegation_id);

  unlink(filename);
  
  return ret;
}

/// Get start and finish validity times of stored GSI proxy file
int GRSTx509ProxyGetTimes(char *proxydir, char *delegation_id, char *user_dn, 
                          time_t *start, time_t *finish)
///
/// Returns GRST_RET_OK on success, non-zero otherwise.
/// (Including GRST_RET_NO_SUCH_FILE if the cert chain was not found.)
{
  char  filename[16384], *user_dn_enc;
  FILE  *fp;
  X509  *cert;

  if (strcmp(user_dn, "cache") == 0) return GRST_RET_FAILED;
    
  user_dn_enc = GRSThttpUrlEncode(user_dn);
  
  sprintf(filename, "%s/%s/%s/userproxy.pem",
           proxydir, user_dn_enc, delegation_id);
           
  free(user_dn_enc);

  if (filename == NULL) return GRST_RET_FAILED;

  fp = fopen(filename, "r");
  
  if (fp == NULL) return GRST_RET_NO_SUCH_FILE;

  cert = PEM_read_X509(fp, NULL, NULL, NULL); /* first cert is X.509 PC */

  fclose(fp);
  
  *start  = GRSTasn1TimeToTimeT(ASN1_STRING_data(X509_get_notBefore(cert)),0);
  *finish = GRSTasn1TimeToTimeT(ASN1_STRING_data(X509_get_notAfter(cert)),0);

  X509_free(cert);
  
  return GRST_RET_OK;
}

/// Create a stack of X509 certificate from a PEM-encoded string
int GRSTx509StringToChain(STACK_OF(X509) **certstack, char *certstring)
///
/// Creates a dynamically allocated stack of X509 certificate objects
/// by walking through the PEM-encoded X509 certificates.
///
/// Returns GRST_RET_OK on success, non-zero otherwise.
{   
  STACK_OF(X509_INFO) *sk=NULL;
  BIO *certbio;
  X509_INFO *xi;

  *certstack = sk_X509_new_null();
  if (*certstack == NULL) return GRST_RET_FAILED;

  certbio = BIO_new_mem_buf(certstring, -1);
  
  if (!(sk=PEM_X509_INFO_read_bio(certbio, NULL, NULL, NULL)))
    {
      BIO_free(certbio);
      sk_X509_INFO_free(sk);
      sk_X509_free(*certstack);
      return GRST_RET_FAILED;
    }
      
  while (sk_X509_INFO_num(sk))
       {
         xi=sk_X509_INFO_shift(sk);
         if (xi->x509 != NULL)
           {
             sk_X509_push(*certstack, xi->x509);
             xi->x509=NULL;
           }
         X509_INFO_free(xi);
       }
       
   if (!sk_X509_num(*certstack))
     {
       BIO_free(certbio);
       sk_X509_INFO_free(sk);
       sk_X509_free(*certstack);
       return GRST_RET_FAILED;
     }

   BIO_free(certbio);
   sk_X509_INFO_free(sk);
   
   return GRST_RET_OK;
}

/// Returns a Delegation ID based on hash of GRST_CRED_0, ...
char *GRSTx509MakeDelegationID(void)
///
/// Returns a malloc'd string with Delegation ID made by SHA1-hashing the
/// values of the compact credentials exported by mod_gridsite
{ 
  unsigned char hash_delegation_id[EVP_MAX_MD_SIZE];        
  int  i;
  unsigned int delegation_id_len;
  char cred_name[14], *cred_value, *delegation_id;
  const EVP_MD *m;
  EVP_MD_CTX ctx;

  OpenSSL_add_all_digests();

  m = EVP_sha1();
  if (m == NULL) return NULL;

  EVP_DigestInit(&ctx, m);

  for (i=0; i <= 999; ++i)
     {
       snprintf(cred_name, sizeof(cred_name), "GRST_CRED_%d", i);       
       if ((cred_value = getenv(cred_name)) == NULL) break;
       
       EVP_DigestUpdate(&ctx, cred_value, strlen(cred_value));
     }
     
  EVP_DigestFinal(&ctx, hash_delegation_id, &delegation_id_len);

  delegation_id = malloc(17);

  for (i=0; i <=7; ++i)
   sprintf(&delegation_id[i*2], "%02x", hash_delegation_id[i]);

  delegation_id[16] = '\0';

  return delegation_id;
}

#if 0
/// Return the short file name for the given delegation_id and user_dn
char *GRSTx509MakeProxyFileName(char *delegation_id,
                                STACK_OF(X509) *certstack)
///
/// Returns a malloc'd string with the short file name (no paths) that
/// derived from the hashed delegation_id and user_dn
///
/// File name is SHA1_HASH(DelegationID)+"-"+SHA1_HASH(DN) where DN
/// is DER encoded version of user_dn with any trailing CN=proxy removed
/// Hashes are the most significant 8 bytes, in lowercase hexadecimal.
{ 
  int        i, depth, prevIsCA = 1, IsCA, hash_name_len, delegation_id_len,
                 der_name_len;
  unsigned char *der_name, *buf, hash_name[EVP_MAX_MD_SIZE],
                 hash_delegation_id[EVP_MAX_MD_SIZE],
                 filename[34];
  X509_NAME *subject_name;
  X509      *cert;
  const EVP_MD *m;
  EVP_MD_CTX ctx;

  depth = sk_X509_num(certstack);  
  
  for (i=depth-1; i >= 0; --i)
        /* loop through the proxy chain starting at CA end */
     {
       if (cert = sk_X509_value(certstack, i))
         {
           IsCA = (GRSTx509IsCA(cert) == GRST_RET_OK);

           if (prevIsCA && !IsCA) /* the full certificate of the user */
             {
               break;
             }
         }
     }

  if (i < 0) return NULL; /* not found: something wrong with the chain */

  if ((subject_name = X509_get_subject_name(cert)) == NULL) return NULL;
  
  der_name_len = i2d_X509_NAME(X509_get_subject_name(cert), NULL);
  if (der_name_len == 0) return NULL;
  
  buf = OPENSSL_malloc(der_name_len);
  der_name = buf;


  if (!i2d_X509_NAME(X509_get_subject_name(cert), &der_name))
    {
      OPENSSL_free(der_name);
      return NULL;
    }

  OpenSSL_add_all_digests();

  m = EVP_sha1();
  if (m == NULL)
    {
      OPENSSL_free(der_name);
      return NULL;
    }


  EVP_DigestInit(&ctx, m);
  EVP_DigestUpdate(&ctx, delegation_id, strlen(delegation_id));
  EVP_DigestFinal(&ctx, hash_delegation_id, &delegation_id_len);

  /* lots of nasty hard coded numbers: 
     "8bytes/16chars delegation ID" + "-" + "8bytes/16chars DN" */

  for (i=0; i <=7; ++i)
   sprintf(&filename[i*2], "%02x", hash_delegation_id[i]);

  filename[16] = '-';

  EVP_DigestInit(&ctx, m);
  EVP_DigestUpdate(&ctx, buf, der_name_len);
  EVP_DigestFinal(&ctx, hash_name, &hash_name_len);

  for (i=0; i <=7; ++i)
   sprintf(&filename[17 + i*2], "%02x", hash_name[i]);

  return strdup(filename);
}
#endif

/// Store a GSI proxy chain in the proxy cache, along with the private key
int GRSTx509CacheProxy(char *proxydir, char *delegation_id, 
                                       char *user_dn, char *proxychain)
///
/// Returns GRST_RET_OK on success, non-zero otherwise. The existing
/// private key with the same delegation ID and user DN is moved out of
/// the temporary cache.
{
  int   c, i;
  char *user_dn_enc, *ptr, *prvkeyfile, proxyfile[16384];
  STACK_OF(X509) *certstack;
  BIO  *certmem;
  X509 *cert;
  long  ptrlen;        
  FILE *ifp, *ofp;

  if (strcmp(user_dn, "cache") == 0) return GRST_RET_FAILED;
    
  /* find the existing private key file */

  prvkeyfile = GRSTx509CachedProxyKeyFind(proxydir, delegation_id, user_dn);

  if (prvkeyfile == NULL)
    {
      return GRST_RET_FAILED;
    }

  /* open it ready for later */

  if ((ifp = fopen(prvkeyfile, "r")) == NULL)
    {
      free(prvkeyfile);
      return GRST_RET_FAILED;
    }

  /* get the X509 stack */

  if (GRSTx509StringToChain(&certstack, proxychain) != GRST_RET_OK)
    {
      fclose(ifp);
      free(prvkeyfile);
      return GRST_RET_FAILED;
    }

  /* create directories if necessary, and set proxy filename */

  user_dn_enc = GRSThttpUrlEncode(user_dn);

  mkdir_printf(S_IRUSR | S_IWUSR | S_IXUSR, 
               "%s/%s",    proxydir, user_dn_enc);
  mkdir_printf(S_IRUSR | S_IWUSR | S_IXUSR, 
               "%s/%s/%s", proxydir, user_dn_enc, delegation_id);

  sprintf(proxyfile, "%s/%s/%s/userproxy.pem",
           proxydir, user_dn_enc, delegation_id);
           
  free(user_dn_enc);

  /* set up to write proxy file */

  ofp = fopen(proxyfile, "w");
  chmod(proxyfile, S_IRUSR | S_IWUSR);

  if (ofp == NULL)
    {
      fclose(ifp);
      free(prvkeyfile);
      return GRST_RET_FAILED;
    }

  /* write out the most recent proxy by itself */

  if ( (cert = sk_X509_value(certstack, 0)) )
    {
      certmem = BIO_new(BIO_s_mem());
      if (PEM_write_bio_X509(certmem, cert) == 1)
        {
          ptrlen = BIO_get_mem_data(certmem, &ptr);
          fwrite(ptr, 1, ptrlen, ofp);
        }

      BIO_free(certmem);
    }

  /* insert proxy private key, read from private key file */

  while ((c = fgetc(ifp)) != EOF) fputc(c, ofp);
  unlink(prvkeyfile);
  free(prvkeyfile);

  for (i=1; i <= sk_X509_num(certstack) - 1; ++i)
        /* loop through the proxy chain starting at 2nd most recent proxy */
     {
       if ( (cert = sk_X509_value(certstack, i)))
         {
           certmem = BIO_new(BIO_s_mem());
           if (PEM_write_bio_X509(certmem, cert) == 1)
             {
               ptrlen = BIO_get_mem_data(certmem, &ptr);
               fwrite(ptr, 1, ptrlen, ofp);
             }

           BIO_free(certmem);
         }
     }

  sk_X509_free(certstack);

  if (fclose(ifp) != 0) return GRST_RET_FAILED;
  if (fclose(ofp) != 0) return GRST_RET_FAILED;

  return GRST_RET_OK;
}
