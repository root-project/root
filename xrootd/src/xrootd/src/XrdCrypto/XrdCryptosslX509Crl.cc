// $Id$
/******************************************************************************/
/*                                                                            */
/*                X r d C r y p t o s s l X 5 0 9 C r l. c c                  */
/*                                                                            */
/* (c) 2005 G. Ganis , CERN                                                   */
/*                                                                            */
/******************************************************************************/


/* ************************************************************************** */
/*                                                                            */
/* OpenSSL implementation of XrdCryptoX509Crl                                 */
/*                                                                            */
/* ************************************************************************** */
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <errno.h>
#include <time.h>

#include <XrdCrypto/XrdCryptosslRSA.hh>
#include <XrdCrypto/XrdCryptosslX509Crl.hh>
#include <XrdCrypto/XrdCryptosslAux.hh>
#include <XrdCrypto/XrdCryptosslTrace.hh>

#include <openssl/bn.h>
#include <openssl/pem.h>

//_____________________________________________________________________________
XrdCryptosslX509Crl::XrdCryptosslX509Crl(const char *cf)
                 : XrdCryptoX509Crl()
{
   // Constructor certificate from file 'cf'. If 'kf' is defined,
   // complete the key of the certificate with the private key in kf.
   EPNAME("X509Crl::XrdCryptosslX509Crl_file");

   // Init private members
   crl = 0;         // The crl object
   lastupdate = -1;  // begin-validity time in secs since Epoch
   nextupdate = -1;   // end-validity time in secs since Epoch
   issuer = "";     // issuer;
   issuerhash = "";  // hash of issuer;
   srcfile = "";    // source file;
   nrevoked = 0;    // number of revoked certificates

   // Make sure file name is defined;
   if (!cf) {
      DEBUG("file name undefined");
      return;
   }
   // Make sure file exists;
   struct stat st;
   if (stat(cf, &st) != 0) {
      if (errno == ENOENT) {
         DEBUG("file "<<cf<<" does not exist - do nothing");
      } else {
         DEBUG("cannot stat file "<<cf<<" (errno: "<<errno<<")");
      }
      return;
   }
   //
   // Open file in read mode
   FILE *fc = fopen(cf, "r");
   if (!fc) {
      DEBUG("cannot open file "<<cf<<" (errno: "<<errno<<")");
      return;
   }
   //
   // Read the content:
   if (!PEM_read_X509_CRL(fc, &crl, 0, 0)) {
      DEBUG("Unable to load CRL from file");
      return;
   } else {
      DEBUG("CRL successfully loaded");
   }
   //
   // Close the file
   fclose(fc);
   //
   // Save source file name
   srcfile = cf;
   //
   // Init some of the private members (the others upon need)
   Issuer();
   //
   // Load into cache
   LoadCache();
}

//_____________________________________________________________________________
XrdCryptosslX509Crl::~XrdCryptosslX509Crl()
{
   // Destructor

   // Cleanup CRL
   if (crl)
      X509_CRL_free(crl);
}

//_____________________________________________________________________________
int XrdCryptosslX509Crl::LoadCache()
{
   // Load relevant info into the cache
   // Return 0 if ok, -1 in case of error
   EPNAME("LoadCache");

   // The CRL must exists
   if (!crl) {
      DEBUG("CRL undefined");
      return -1;
   }

   // Parse CRL
   STACK_OF(X509_REVOKED *) *rsk = X509_CRL_get_REVOKED(crl);
   if (!rsk) {
      DEBUG("could not get stack of revoked instances");
      return -1;
   }

   // Number of revocations
   nrevoked = sk_num(rsk);
   DEBUG(nrevoked << "certificates have been revoked");
   if (nrevoked <= 0) {
      DEBUG("no valid certificate has been revoked - nothing to do");
      return 0;
   }

   // Init cache
   if (cache.Init(nrevoked) != 0) {
      DEBUG("problems init cache for CRL info");
      return -1;
   }

   // Get serial numbers of revoked certificates
   char *tagser = 0;
   int i = 0;
   for (; i < nrevoked; i++ ){
      X509_REVOKED *rev = (X509_REVOKED *)sk_value(rsk,i);
      if (rev) {
         BIGNUM *bn = BN_new();
         ASN1_INTEGER_to_BN(rev->serialNumber, bn);
         tagser = BN_bn2hex(bn);
         BN_free(bn);
         TRACE(Dump, "certificate with serial number: "<<tagser<<
                     "  has been revoked");
         // Add to the cache
         XrdSutPFEntry *cent = cache.Add((const char *)tagser);
         if (!cent) {
            DEBUG("problems updating the cache");
            return -1;
         }
         // Add revocation date
         cent->mtime = XrdCryptosslASN1toUTC(rev->revocationDate);
         // Release the string for the serial number
         OPENSSL_free(tagser);
      }
   }

   // rehash the cache
   cache.Rehash(1);

   return 0;
}

//_____________________________________________________________________________
int XrdCryptosslX509Crl::LastUpdate()
{
   // Time of last update

   // If we do not have it already, try extraction
   if (lastupdate < 0) {
      // Make sure we have a CRL
      if (crl)
         // Extract UTC time in secs from Epoch
         lastupdate = XrdCryptosslASN1toUTC(X509_CRL_get_lastUpdate(crl));
   }
   // return what we have
   return lastupdate;
}

//_____________________________________________________________________________
int XrdCryptosslX509Crl::NextUpdate()
{
   // Time of next update

   // If we do not have it already, try extraction
   if (nextupdate < 0) {
      // Make sure we have a CRL
      if (crl)
         // Extract UTC time in secs from Epoch
         nextupdate = XrdCryptosslASN1toUTC(X509_CRL_get_nextUpdate(crl));
   }
   // return what we have
   return nextupdate;
}

//_____________________________________________________________________________
const char *XrdCryptosslX509Crl::Issuer()
{
   // Return issuer name
   EPNAME("X509Crl::Issuer");

   // If we do not have it already, try extraction
   if (issuer.length() <= 0) {

      // Make sure we have a CRL
      if (!crl) {
         DEBUG("WARNING: no CRL available - cannot extract issuer name");
         return (const char *)0;
      }

      // Extract issuer name
      issuer = X509_NAME_oneline(X509_CRL_get_issuer(crl), 0, 0);
   }

   // return what we have
   return (issuer.length() > 0) ? issuer.c_str() : (const char *)0;
}

//_____________________________________________________________________________
const char *XrdCryptosslX509Crl::IssuerHash()
{
   // Return issuer name
   EPNAME("X509Crl::IssuerHash");

   // If we do not have it already, try extraction
   if (issuerhash.length() <= 0) {

      // Make sure we have a CRL
      if (crl) {
         char chash[15];
         sprintf(chash,"%08lx.0",X509_NAME_hash(crl->crl->issuer));
         issuerhash = chash;
      } else {
         DEBUG("WARNING: no CRL available - cannot extract issuer hash");
      }
   }

   // return what we have
   return (issuerhash.length() > 0) ? issuerhash.c_str() : (const char *)0;
}

//_____________________________________________________________________________
bool XrdCryptosslX509Crl::Verify(XrdCryptoX509 *ref)
{
   // Verify certificate signature with pub key of ref cert

   // We must have been initialized
   if (!crl)
      return 0;

   // We must have something to check with
   X509 *r = ref ? (X509 *)(ref->Opaque()) : 0;
   EVP_PKEY *rk = r ? X509_get_pubkey(r) : 0;
   if (!rk)
      return 0;

   // Ok: we can verify
   return (X509_CRL_verify(crl, rk) > 0);
}

//_____________________________________________________________________________
bool XrdCryptosslX509Crl::IsRevoked(int serialnumber, int when)
{
   // Check if certificate with serialnumber is in the
   // list of revocated certificates
   EPNAME("IsRevoked");

   // Reference time
   int now = (when > 0) ? when : time(0);

   // Warn if CRL should be updated
   if (now > NextUpdate()) {
      DEBUG("WARNING: CRL is expired: you should download the updated one");
   }

   // We must have something to check against
   if (nrevoked <= 0) {
      DEBUG("No certificate in the list");
      return 0;
   }

   // Ok, build the tag
   char tagser[20] = {0};
   sprintf(tagser,"%x",serialnumber);

   // Look into the cache
   XrdSutPFEntry *cent = cache.Get((const char *)tagser);
   if (cent) {
      // Check the revocation time
      if (now > cent->mtime) {
         DEBUG("certificate "<<tagser<<" has been revoked");
         return 1;
      }
   }

   // Certificate not revoked
   return 0;
}

//_____________________________________________________________________________
bool XrdCryptosslX509Crl::IsRevoked(const char *sernum, int when)
{
   // Check if certificate with 'sernum' is in the
   // list of revocated certificates
   EPNAME("IsRevoked");

   // Reference time
   int now = (when > 0) ? when : time(0);

   // Warn if CRL should be updated
   if (now > NextUpdate()) {
      DEBUG("WARNING: CRL is expired: you should download the updated one");
   }

   // We must have something to check against
   if (nrevoked <= 0) {
      DEBUG("No certificate in the list");
      return 0;
   }

   // Look into the cache
   XrdSutPFEntry *cent = cache.Get((const char *)sernum);
   if (cent) {
      // Check the revocation time
      if (now > cent->mtime) {
         DEBUG("certificate "<<sernum<<" has been revoked");
         return 1;
      }
   }

   // Certificate not revoked
   return 0;
}

//_____________________________________________________________________________
void XrdCryptosslX509Crl::Dump()
{
   // Dump content
   EPNAME("X509Crl::Dump");

   // Time strings
   struct tm tst;
   char stbeg[256] = {0};
   time_t tbeg = LastUpdate();
   localtime_r(&tbeg,&tst);
   asctime_r(&tst,stbeg);
   stbeg[strlen(stbeg)-1] = 0;
   char stend[256] = {0};
   time_t tend = NextUpdate();
   localtime_r(&tend,&tst);
   asctime_r(&tst,stend);
   stend[strlen(stend)-1] = 0;

   PRINT("+++++++++++++++ X509 CRL dump +++++++++++++++++++++++");
   PRINT("+");
   PRINT("+ File:    "<<ParentFile());
   PRINT("+");
   PRINT("+ Issuer:  "<<Issuer());
   PRINT("+ Issuer hash:  "<<IssuerHash());
   PRINT("+");
   if (IsExpired()) {
      PRINT("+ Validity: (expired!)");
   } else {
      PRINT("+ Validity:");
   }
   PRINT("+ LastUpdate:  "<<tbeg<<" UTC - "<<stbeg);
   PRINT("+ NextUpdate:  "<<tend<<" UTC - "<<stend);
   PRINT("+");
   PRINT("+ Number of revoked certificates: "<<nrevoked);
   PRINT("+");
   PRINT("+++++++++++++++++++++++++++++++++++++++++++++++++");
}
