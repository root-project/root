// $Id$
#ifndef __CRYPTO_SSLX509_H__
#define __CRYPTO_SSLX509_H__
/******************************************************************************/
/*                                                                            */
/*                   X r d C r y p t o s s l X 5 0 9 . h h                    */
/*                                                                            */
/*                                                                            */
/* (c) 2005 G. Ganis , CERN                                                   */
/*                                                                            */
/******************************************************************************/

/* ************************************************************************** */
/*                                                                            */
/* OpenSSL implementation of XrdCryptoX509                                    */
/*                                                                            */
/* ************************************************************************** */

#include <XrdCrypto/XrdCryptoX509.hh>

#include <openssl/x509v3.h>
#include <openssl/bio.h>
#include <openssl/evp.h>

// ---------------------------------------------------------------------------//
//
// OpenSSL X509 implementation
//
// ---------------------------------------------------------------------------//
class XrdCryptosslX509 : public XrdCryptoX509
{

public:
   XrdCryptosslX509(const char *cf, const char *kf = 0);
   XrdCryptosslX509(XrdSutBucket *bck);
   XrdCryptosslX509(X509 *cert);
   virtual ~XrdCryptosslX509();

   // Access underlying data (in opaque form: used in chains)
   XrdCryptoX509data Opaque() { return (XrdCryptoX509data)cert; }

   // Access certificate key
   XrdCryptoRSA *PKI() { return pki; }
   void SetPKI(XrdCryptoX509data pki);

   // Export in form of bucket (for transfers)
   XrdSutBucket *Export();

   // Parent file
   const char *ParentFile() { return (const char *)(srcfile.c_str()); }

   // Key strength
   int BitStrength() { return ((cert) ? EVP_PKEY_bits(X509_get_pubkey(cert)) : -1);}

   // Serial number
   kXR_int64 SerialNumber();
   XrdOucString SerialNumberString();

   // Validity
   int NotBefore();  // get begin-validity time in secs since Epoch
   int NotAfter();   // get end-validity time in secs since Epoch

   // Relevant Names
   const char *Subject();  // get subject name
   const char *Issuer();   // get issuer name

   // Relevant hashes
   const char *SubjectHash();  // get hash of subject name
   const char *IssuerHash();   // get hash of issuer name 

   // Retrieve a given extension if there (in opaque form)
   XrdCryptoX509data GetExtension(const char *oid);

   // Verify signature
   bool        Verify(XrdCryptoX509 *ref);

private:
   X509        *cert;       // The certificate object
   int          notbefore;  // begin-validity time in secs since Epoch
   int          notafter;   // end-validity time in secs since Epoch
   XrdOucString subject;    // subject;
   XrdOucString issuer;     // issuer name;
   XrdOucString subjecthash; // hash of subject;
   XrdOucString issuerhash;  // hash of issuer name;
   XrdOucString srcfile;    // source file name, if any;
   XrdSutBucket *bucket;    // Bucket for export operations
   XrdCryptoRSA *pki;       // PKI of the certificate

   bool         IsCA();     // Find out if we are a CA

};

#endif
