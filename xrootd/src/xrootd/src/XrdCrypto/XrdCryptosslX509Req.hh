// $Id$
#ifndef __CRYPTO_SSLX509REQ_H__
#define __CRYPTO_SSLX509REQ_H__
/******************************************************************************/
/*                                                                            */
/*               X r d C r y p t o s s l X 5 0 9 R e q . h h                  */
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

#include <XrdCrypto/XrdCryptoX509Req.hh>

#include <openssl/x509v3.h>
#include <openssl/bio.h>

// ---------------------------------------------------------------------------//
//
// OpenSSL X509 request implementation
//
// ---------------------------------------------------------------------------//
class XrdCryptosslX509Req : public XrdCryptoX509Req
{

public:
   XrdCryptosslX509Req(XrdSutBucket *bck);
   XrdCryptosslX509Req(X509_REQ *creq);
   virtual ~XrdCryptosslX509Req();

   // Access underlying data (in opaque form: used in chains)
   XrdCryptoX509Reqdata Opaque() { return (XrdCryptoX509Reqdata)creq; }

   // Access certificate key
   XrdCryptoRSA *PKI() { return pki; }

   // Export in form of bucket (for transfers)
   XrdSutBucket *Export();

   // Relevant Names
   const char *Subject();  // get subject name

   // Relevant hashes
   const char *SubjectHash();  // get hash of subject name

   // Retrieve a given extension if there (in opaque form)
   XrdCryptoX509Reqdata GetExtension(const char *oid);

   // Verify signature
   bool        Verify();

private:
   X509_REQ    *creq;       // The certificate request object
   XrdOucString subject;    // subject;
   XrdOucString subjecthash; // hash of subject;
   XrdSutBucket *bucket;    // Bucket for export operations
   XrdCryptoRSA *pki;       // PKI of the certificate
};

#endif
