// $Id$
#ifndef __CRYPTO_SSLX509CRL_H__
#define __CRYPTO_SSLX509CRL_H__
/******************************************************************************/
/*                                                                            */
/*                X r d C r y p t o s s l X 5 0 9 C r l . h h                 */
/*                                                                            */
/*                                                                            */
/* (c) 2005 G. Ganis , CERN                                                   */
/*                                                                            */
/******************************************************************************/
#include <openssl/x509v3.h>

/* ************************************************************************** */
/*                                                                            */
/* OpenSSL X509 CRL implementation        .                                   */
/*                                                                            */
/* ************************************************************************** */

#include <XrdSut/XrdSutCache.hh>
#include <XrdCrypto/XrdCryptoX509Crl.hh>

// ---------------------------------------------------------------------------//
//
// X509 CRL interface
// Describes one CRL certificate
//
// ---------------------------------------------------------------------------//

class XrdSutCache;
class XrdCryptoX509;

class XrdCryptosslX509Crl : public XrdCryptoX509Crl {
public:

   XrdCryptosslX509Crl(const char *crlf, int opt = 0);
   XrdCryptosslX509Crl(XrdCryptoX509 *cacert);
   virtual ~XrdCryptosslX509Crl();

   // Status
   bool IsValid() { return (crl != 0); }

   // Access underlying data (in opaque form: used in chains)
   XrdCryptoX509Crldata Opaque() { return (XrdCryptoX509Crldata)crl; }

   // Dump information
   void Dump();
   const char *ParentFile() { return (const char *)(srcfile.c_str()); }

   // Validity interval
   int  LastUpdate();  // time when last updated
   int  NextUpdate();  // time foreseen for next update

   // Issuer of top certificate
   const char *Issuer();
   const char *IssuerHash();   // hash 

   // Chec certificate revocation
   bool IsRevoked(int serialnumber, int when = 0);
   bool IsRevoked(const char *sernum, int when = 0);

   // Verify signature
   bool Verify(XrdCryptoX509 *ref);

private:
   X509_CRL    *crl;       // The CRL object
   int          lastupdate; // time of last update
   int          nextupdate; // time of next update
   XrdOucString issuer;     // issuer name;
   XrdOucString issuerhash; // hash of issuer name;
   XrdOucString srcfile;    // source file name, if any;
   XrdOucString crluri;     // URI from where to get the CRL file, if any;

   int          nrevoked;   // Number of certificates revoked
   XrdSutCache  cache;      // cached infor about revoked certificates

   int LoadCache();         // Load the cache
   int Init(const char *crlf); // Init from file
   int InitFromURI(const char *uri, const char *hash); // Init from URI
};

#endif
