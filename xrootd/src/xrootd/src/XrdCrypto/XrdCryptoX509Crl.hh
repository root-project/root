// $Id$
#ifndef __CRYPTO_X509CRL_H__
#define __CRYPTO_X509CRL_H__
/******************************************************************************/
/*                                                                            */
/*                   X r d C r y p t o X 5 0 9 C r l . h h                    */
/*                                                                            */
/*                                                                            */
/* (c) 2005 G. Ganis , CERN                                                   */
/*                                                                            */
/******************************************************************************/

/* ************************************************************************** */
/*                                                                            */
/* Abstract interface for X509 CRLs        .                                  */
/* Allows to plug-in modules based on different crypto implementation         */
/* (OpenSSL, Botan, ...)                                                      */
/*                                                                            */
/* ************************************************************************** */

#include <XrdCrypto/XrdCryptoX509.hh>

typedef void * XrdCryptoX509Crldata;

// ---------------------------------------------------------------------------//
//
// X509 CRL interface
// Describes one CRL certificate
//
// ---------------------------------------------------------------------------//
class XrdCryptoX509Crl {
public:

   XrdCryptoX509Crl() { }
   virtual ~XrdCryptoX509Crl() { }

   // Status
   virtual bool IsValid();
   virtual bool IsExpired(int when = 0);  // Expired

   // Access underlying data (in opaque form: used in chains)
   virtual XrdCryptoX509Crldata Opaque();

   // Dump information
   virtual void Dump();
   virtual const char *ParentFile();

   // Validity interval
   virtual int  LastUpdate();  // time when last updated
   virtual int  NextUpdate();  // time foreseen for next update

   // Issuer of top certificate
   virtual const char *Issuer();
   virtual const char *IssuerHash();   // hash 

   // Chec certificate revocation
   virtual bool IsRevoked(int serialnumber, int when);
   virtual bool IsRevoked(const char *sernum, int when);

   // Verify signature
   virtual bool Verify(XrdCryptoX509 *ref);

};

#endif
