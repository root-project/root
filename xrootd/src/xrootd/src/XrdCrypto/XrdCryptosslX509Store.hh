// $Id$
#ifndef __CRYPTO_SSLX509STORE_H__
#define __CRYPTO_SSLX509STORE_H__
/******************************************************************************/
/*                                                                            */
/*               X r d C r y p t o s s l X 5 0 9 S t o r e . h h              */
/*                                                                            */
/* (c) 2005 G. Ganis , CERN                                                   */
/*                                                                            */
/******************************************************************************/

/* ************************************************************************** */
/*                                                                            */
/* OpenSSL implementation of XrdCryptoX509Store                               */
/*                                                                            */
/* ************************************************************************** */

#include <XrdCrypto/XrdCryptoX509Store.hh>

// ---------------------------------------------------------------------------//
//
// OpenSSL X509 implementation
//
// ---------------------------------------------------------------------------//
class XrdCryptosslX509Store : public XrdCryptoX509Store
{
public:
   XrdCryptosslX509Store();
   virtual ~XrdCryptosslX509Store();

   // Dump information
   void Dump();

   // Validity
   bool IsValid();

   // Add certificates to store
   int  Add(XrdCryptoX509 *);

   // Verify the chain stored
   bool Verify();

private:
   X509_STORE     *store;        // the store
   STACK_OF(X509) *chain;        // chain of certificates other than CA
};

#endif
