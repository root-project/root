// $Id$
#ifndef __CRYPTO_MSGDGSTSSL_H__
#define __CRYPTO_MSGDGSTSSL_H__
/******************************************************************************/
/*                                                                            */
/*             X r d C r y p t o S s l M s g D i g e s t . h h                */
/*                                                                            */
/* (c) 2004 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*       All Rights Reserved. See XrdInfo.cc for complete License Terms       */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

/* ************************************************************************** */
/*                                                                            */
/* OpenSSL implementation of XrdSecCMsgDigest                                 */
/*                                                                            */
/* ************************************************************************** */

#include <openssl/evp.h>

#include <XrdCrypto/XrdCryptoMsgDigest.hh>

// ---------------------------------------------------------------------------//
//
// Message Digest implementation buffer
//
// ---------------------------------------------------------------------------//
class XrdCryptosslMsgDigest : public XrdCryptoMsgDigest
{
private:
   bool valid;
   EVP_MD_CTX mdctx;

   int Init(const char *dgst);

public:
   XrdCryptosslMsgDigest(const char *dgst);
   virtual ~XrdCryptosslMsgDigest() { }

   // Validity
   bool IsValid() { return valid; }

   // Support
   static bool IsSupported(const char *dgst);

   int Reset(const char *dgst = 0);
   int Update(const char *b, int l);
   int Final();
};

#endif
