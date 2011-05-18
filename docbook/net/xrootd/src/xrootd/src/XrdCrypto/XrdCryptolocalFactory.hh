// $Id$
#ifndef __CRYPTO_LOCALFACTORY_H__
#define __CRYPTO_LOCALFACTORY_H__
/******************************************************************************/
/*                                                                            */
/*             X r d C r y p t o L o c a l F a c t o r y . h h                */
/*                                                                            */
/* (c) 2004 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*       All Rights Reserved. See XrdInfo.cc for complete License Terms       */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

/* ************************************************************************** */
/*                                                                            */
/* Implementation of the local crypto factory                                 */
/*                                                                            */
/* ************************************************************************** */

#include <XrdCrypto/XrdCryptoFactory.hh>

// The ID must be a unique number
#define XrdCryptolocalFactoryID  0

class XrdCryptolocalFactory : public XrdCryptoFactory 
{
public:
   XrdCryptolocalFactory();
   virtual ~XrdCryptolocalFactory() { }

   // Set trace flags
   void SetTrace(kXR_int32 trace);

   // Hook to local KDFun
   XrdCryptoKDFunLen_t KDFunLen(); // Length of buffer
   XrdCryptoKDFun_t KDFun();

   // Cipher constructors
   XrdCryptoCipher *Cipher(const char *t, int l = 0);
   XrdCryptoCipher *Cipher(const char *t, int l, const char *k,
                                          int liv, const char *iv);
   XrdCryptoCipher *Cipher(XrdSutBucket *b);
   XrdCryptoCipher *Cipher(int bits, char *pub, int lpub, const char *t = 0);
   XrdCryptoCipher *Cipher(const XrdCryptoCipher &c);

   // MsgDigest constructors
   XrdCryptoMsgDigest *MsgDigest(const char *dgst);

   // RSA constructors
   XrdCryptoRSA *RSA(int bits = XrdCryptoDefRSABits, int exp = XrdCryptoDefRSAExp);
   XrdCryptoRSA *RSA(const char *pub, int lpub = 0);
   XrdCryptoRSA *RSA(const XrdCryptoRSA &r);
};

#endif
