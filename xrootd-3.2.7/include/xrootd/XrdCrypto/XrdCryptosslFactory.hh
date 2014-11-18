// $Id$
#ifndef __CRYPTO_SSLFACTORY_H__
#define __CRYPTO_SSLFACTORY_H__
/******************************************************************************/
/*                                                                            */
/*               X r d C r y p t o S s l F a c t o r y . h h                  */
/*                                                                            */
/* (c) 2004 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*       All Rights Reserved. See XrdInfo.cc for complete License Terms       */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

/* ************************************************************************** */
/*                                                                            */
/* Implementation of the OpenSSL crypto factory                               */
/*                                                                            */
/* ************************************************************************** */

#ifndef __CRYPTO_FACTORY_H__
#include "XrdCrypto/XrdCryptoFactory.hh"
#endif

#include "XrdSys/XrdSysPthread.hh"

int DebugON = 1;

// The ID must be a unique number
#define XrdCryptosslFactoryID  1

#define SSLFACTORY_MAX_CRYPTO_MUTEX 256

class XrdCryptosslFactory : public XrdCryptoFactory 
{
public:
   XrdCryptosslFactory();
   virtual ~XrdCryptosslFactory() { }

   // Set trace flags
   void SetTrace(kXR_int32 trace);

   // Hook to Key Derivation Function (PBKDF2)
   XrdCryptoKDFunLen_t KDFunLen(); // Default Length of buffer
   XrdCryptoKDFun_t KDFun();

   // Cipher constructors
   bool SupportedCipher(const char *t);
   XrdCryptoCipher *Cipher(const char *t, int l = 0);
   XrdCryptoCipher *Cipher(const char *t, int l, const char *k,
                                          int liv, const char *iv);
   XrdCryptoCipher *Cipher(XrdSutBucket *b);
   XrdCryptoCipher *Cipher(int bits, char *pub, int lpub, const char *t = 0);
   XrdCryptoCipher *Cipher(const XrdCryptoCipher &c);

   // MsgDigest constructors
   bool SupportedMsgDigest(const char *dgst);
   XrdCryptoMsgDigest *MsgDigest(const char *dgst);

   // RSA constructors
   XrdCryptoRSA *RSA(int bits = XrdCryptoDefRSABits, int exp = XrdCryptoDefRSAExp);
   XrdCryptoRSA *RSA(const char *pub, int lpub = 0);
   XrdCryptoRSA *RSA(const XrdCryptoRSA &r);

   // X509 constructors
   XrdCryptoX509 *X509(const char *cf, const char *kf = 0);
   XrdCryptoX509 *X509(XrdSutBucket *b);

   // X509 CRL constructor
   XrdCryptoX509Crl *X509Crl(const char *crlfile, int opt = 0);
   XrdCryptoX509Crl *X509Crl(XrdCryptoX509 *cacert);

   // X509 REQ constructors
   XrdCryptoX509Req *X509Req(XrdSutBucket *bck);

   // Hooks to handle X509 certificates
   XrdCryptoX509VerifyCert_t X509VerifyCert();
   XrdCryptoX509VerifyChain_t X509VerifyChain();
   XrdCryptoX509ParseFile_t X509ParseFile();
   XrdCryptoX509ParseBucket_t X509ParseBucket();
   XrdCryptoX509ExportChain_t X509ExportChain();
   XrdCryptoX509ChainToFile_t X509ChainToFile();

   // Required SSL mutexes.
  static  XrdSysMutex*              CryptoMutexPool[SSLFACTORY_MAX_CRYPTO_MUTEX];

};

#endif
