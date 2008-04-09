// $Id$
#ifndef __CRYPTO_SSLCIPHER_H__
#define __CRYPTO_SSLCIPHER_H__
/******************************************************************************/
/*                                                                            */
/*                  X r d C r y p t o S s l C i p h e r . h h                 */
/*                                                                            */
/* (c) 2004 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*       All Rights Reserved. See XrdInfo.cc for complete License Terms       */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

/* ************************************************************************** */
/*                                                                            */
/* OpenSSL implementation of XrdCryptoCipher                                  */
/*                                                                            */
/* ************************************************************************** */

#include <XrdCrypto/XrdCryptoCipher.hh>

#include <openssl/evp.h>
#include <openssl/dh.h>

#define kDHMINBITS 128

// ---------------------------------------------------------------------------//
//
// OpenSSL Cipher Implementation
//
// ---------------------------------------------------------------------------//
class XrdCryptosslCipher : public XrdCryptoCipher
{
private:
   char       *fIV;
   int         lIV;
   const EVP_CIPHER *cipher;
   EVP_CIPHER_CTX ctx;
   DH         *fDH;
   bool        deflength;
   bool        valid;

   void        GenerateIV();
   int         EncDec(int encdec, const char *bin, int lin, char *out);
   void        PrintPublic(BIGNUM *pub);
   int         Publen();

public:
   XrdCryptosslCipher(const char *t, int l = 0);
   XrdCryptosslCipher(const char *t, int l, const char *k,
                                     int liv, const char *iv);
   XrdCryptosslCipher(XrdSutBucket *b);
   XrdCryptosslCipher(int len, char *pub, int lpub, const char *t);
   XrdCryptosslCipher(const XrdCryptosslCipher &c);
   virtual ~XrdCryptosslCipher();

   // Finalize key computation (key agreement)
   bool Finalize(char *pub, int lpub, const char *t);
   void Cleanup();

   // Validity
   bool IsValid() { return valid; }

   // Support
   static bool IsSupported(const char *cip);

   // Required buffer size for encrypt / decrypt operations on l bytes
   int EncOutLength(int l);
   int DecOutLength(int l);
   char *Public(int &lpub);

   // Additional getter
   XrdSutBucket *AsBucket();
   char *IV(int &l) const { l = lIV; return fIV; }
   bool IsDefaultLength() const { return deflength; }

   // Additional setter
   void  SetIV(int l, const char *iv);

   // Additional methods
   int Encrypt(const char *bin, int lin, char *out);
   int Decrypt(const char *bin, int lin, char *out);
   char *RefreshIV(int &l);
};
#endif
