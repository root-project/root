// $Id$
#ifndef __CRYPTO_LOCALCIPHER_H__
#define __CRYPTO_LOCALCIPHER_H__
/******************************************************************************/
/*                                                                            */
/*               X r d C r y p t o L o c a l C i p h e r . h h                */
/*                                                                            */
/* (c) 2004 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*       All Rights Reserved. See XrdInfo.cc for complete License Terms       */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

/* ************************************************************************** */
/*                                                                            */
/* Local implentation of XrdCryptoCipher based on PC1.                        */
/*                                                                            */
/* ************************************************************************** */

#include <XrdCrypto/XrdCryptoCipher.hh>

// ---------------------------------------------------------------------------//
//
// Cipher interface
//
// ---------------------------------------------------------------------------//
class XrdCryptolocalCipher : public XrdCryptoCipher
{
private:
   bool valid;
   unsigned char *bpub;      // Key agreement: temporary store local public info
   unsigned char *bpriv;     // Key agreement: temporary store local private info

public:
   XrdCryptolocalCipher(const char *t = "PC1", int l = 0);
   XrdCryptolocalCipher(const char *t, int l, const char *k);
   XrdCryptolocalCipher(XrdSutBucket *b);
   XrdCryptolocalCipher(int len, char *pub, int lpub, const char *t = "PC1");
   XrdCryptolocalCipher(const XrdCryptolocalCipher &c);
   virtual ~XrdCryptolocalCipher() { Cleanup(); }

   // Finalize key computation (key agreement)
   bool Finalize(char *pub, int lpub, const char *t = "PC1");
   void Cleanup();

   // Validity
   bool IsValid() { return valid; }

   // Additional getters
   XrdSutBucket *AsBucket();
   bool IsDefaultLength() const;
   char *Public(int &lpub);

   // Required buffer size for encrypt / decrypt operations on l bytes
   int EncOutLength(int l);
   int DecOutLength(int l);

   // Additional methods
   int Encrypt(const char *in, int lin, char *out);
   int Decrypt(const char *in, int lin, char *out);
};

#endif
