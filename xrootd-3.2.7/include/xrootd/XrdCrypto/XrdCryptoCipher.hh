// $Id$
#ifndef __CRYPTO_CIPHER_H__
#define __CRYPTO_CIPHER_H__
/******************************************************************************/
/*                                                                            */
/*                     X r d C r y p t o C i p h e r . h h                    */
/*                                                                            */
/* (c) 2004 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*       All Rights Reserved. See XrdInfo.cc for complete License Terms       */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

/* ************************************************************************** */
/*                                                                            */
/* Abstract interface for a symmetric Cipher functionality.                   */
/* Allows to plug-in modules based on different crypto implementation         */
/* (OpenSSL, Botan, ...)                                                      */
/*                                                                            */
/* ************************************************************************** */

#include <XrdSut/XrdSutBucket.hh>
#include <XrdCrypto/XrdCryptoBasic.hh>

// ---------------------------------------------------------------------------//
//
// Cipher interface
//
// ---------------------------------------------------------------------------//
class XrdCryptoCipher : public XrdCryptoBasic
{
public:
   XrdCryptoCipher() : XrdCryptoBasic() {}
   virtual ~XrdCryptoCipher() {}

   // Finalize key computation (key agreement)
   virtual bool Finalize(char *pub, int lpub, const char *t);

   // Validity
   virtual bool IsValid();

   // Required buffer size for encrypt / decrypt operations on l bytes
   virtual int EncOutLength(int l);
   virtual int DecOutLength(int l);

   // Additional getters
   virtual XrdSutBucket *AsBucket();
   virtual char *IV(int &l) const;
   virtual bool IsDefaultLength() const;
   virtual char *Public(int &lpub);

   // Additional setters
   virtual void SetIV(int l, const char *iv);

   // Additional methods
   virtual int Encrypt(const char *in, int lin, char *out);
   virtual int Decrypt(const char *in, int lin, char *out);
   int Encrypt(XrdSutBucket &buck);
   int Decrypt(XrdSutBucket &buck);
   virtual char *RefreshIV(int &l); 
};

#endif
