// $Id$
#ifndef __CRYPTO_SSLRSA_H__
#define __CRYPTO_SSLRSA_H__
/******************************************************************************/
/*                                                                            */
/*                   X r d C r y p t o S s l R S A . h h                      */
/*                                                                            */
/* (c) 2004 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*       All Rights Reserved. See XrdInfo.cc for complete License Terms       */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

/* ************************************************************************** */
/*                                                                            */
/* OpenSSL implementation of XrdCryptoRSA                                     */
/*                                                                            */
/* ************************************************************************** */

#include <XrdCrypto/XrdCryptoRSA.hh>

#include <openssl/evp.h>

// ---------------------------------------------------------------------------//
//
// RSA interface
//
// ---------------------------------------------------------------------------//
class XrdCryptosslRSA : public XrdCryptoRSA
{
private:
   EVP_PKEY *fEVP;     // The key pair
   int       publen;   // Length of export public key
   int       prilen;   // Length of export private key

public:
   XrdCryptosslRSA(int bits = XrdCryptoMinRSABits, int exp = XrdCryptoDefRSAExp);
   XrdCryptosslRSA(const char *pub, int lpub = 0);
   XrdCryptosslRSA(EVP_PKEY *key, bool check = 1);
   XrdCryptosslRSA(const XrdCryptosslRSA &r);
   virtual ~XrdCryptosslRSA();

   // Access underlying data (in opaque form)
   XrdCryptoRSAdata Opaque() { return fEVP; }

   // Dump information
   void Dump();

   // Output lengths
   int GetOutlen(int lin);   // Length of encrypted buffers
   int GetPublen();          // Length of export public key
   int GetPrilen();          // Length of export private key

   // Import / Export methods
   int ImportPublic(const char *in, int lin);
   int ExportPublic(char *out, int lout);
   int ImportPrivate(const char *in, int lin);
   int ExportPrivate(char *out, int lout);

   // Encryption / Decryption methods
   int EncryptPrivate(const char *in, int lin, char *out, int lout);
   int DecryptPublic(const char *in, int lin, char *out, int lout);
   int EncryptPublic(const char *in, int lin, char *out, int lout);
   int DecryptPrivate(const char *in, int lin, char *out, int lout);
};

#endif
