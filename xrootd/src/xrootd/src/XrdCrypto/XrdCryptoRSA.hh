// $Id$
#ifndef __CRYPTO_RSA_H__
#define __CRYPTO_RSA_H__
/******************************************************************************/
/*                                                                            */
/*                       X r d C r y p t o R S A . h h                        */
/*                                                                            */
/* (c) 2004 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*       All Rights Reserved. See XrdInfo.cc for complete License Terms       */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

/* ************************************************************************** */
/*                                                                            */
/* Abstract interface for RSA PKI functionality.                              */
/* Allows to plug-in modules based on different crypto implementation         */
/* (OpenSSL, Botan, ...)                                                      */
/*                                                                            */
/* ************************************************************************** */

#include <XrdSut/XrdSutBucket.hh>
#include <XrdOuc/XrdOucString.hh>
#include <XrdCrypto/XrdCryptoAux.hh>

typedef void * XrdCryptoRSAdata;

// ---------------------------------------------------------------------------//
//
// RSA interface
//
// ---------------------------------------------------------------------------//
class XrdCryptoRSA
{
public:
   XrdCryptoRSA() { status = kInvalid; }
   virtual ~XrdCryptoRSA() {}

   // Status
   enum ERSAStatus { kInvalid = 0, kPublic = 1, kComplete = 2};
   ERSAStatus  status;
   const char *Status(ERSAStatus t = kInvalid) const
                 { return ((t == kInvalid) ? cstatus[status] : cstatus[t]); }

   // Access underlying data (in opaque form)
   virtual XrdCryptoRSAdata Opaque();

   // Dump information
   virtual void Dump();

   // Validity
   bool IsValid() { return (status != kInvalid); }

   // Output lengths
   virtual int GetOutlen(int lin);   // Length of encrypted buffers
   virtual int GetPublen();          // Length of export public key
   virtual int GetPrilen();          // Length of export private key

   // Import / Export methods
   virtual int ImportPublic(const char *in, int lin);
   virtual int ExportPublic(char *out, int lout);
   int ExportPublic(XrdOucString &exp);
   virtual int ImportPrivate(const char *in, int lin);
   virtual int ExportPrivate(char *out, int lout);
   int ExportPrivate(XrdOucString &exp);

   // Encryption / Decryption methods
   virtual int EncryptPrivate(const char *in, int lin, char *out, int lout);
   virtual int DecryptPublic(const char *in, int lin, char *out, int lout);
   virtual int EncryptPublic(const char *in, int lin, char *out, int lout);
   virtual int DecryptPrivate(const char *in, int lin, char *out, int lout);
   int EncryptPrivate(XrdSutBucket &buck);
   int DecryptPublic (XrdSutBucket &buck);
   int EncryptPublic (XrdSutBucket &buck);
   int DecryptPrivate(XrdSutBucket &buck);

private:
   static const char *cstatus[3];  // Names of status
};

#endif
