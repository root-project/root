// $Id$

const char *XrdCryptosslMsgDigestCVSID = "$Id$";
/******************************************************************************/
/*                                                                            */
/*               X r d C r y p t o M s g D i g e s t . c c                    */
/*                                                                            */
/* (c) 2004 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*       All Rights Reserved. See XrdInfo.cc for complete License Terms       */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

/* ************************************************************************** */
/*                                                                            */
/* OpenSSL implementation of XrdCryptoMsgDigest                               */
/*                                                                            */
/* ************************************************************************** */

#include <XrdCrypto/XrdCryptoAux.hh>
#include <XrdCrypto/XrdCryptosslTrace.hh>
#include <XrdCrypto/XrdCryptosslMsgDigest.hh>

//_____________________________________________________________________________
XrdCryptosslMsgDigest::XrdCryptosslMsgDigest(const char *dgst) : 
                     XrdCryptoMsgDigest()
{
   // Constructor.
   // Init the message digest calculation

   valid = 0;
   SetType(0);
   Init(dgst);
}

//_____________________________________________________________________________
bool XrdCryptosslMsgDigest::IsSupported(const char *dgst)
{
   // Check if the specified MD is supported

   return (EVP_get_digestbyname(dgst) != 0);
}

//_____________________________________________________________________________
int XrdCryptosslMsgDigest::Init(const char *dgst)
{
   // Initialize the buffer for the message digest calculation
   EPNAME("MsgDigest::Init");

   // Get message digest handle
   const EVP_MD *md = 0;
   // We first try the one input, if any
   if (dgst)
      md = EVP_get_digestbyname(dgst);

   // If it did not work, we reuse the old one, or we use the default
   if (!md) {
      if (Type())
         md = EVP_get_digestbyname(Type());
      else
         md = EVP_get_digestbyname("sha1");
   }
   if (!md) {
      DEBUG("cannot get msg digest by name");
      return -1;
   }

   // Init digest machine
   EVP_DigestInit(&mdctx, md);

   // Successful initialization
   SetType(dgst);
   valid = 1;

   // OK
   return 0;   
}

//_____________________________________________________________________________
int XrdCryptosslMsgDigest::Reset(const char *dgst)
{
   // Re-Init the message digest calculation

   valid = 0;
   Init(dgst);
   if (!valid)
      // unsuccessful initialization
      return -1;

   return 0;
}

//_____________________________________________________________________________
int XrdCryptosslMsgDigest::Update(const char *b, int l)
{
   // Update message digest with the MD of l bytes at b.
   // Create the internal buffer if needed (first call)
   // Returns -1 if unsuccessful (digest not initialized), 0 otherwise.

   if (Type()) {
      EVP_DigestUpdate(&mdctx, (char *)b, l);
      return 0;
   }
   return -1;   
}

//_____________________________________________________________________________
int XrdCryptosslMsgDigest::Final()
{
   // Finalize message digest calculation.
   // Finalize the operation
   // Returns -1 if unsuccessful (digest not initialized), 0 otherwise.
   EPNAME("MsgDigest::Final");

   // MD outputs in these variables
   unsigned char mdval[EVP_MAX_MD_SIZE] = {0};
   unsigned int mdlen = 0;

   if (Type()) {
      // Finalize what we have
      EVP_DigestFinal(&mdctx, mdval, &mdlen);
      // Save result
      SetBuffer(mdlen,(const char *)mdval);
      // Notify, if requested
      DEBUG("result length is "<<mdlen <<
            " bytes (hex: " << AsHexString() <<")");
      return 0;
   }
   return -1;   
}
