// $Id$

const char *XrdCryptoMsgDigestCVSID = "$Id$";
/******************************************************************************/
/*                                                                            */
/*                 X r d C r y p t o M s g D i g e s t . c c                  */
/*                                                                            */
/* (c) 2004 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*       All Rights Reserved. See XrdInfo.cc for complete License Terms       */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

/* ************************************************************************** */
/*                                                                            */
/* Abstract interface for Message Digest crypto functionality.                */
/* Allows to plug-in modules based on different crypto implementation         */
/* (OpenSSL, Botan, ...)                                                      */
/*                                                                            */
/* ************************************************************************** */

#include <string.h>

#include <XrdCrypto/XrdCryptoAux.hh>
#include <XrdCrypto/XrdCryptoMsgDigest.hh>

//_____________________________________________________________________________
bool XrdCryptoMsgDigest::IsValid()
{
   // Check key validity
   ABSTRACTMETHOD("XrdCryptoMsgDigest::IsValid");
   return 0;
}

//______________________________________________________________________________
bool XrdCryptoMsgDigest::operator==(const XrdCryptoMsgDigest md)
{
   // Compare msg digest md to local md: return 1 if matches, 0 if not

   if (md.Length() == Length()) {
      if (!memcmp(md.Buffer(),Buffer(),Length()))
         return 1;
   }
   return 0;
}
//_____________________________________________________________________________
int XrdCryptoMsgDigest::Reset(const char *dgst)
{
   // Re-Init the message digest calculation

   ABSTRACTMETHOD("XrdCryptoMsgDigest::Reset");
   return -1;
}

//_____________________________________________________________________________
int XrdCryptoMsgDigest::Update(const char *b, int l)
{
   // Update message digest with the MD of l bytes at b.

   ABSTRACTMETHOD("XrdCryptoMsgDigest::Update");
   return -1;   
}

//_____________________________________________________________________________
int XrdCryptoMsgDigest::Final()
{
   // Finalize message digest calculation.

   ABSTRACTMETHOD("XrdCryptoMsgDigest::Final");
   return -1;   
}
