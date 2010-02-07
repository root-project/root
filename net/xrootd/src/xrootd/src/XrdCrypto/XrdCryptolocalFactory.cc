// $Id$

const char *XrdCryptolocalFactoryCVSID = "$Id$";
/******************************************************************************/
/*                                                                            */
/*          X r d C r y p t o L o c a l F a c t o r y . c c                   */
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

#include <XrdCrypto/PC1.hh>
#include <XrdCrypto/XrdCryptolocalCipher.hh>
#include <XrdCrypto/XrdCryptolocalFactory.hh>
#include <XrdCrypto/XrdCryptoTrace.hh>

#include <string.h>
#include <stdlib.h>
#include <errno.h>

//____________________________________________________________________________
static int XrdCryptolocalKDFunLen()
{
   // Length of buffer needed by XrdCryptolocalKDFun

   return (2*kPC1LENGTH + 1);
}
//____________________________________________________________________________
static int XrdCryptolocalKDFun(const char *pass, int plen,
                               const char *salt, int slen,
                               char *key, int)
{
   // Wrapper to the PSC (Pukall Stream Cipher) Hash Function, returning 
   // a 256-bits hash (http://membres.lycos.fr/pc1/).
   // Max length for pass and salt is 32 bytes (256 bits).
   // Additional bytes are ignored.
   // The output is a null-terminated human readable 64-byte string (65 bytes).
   // The caller is responsible to allocate enough space to contain it.
   // The length of the output string is returned or -1 in case of problems.
   // The author sets the number of iterations to 63254; this will be 
   // the default.
   // It can be specified at the beginning of the salt using a construct
   // like this: salt = "$$<number_of_iterations>$<effective_salt>"

   // Defaults
   char *realsalt = (char *)salt;
   int realslen = slen;
   int it = 63254;
   //
   // Extract iteration number from salt, if any
   char *ibeg = (char *)memchr(salt+1,'$',slen-1);
   if (ibeg) {
      char *del = 0;
      int newit = strtol(ibeg+1, &del, 10);
      if (newit > 0 && del[0] == '$' && errno != ERANGE) {
         // found iteration number
         it = newit;
         realsalt = del+1;
         realslen = slen - (int)(realsalt-salt);
      }
   }

   //
   // Calculate one-way hash
   return PC1HashFun(pass, plen, realsalt, realslen, it, key);
}

//______________________________________________________________________________
XrdCryptolocalFactory::XrdCryptolocalFactory() : 
                       XrdCryptoFactory("local",XrdCryptolocalFactoryID)
{
   // Constructor:
}

//______________________________________________________________________________
void XrdCryptolocalFactory::SetTrace(kXR_int32 trace)
{
   // Set trace flags according to 'trace'

   if (cryptoTrace) {
      // Set debug mask
      cryptoTrace->What = 0;
      // Low level only
      if ((trace & cryptoTRACE_Notify))
         cryptoTrace->What |= cryptoTRACE_Notify;
      // Medium level
      if ((trace & cryptoTRACE_Debug))
         cryptoTrace->What |= (cryptoTRACE_Notify | cryptoTRACE_Debug);
      // High level
      if ((trace & cryptoTRACE_Dump))
         cryptoTrace->What |= cryptoTRACE_ALL;
   }
}

//______________________________________________________________________________
XrdCryptoKDFunLen_t XrdCryptolocalFactory::KDFunLen()
{
   // Return an instance of an implementation of the local KD fun length.

   return &XrdCryptolocalKDFunLen;
}

//______________________________________________________________________________
XrdCryptoKDFun_t XrdCryptolocalFactory::KDFun()
{
   // Return an instance of an implementation of the local KD function.

   return &XrdCryptolocalKDFun;
}

//______________________________________________________________________________
XrdCryptoCipher *XrdCryptolocalFactory::Cipher(const char *t, int l)
{
   // Return an instance of a local implementation of XrdCryptoCipher.

   XrdCryptoCipher *cip = new XrdCryptolocalCipher(t,l);
   if (cip) {
      if (cip->IsValid())
         return cip;
      else
         delete cip;
   }
   return (XrdCryptoCipher *)0;
}

//______________________________________________________________________________
XrdCryptoCipher *XrdCryptolocalFactory::Cipher(const char *t, int l,
                                               const char *k, int, const char *)
{
   // Return an instance of a local implementation of XrdCryptoCipher.

   XrdCryptoCipher *cip = new XrdCryptolocalCipher(t,l,k);
   if (cip) {
      if (cip->IsValid())
         return cip;
      else
         delete cip;
   }
   return (XrdCryptoCipher *)0;
}

//______________________________________________________________________________
XrdCryptoCipher *XrdCryptolocalFactory::Cipher(XrdSutBucket *b)
{
   // Return an instance of a local implementation of XrdCryptoCipher.

   XrdCryptoCipher *cip = new XrdCryptolocalCipher(b);
   if (cip) {
      if (cip->IsValid())
         return cip;
      else
         delete cip;
   }
   return (XrdCryptoCipher *)0;
}

//______________________________________________________________________________
XrdCryptoCipher *XrdCryptolocalFactory::Cipher(int b, char *p,
                                               int l, const char *t)
{
   // Return an instance of a local implementation of XrdCryptoCipher.

   XrdCryptoCipher *cip = new XrdCryptolocalCipher(b,p,l,t);
   if (cip) {
      if (cip->IsValid())
         return cip;
      else
         delete cip;
   }
   return (XrdCryptoCipher *)0;
}

//______________________________________________________________________________
XrdCryptoCipher *XrdCryptolocalFactory::Cipher(const XrdCryptoCipher &c)
{
   // Return an instance of a local implementation of XrdCryptoCipher.

   XrdCryptoCipher *cip = new XrdCryptolocalCipher(*((XrdCryptolocalCipher *)&c));
   if (cip) {
      if (cip->IsValid())
         return cip;
      else
         delete cip;
   }
   return (XrdCryptoCipher *)0;
}

//______________________________________________________________________________
XrdCryptoMsgDigest *XrdCryptolocalFactory::MsgDigest(const char *)
{
   // Return an instance of a local implementation of XrdCryptoMsgDigest.

   ABSTRACTMETHOD("XrdCryptoFactory::MsgDigest");
   return 0;
}

//______________________________________________________________________________
XrdCryptoRSA *XrdCryptolocalFactory::RSA(int bits, int exp)
{
   // Return an instance of a local implementation of XrdCryptoRSA.

   ABSTRACTMETHOD("XrdCryptoFactory::RSA");
   return (XrdCryptoRSA *)0;
}

//______________________________________________________________________________
XrdCryptoRSA *XrdCryptolocalFactory::RSA(const char *pub, int lpub)
{
   // Return an instance of a local implementation of XrdCryptoRSA.

   ABSTRACTMETHOD("XrdCryptoFactory::RSA");
   return (XrdCryptoRSA *)0;
}

//______________________________________________________________________________
XrdCryptoRSA *XrdCryptolocalFactory::RSA(const XrdCryptoRSA &r)
{
   // Return an instance of a local implementation of XrdCryptoRSA.

   ABSTRACTMETHOD("XrdCryptoFactory::RSA");
   return (XrdCryptoRSA *)0;
}
