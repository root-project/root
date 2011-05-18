// $Id$

const char *XrdCryptoCipherCVSID = "$Id$";
/******************************************************************************/
/*                                                                            */
/*                   X r d C r y p t o C i p h e r . c c                      */
/*                                                                            */
/* (c) 2004 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*       All Rights Reserved. See XrdInfo.cc for complete License Terms       */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

/* ************************************************************************** */
/*                                                                            */
/* Generic interface to a cipher class                                        */
/* Allows to plug-in modules based on different crypto implementation         */
/* (OpenSSL, Botan, ...)                                                      */
/*                                                                            */
/* ************************************************************************** */

#include <string.h>

#include <XrdCrypto/XrdCryptoAux.hh>
#include <XrdCrypto/XrdCryptoCipher.hh>

//_____________________________________________________________________________
bool XrdCryptoCipher::Finalize(char *, int, const char *)
{
   // Finalize key computation (key agreement)
   ABSTRACTMETHOD("XrdCryptoCipher::Finalize");
   return 0;
}

//_____________________________________________________________________________
bool XrdCryptoCipher::IsValid()
{
   // Check key validity
   ABSTRACTMETHOD("XrdCryptoCipher::IsValid");
   return 0;
}

//____________________________________________________________________________
void XrdCryptoCipher::SetIV(int l, const char *iv)
{
   // Set IV from l bytes at iv

   ABSTRACTMETHOD("XrdCryptoCipher::SetIV");
}

//____________________________________________________________________________
char *XrdCryptoCipher::RefreshIV(int &l)
{
   // Regenerate IV and return it

   ABSTRACTMETHOD("XrdCryptoCipher::RefreshIV");
   return 0;
}

//____________________________________________________________________________
char *XrdCryptoCipher::IV(int &l) const
{
   // Get IV

   ABSTRACTMETHOD("XrdCryptoCipher::IV");
   return 0;
}

//____________________________________________________________________________
char *XrdCryptoCipher::Public(int &lpub)
{
   // Getter for public part during key agreement

   ABSTRACTMETHOD("XrdCryptoCipher::Public");
   return 0;
}

//_____________________________________________________________________________
XrdSutBucket *XrdCryptoCipher::AsBucket()
{
   // Return pointer to a bucket created using the internal information
   // serialized
 
   ABSTRACTMETHOD("XrdCryptoCipher::AsBucket");
   return 0;
}
//____________________________________________________________________________
int XrdCryptoCipher::Encrypt(const char *, int, char *)
{
   // Encrypt lin bytes at in with local cipher.

   ABSTRACTMETHOD("XrdCryptoCipher::Encrypt");
   return 0;
}

//____________________________________________________________________________
int XrdCryptoCipher::Decrypt(const char *, int, char *)
{
   // Decrypt lin bytes at in with local cipher.

   ABSTRACTMETHOD("XrdCryptoCipher::Decrypt");
   return 0;
}

//____________________________________________________________________________
int XrdCryptoCipher::EncOutLength(int)
{
   // Required buffer size for encrypting l bytes

   ABSTRACTMETHOD("XrdCryptoCipher::EncOutLength");
   return 0;
}

//____________________________________________________________________________
int XrdCryptoCipher::DecOutLength(int)
{
   // Required buffer size for decrypting l bytes

   ABSTRACTMETHOD("XrdCryptoCipher::DecOutLength");
   return 0;
}

//____________________________________________________________________________
bool XrdCryptoCipher::IsDefaultLength() const
{
   // Test if cipher length is the default one

   ABSTRACTMETHOD("XrdCryptoCipher::IsDefaultLength");
   return 0;
}

//____________________________________________________________________________
int XrdCryptoCipher::Encrypt(XrdSutBucket &bck)
{
   // Encrypt bucket bck with local cipher
   // Return size of encoded bucket or -1 in case of error
   int snew = -1;

   int sz = EncOutLength(bck.size);
   char *newbck = new char[sz];
   if (newbck) {
      memset(newbck, 0, sz);
      snew = Encrypt(bck.buffer,bck.size,newbck);
      if (snew > -1)
         bck.Update(newbck,snew);
   }
   return snew;
}

//____________________________________________________________________________
int XrdCryptoCipher::Decrypt(XrdSutBucket &bck)
{
   // Decrypt bucket bck with local cipher
   // Return size of encoded bucket or -1 in case of error
   int snew = -1;

   int sz = DecOutLength(bck.size);
   char *newbck = new char[sz];
   if (newbck) {
      memset(newbck, 0, sz);
      snew = Decrypt(bck.buffer,bck.size,newbck);
      if (snew > -1)
         bck.Update(newbck,snew);
   }
   return snew;
}
