// $Id$

const char *XrdCryptoRSACVSID = "$Id$";
/******************************************************************************/
/*                                                                            */
/*                       X r d C r y p t o R S A . c c                        */
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

#include <string.h>

#include <XrdCrypto/XrdCryptoRSA.hh>

const char *XrdCryptoRSA::cstatus[3] = { "Invalid", "Public", "Complete" };

//_____________________________________________________________________________
void XrdCryptoRSA::Dump()
{
   // Check key validity
   ABSTRACTMETHOD("XrdCryptoRSA::Dump");
}

//_____________________________________________________________________________
XrdCryptoRSAdata XrdCryptoRSA::Opaque()
{
   // Return underlying key in raw format
   ABSTRACTMETHOD("XrdCryptoRSA::Opaque");
   return (XrdCryptoRSAdata)0;
}

//_____________________________________________________________________________
int XrdCryptoRSA::GetOutlen(int)
{
   // Get length of output
   ABSTRACTMETHOD("XrdCryptoRSA::GetOutlen");
   return 0;
}

//_____________________________________________________________________________
int XrdCryptoRSA::GetPublen()
{
   // Get length of public key export form
   ABSTRACTMETHOD("XrdCryptoRSA::GetPublen");
   return 0;
}

//_____________________________________________________________________________
int XrdCryptoRSA::GetPrilen()
{
   // Get length of private key export form
   ABSTRACTMETHOD("XrdCryptoRSA::GetPrilen");
   return 0;
}

//_____________________________________________________________________________
int XrdCryptoRSA::ImportPublic(const char *, int)
{
   // Abstract method to import a public key
   ABSTRACTMETHOD("XrdCryptoRSA::ImportPublic");
   return -1;
}

//_____________________________________________________________________________
int XrdCryptoRSA::ExportPublic(char *, int)
{
   // Abstract method to export the public key
   ABSTRACTMETHOD("XrdCryptoRSA::ExportPublic");
   return -1;
}

//_____________________________________________________________________________
int XrdCryptoRSA::ImportPrivate(const char *, int)
{
   // Abstract method to import a private key
   ABSTRACTMETHOD("XrdCryptoRSA::ImportPrivate");
   return -1;
}

//_____________________________________________________________________________
int XrdCryptoRSA::ExportPrivate(char *, int)
{
   // Abstract method to export the private key
   ABSTRACTMETHOD("XrdCryptoRSA::ExportPrivate");
   return -1;
}

//_____________________________________________________________________________
int XrdCryptoRSA::ExportPublic(XrdOucString &s)
{
   // Export the public key into string s

   int newlen = GetPublen();
   if (newlen > 0) {
      char *newbuf = new char[newlen+1];
      if (newbuf) {
         memset(newbuf, 0, newlen+1);
         if (ExportPublic(newbuf,newlen+1) > -1) {
            s = (const char *)newbuf;
            delete[] newbuf;
            return 0;
         }
         delete[] newbuf;
      }
   }
   return -1;
}

//_____________________________________________________________________________
int XrdCryptoRSA::ExportPrivate(XrdOucString &s)
{
   // Export the private key into string s

   int newlen = GetPrilen();
   if (newlen > 0) {
      char *newbuf = new char[newlen+1];
      if (newbuf) {
         memset(newbuf, 0, newlen+1);
         if (ExportPrivate(newbuf,newlen+1) > -1) {
            s = (const char *)newbuf;
            delete[] newbuf;
            return 0;
         }
         delete[] newbuf;
      }
   }
   return -1;
}

//_____________________________________________________________________________
int XrdCryptoRSA::EncryptPrivate(const char *, int, char *, int)
{
   // Abstract method to encrypt using the private key
   ABSTRACTMETHOD("XrdCryptoRSA::EncryptPrivate");
   return -1;
}

//_____________________________________________________________________________
int XrdCryptoRSA::EncryptPublic(const char *, int, char *, int)
{
   // Abstract method to encrypt using the public key
   ABSTRACTMETHOD("XrdCryptoRSA::EncryptPublic");
   return -1;
}

//_____________________________________________________________________________
int XrdCryptoRSA::DecryptPrivate(const char *, int, char *, int)
{
   // Abstract method to decrypt using the private key
   ABSTRACTMETHOD("XrdCryptoRSA::DecryptPrivate");
   return -1;
}

//_____________________________________________________________________________
int XrdCryptoRSA::DecryptPublic(const char *, int, char *, int)
{
   // Abstract method to decrypt using the public key
   ABSTRACTMETHOD("XrdCryptoRSA::DecryptPublic");
   return -1;
}

//_____________________________________________________________________________
int XrdCryptoRSA::EncryptPrivate(XrdSutBucket &bck)
{
   // Encrypt bucket bck using the private key
   // Return new bucket size, or -1 in case of error
   int snew = -1;

   int sz = GetOutlen(bck.size);
   char *newbuf = new char[sz];
   if (newbuf) {
      memset(newbuf, 0, sz);
      snew = EncryptPrivate(bck.buffer,bck.size,newbuf,sz);
      if (snew > -1)
         bck.Update(newbuf,snew);
   }
   return snew;
}

//_____________________________________________________________________________
int XrdCryptoRSA::EncryptPublic(XrdSutBucket &bck)
{
   // Encrypt bucket bck using the public key
   // Return new bucket size, or -1 in case of error
   int snew = -1;

   int sz = GetOutlen(bck.size);
   char *newbuf = new char[sz];
   if (newbuf) {
      memset(newbuf, 0, sz);
      snew = EncryptPublic(bck.buffer,bck.size,newbuf,sz);
      if (snew > -1)
         bck.Update(newbuf,snew);
   }
   return snew;
}

//_____________________________________________________________________________
int XrdCryptoRSA::DecryptPrivate(XrdSutBucket &bck)
{
   // Decrypt bucket bck using the private key
   // Return new bucket size, or -1 in case of error
   int snew = -1;

   int sz = GetOutlen(bck.size);
   char *newbuf = new char[sz];
   if (newbuf) {
      memset(newbuf, 0, sz);
      snew = DecryptPrivate(bck.buffer,bck.size,newbuf,sz);
      if (snew > -1)
         bck.Update(newbuf,snew);
   }
   return snew;
}

//_____________________________________________________________________________
int XrdCryptoRSA::DecryptPublic(XrdSutBucket &bck)
{
   // Decrypt bucket bck using the public key
   // Return new bucket size, or -1 in case of error
   int snew = -1;

   int sz = GetOutlen(bck.size);
   char *newbuf = new char[sz];
   if (newbuf) {
      memset(newbuf, 0, sz);
      snew = DecryptPublic(bck.buffer,bck.size,newbuf,sz);
      if (snew > -1)
         bck.Update(newbuf,snew);
   }
   return snew;
}
