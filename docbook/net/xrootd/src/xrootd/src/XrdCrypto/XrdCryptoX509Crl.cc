// $Id$

const char *XrdCryptoX509CrlCVSID = "$Id$";
/******************************************************************************/
/*                                                                            */
/*                  X r d C r y p t o X 5 0 9 C r l. c c                      */
/*                                                                            */
/* (c) 2005 G. Ganis , CERN                                                   */
/*                                                                            */
/******************************************************************************/

/* ************************************************************************** */
/*                                                                            */
/* Abstract interface for X509 CRLs.                                          */
/* Allows to plug-in modules based on different crypto implementation         */
/* (OpenSSL, Botan, ...)                                                      */
/*                                                                            */
/* ************************************************************************** */
#include <time.h>
#include <XrdCrypto/XrdCryptoX509Crl.hh>

//_____________________________________________________________________________
void XrdCryptoX509Crl::Dump()
{
   // Dump content
   ABSTRACTMETHOD("XrdCryptoX509Crl::Dump");
}

//_____________________________________________________________________________
bool XrdCryptoX509Crl::IsValid()
{
   // Check validity
   ABSTRACTMETHOD("XrdCryptoX509Crl::IsValid");
   return 0;
}

//_____________________________________________________________________________
bool XrdCryptoX509Crl::IsExpired(int when)
{
   // Check expiration at UTC time 'when'. Use when =0 (default) to check
   // at present time.

   int now = (when > 0) ? when : (int)time(0);
   return (now > NextUpdate());
}

//_____________________________________________________________________________
int XrdCryptoX509Crl::LastUpdate()
{
   // Time of last update
   ABSTRACTMETHOD("XrdCryptoX509Crl::LastUpdate");
   return -1;
}

//_____________________________________________________________________________
int XrdCryptoX509Crl::NextUpdate()
{
   // Time of next update
   ABSTRACTMETHOD("XrdCryptoX509Crl::NextUpdate");
   return -1;
}

//_____________________________________________________________________________
const char *XrdCryptoX509Crl::ParentFile()
{
   // Return parent file name
   ABSTRACTMETHOD("XrdCryptoX509Crl::ParentFile");
   return (const char *)0;
}

//_____________________________________________________________________________
const char *XrdCryptoX509Crl::Issuer()
{
   // Return issuer name
   ABSTRACTMETHOD("XrdCryptoX509Crl::Issuer");
   return (const char *)0;
}

//_____________________________________________________________________________
const char *XrdCryptoX509Crl::IssuerHash()
{
   // Return issuer name
   ABSTRACTMETHOD("XrdCryptoX509Crl::IssuerHash");
   return (const char *)0;
}

//_____________________________________________________________________________
XrdCryptoX509Crldata XrdCryptoX509Crl::Opaque()
{
   // Return underlying certificate in raw format
   ABSTRACTMETHOD("XrdCryptoX509Crl::Opaque");
   return (XrdCryptoX509Crldata)0;
}

//_____________________________________________________________________________
bool XrdCryptoX509Crl::Verify(XrdCryptoX509 *)
{
   // Verify certificate signature with pub key of ref cert
   ABSTRACTMETHOD("XrdCryptoX509Crl::Verify");
   return 0;
}

//_____________________________________________________________________________
bool XrdCryptoX509Crl::IsRevoked(int, int)
{
   // Verify if certificate with specified serial number has been revoked
   ABSTRACTMETHOD("XrdCryptoX509Crl::IsRevoked");
   return 1;
}

//_____________________________________________________________________________
bool XrdCryptoX509Crl::IsRevoked(const char *, int)
{
   // Verify if certificate with specified serial number has been revoked
   ABSTRACTMETHOD("XrdCryptoX509Crl::IsRevoked");
   return 1;
}
