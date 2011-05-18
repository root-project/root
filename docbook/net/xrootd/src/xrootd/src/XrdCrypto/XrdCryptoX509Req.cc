// $Id$

const char *XrdCryptoX509ReqCVSID = "$Id$";
/******************************************************************************/
/*                                                                            */
/*                  X r d C r y p t o X 5 0 9 R e q. c c                      */
/*                                                                            */
/* (c) 2005 G. Ganis , CERN                                                   */
/*                                                                            */
/******************************************************************************/

/* ************************************************************************** */
/*                                                                            */
/* Abstract interface for X509 certificates requests.                         */
/* Allows to plug-in modules based on different crypto implementation         */
/* (OpenSSL, Botan, ...)                                                      */
/*                                                                            */
/* ************************************************************************** */

#include <XrdCrypto/XrdCryptoX509Req.hh>
#include <XrdCrypto/XrdCryptoTrace.hh>

//_____________________________________________________________________________
void XrdCryptoX509Req::Dump()
{
   // Dump content
   EPNAME("X509Req::Dump");

   PRINT("+++++++++++++++ X509 request dump ++++++++++++++++");
   PRINT("+");
   PRINT("+ Subject: "<<Subject());
   PRINT("+ Subject hash: "<<SubjectHash());
   PRINT("+");
   if (PKI()) {
      PRINT("+ PKI: "<<PKI()->Status());
   } else {
      PRINT("+ PKI: missing");
   }
   PRINT("+");
   PRINT("+++++++++++++++++++++++++++++++++++++++++++++++++");
}

//_____________________________________________________________________________
bool XrdCryptoX509Req::IsValid()
{
   // Check validity
   ABSTRACTMETHOD("XrdCryptoX509Req::IsValid");
   return 0;
}

//_____________________________________________________________________________
const char *XrdCryptoX509Req::Subject()
{
   // Return subject name
   ABSTRACTMETHOD("XrdCryptoX509Req::Subject");
   return (const char *)0;
}

//_____________________________________________________________________________
const char *XrdCryptoX509Req::SubjectHash()
{
   // Return subject name
   ABSTRACTMETHOD("XrdCryptoX509Req::SubjectHash");
   return (const char *)0;
}

//_____________________________________________________________________________
XrdCryptoX509Reqdata XrdCryptoX509Req::Opaque()
{
   // Return underlying certificate in raw format
   ABSTRACTMETHOD("XrdCryptoX509Req::Opaque");
   return (XrdCryptoX509Reqdata)0;
}

//_____________________________________________________________________________
XrdCryptoRSA *XrdCryptoX509Req::PKI()
{
   // Return PKI key of the certificate
   ABSTRACTMETHOD("XrdCryptoX509Req::PKI");
   return (XrdCryptoRSA *)0;
}

//_____________________________________________________________________________
XrdCryptoX509Reqdata XrdCryptoX509Req::GetExtension(const char *)
{
   // Return issuer name
   ABSTRACTMETHOD("XrdCryptoX509Req::GetExtension");
   return (XrdCryptoX509Reqdata)0;
}

//_____________________________________________________________________________
XrdSutBucket *XrdCryptoX509Req::Export()
{
   // EXport in form of bucket
   ABSTRACTMETHOD("XrdCryptoX509Req::Export");
   return (XrdSutBucket *)0;
}

//_____________________________________________________________________________
bool XrdCryptoX509Req::Verify()
{
   // Verify certificate signature with pub key of ref cert
   ABSTRACTMETHOD("XrdCryptoX509Req::Verify");
   return 0;
}
