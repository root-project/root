// $Id$

const char *XrdCryptoX509CVSID = "$Id$";
/******************************************************************************/
/*                                                                            */
/*                       X r d C r y p t o X 5 0 9 . c c                      */
/*                                                                            */
/* (c) 2005 G. Ganis , CERN                                                   */
/*                                                                            */
/******************************************************************************/

/* ************************************************************************** */
/*                                                                            */
/* Abstract interface for X509 certificates.                                  */
/* Allows to plug-in modules based on different crypto implementation         */
/* (OpenSSL, Botan, ...)                                                      */
/*                                                                            */
/* ************************************************************************** */
#include <time.h>

#include <XrdCrypto/XrdCryptoX509.hh>
#include <XrdCrypto/XrdCryptoTrace.hh>

const char *XrdCryptoX509::ctype[4] = { "Unknown", "CA", "EEC", "Proxy" };

#define kAllowedSkew 600

//_____________________________________________________________________________
void XrdCryptoX509::Dump()
{
   // Dump content
   EPNAME("X509::Dump");

   // Time strings
   struct tm tst;
   char stbeg[256] = {0};
   time_t tbeg = NotBefore();
   localtime_r(&tbeg,&tst);
   asctime_r(&tst,stbeg);
   stbeg[strlen(stbeg)-1] = 0;
   char stend[256] = {0};
   time_t tend = NotAfter();
   localtime_r(&tend,&tst);
   asctime_r(&tst,stend);
   stend[strlen(stend)-1] = 0;

   PRINT("+++++++++++++++ X509 dump +++++++++++++++++++++++");
   PRINT("+");
   PRINT("+ File:    "<<ParentFile());
   PRINT("+");
   PRINT("+ Type: "<<Type());
   PRINT("+ Serial Number: "<<SerialNumber());
   PRINT("+ Subject: "<<Subject());
   PRINT("+ Subject hash: "<<SubjectHash());
   PRINT("+ Issuer:  "<<Issuer());
   PRINT("+ Issuer hash:  "<<IssuerHash());
   PRINT("+");
   if (IsExpired()) {
      PRINT("+ Validity: (expired!)");
   } else {
      PRINT("+ Validity:");
   }
   PRINT("+ NotBefore:  "<<tbeg<<" UTC - "<<stbeg);
   PRINT("+ NotAfter:   "<<tend<<" UTC - "<<stend);
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
int XrdCryptoX509::BitStrength()
{
   // Return number of bits in key
   ABSTRACTMETHOD("XrdCryptoX509::BitStrength");
   return -1;
}

//_____________________________________________________________________________
bool XrdCryptoX509::IsValid(int when)
{
   // Check validity at UTC time 'when'. Use when =0 (default) to check
   // at present time.

   int now = (when > 0) ? when : (int)time(0);
   return (now >= (NotBefore()-kAllowedSkew) && now <= NotAfter());
}

//_____________________________________________________________________________
bool XrdCryptoX509::IsExpired(int when)
{
   // Check expiration at UTC time 'when'. Use when =0 (default) to check
   // at present time.

   int now = (when > 0) ? when : (int)time(0);
   return (now > NotAfter());
}

//_____________________________________________________________________________
int XrdCryptoX509::NotBefore()
{
   // Begin-validity time in secs since Epoch
   ABSTRACTMETHOD("XrdCryptoX509::NotBefore");
   return -1;
}

//_____________________________________________________________________________
int XrdCryptoX509::NotAfter()
{
   // End-validity time in secs since Epoch
   ABSTRACTMETHOD("XrdCryptoX509::NotAfter");
   return -1;
}

//_____________________________________________________________________________
const char *XrdCryptoX509::Subject()
{
   // Return subject name
   ABSTRACTMETHOD("XrdCryptoX509::Subject");
   return (const char *)0;
}

//_____________________________________________________________________________
const char *XrdCryptoX509::ParentFile()
{
   // Return parent file name
   ABSTRACTMETHOD("XrdCryptoX509::ParentFile");
   return (const char *)0;
}

//_____________________________________________________________________________
const char *XrdCryptoX509::Issuer()
{
   // Return issuer name
   ABSTRACTMETHOD("XrdCryptoX509::Issuer");
   return (const char *)0;
}

//_____________________________________________________________________________
const char *XrdCryptoX509::SubjectHash()
{
   // Return subject name
   ABSTRACTMETHOD("XrdCryptoX509::SubjectHash");
   return (const char *)0;
}

//_____________________________________________________________________________
const char *XrdCryptoX509::IssuerHash()
{
   // Return issuer name
   ABSTRACTMETHOD("XrdCryptoX509::IssuerHash");
   return (const char *)0;
}

//_____________________________________________________________________________
XrdCryptoX509data XrdCryptoX509::Opaque()
{
   // Return underlying certificate in raw format
   ABSTRACTMETHOD("XrdCryptoX509::Opaque");
   return (XrdCryptoX509data)0;
}

//_____________________________________________________________________________
XrdCryptoRSA *XrdCryptoX509::PKI()
{
   // Return PKI key of the certificate
   ABSTRACTMETHOD("XrdCryptoX509::PKI");
   return (XrdCryptoRSA *)0;
}

//_____________________________________________________________________________
void XrdCryptoX509::SetPKI(XrdCryptoX509data)
{
   // Set PKI

   ABSTRACTMETHOD("XrdCryptoX509::SetPKI");
}

//_____________________________________________________________________________
kXR_int64 XrdCryptoX509::SerialNumber()
{
   // Return issuer name
   ABSTRACTMETHOD("XrdCryptoX509::SerialNumber");
   return -1;
}

//_____________________________________________________________________________
XrdOucString XrdCryptoX509::SerialNumberString()
{
   // Return issuer name
   ABSTRACTMETHOD("XrdCryptoX509::SerialNumberString");
   return XrdOucString("");
}

//_____________________________________________________________________________
XrdCryptoX509data XrdCryptoX509::GetExtension(const char *)
{
   // Return issuer name
   ABSTRACTMETHOD("XrdCryptoX509::GetExtension");
   return (XrdCryptoX509data)0;
}

//_____________________________________________________________________________
XrdSutBucket *XrdCryptoX509::Export()
{
   // EXport in form of bucket
   ABSTRACTMETHOD("XrdCryptoX509::Export");
   return (XrdSutBucket *)0;
}

//_____________________________________________________________________________
bool XrdCryptoX509::Verify(XrdCryptoX509 *)
{
   // Verify certificate signature with pub key of ref cert
   ABSTRACTMETHOD("XrdCryptoX509::Verify");
   return 0;
}
