// @(#)root/net:$Id$
// Author: Marcelo Sousa   23/08/2011

/*************************************************************************
 * Copyright (C) 1995-2011, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// THTTPMessage                                                         //
//                                                                      //
// A THTTPMessage object represents a generic HTTP request for the      //
// Amazon S3 and the Google Storage services. It can easily be extended //
// to other API's. It assumes that each request is signed with the      //
// client id and an encripted key, Base64(HMAC + SHA1 (HTTP Request))   //
// which is based on a secret key provided in the constructor.          //
// For more information about the authentication :                      //
// Google Storage:                                                      //
//   http://code.google.com/apis/storage/docs/reference/v1/developer-guidev1.html#authentication //
// Amazon S3:                                                           //
//   http://awsdocs.s3.amazonaws.com/S3/latest/s3-qrc.pdf               //
// At the moment THTTPMessage is used for derived classes of TWebFile   //
// (read only) files supporting HEAD and GET requests.                  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "THTTPMessage.h"
#include "TBase64.h"
#if defined(MAC_OS_X_VERSION_10_7)
#include <CommonCrypto/CommonHMAC.h>
#define SHA_DIGEST_LENGTH 20
#else
#include <openssl/sha.h>
#include <openssl/hmac.h>
#include <openssl/evp.h>
#include <openssl/bio.h>
#include <openssl/buffer.h>
#endif

#include <stdio.h>
#include <time.h>
#include <string.h>

ClassImp(THTTPMessage)

//______________________________________________________________________________
THTTPMessage::THTTPMessage(EHTTP_Verb mverb, TString mpath, TString mbucket, TString mhost,
             TString maprefix, TString maid, TString maidkey)
{
   // THTTPMessage for HTTP requests without the Range attribute.

   fVerb         = mverb;
   fPath         = mpath;
   fBucket       = mbucket;
   fHost         = mhost;
   fDate         = DatimeToTString();
   fAuthPrefix   = maprefix;
   fAccessId     = maid;
   fAccessIdKey  = maidkey;
   fHasRange     = kFALSE;
   fInitByte     = 0;
   fOffset       = 0;  
   fLen          = 0;   
   fNumBuf       = 0;   
   fCurrentBuf   = 0;   
                        
   fSignature    = Sign();
}                

//______________________________________________________________________________
THTTPMessage::THTTPMessage(EHTTP_Verb mverb, TString mpath, TString mbucket, TString mhost,
                           TString maprefix, TString maid, TString maidkey, Long64_t offset, 
                           Long64_t *pos, Int_t *len, Int_t nbuf)
{
   // THTTPMessage for HTTP Get Requests with Range.

   fVerb        = mverb;
   fPath        = mpath;
   fBucket      = mbucket;
   fHost        = mhost;
   fDate        = DatimeToTString();
   fAuthPrefix  = maprefix;
   fAccessId    = maid;
   fAccessIdKey = maidkey;
   fHasRange    = kTRUE;
   fInitByte    = pos;
   fOffset      = offset;
   fLen         = len;
   fNumBuf      = nbuf;
   fCurrentBuf  = 0;

   fSignature   = Sign();
}

//______________________________________________________________________________
THTTPMessage &THTTPMessage::operator=(const THTTPMessage &rhs)
{
   // Copy ctor.

   if (this != &rhs){
      TObject::operator=(rhs);
      fVerb        = rhs.fVerb;
      fPath        = rhs.fPath;
      fBucket      = rhs.fBucket;
      fHost        = rhs.fHost;
      fDate        = rhs.fDate;
      fHasRange    = rhs.fHasRange;
      fInitByte    = rhs.fInitByte;
      fOffset      = rhs.fOffset;
      fLen         = rhs.fLen;
      fNumBuf      = rhs.fNumBuf;
      fCurrentBuf  = rhs.fCurrentBuf;
      fAuthPrefix  = rhs.fAuthPrefix;
      fAccessId    = rhs.fAccessId;
      fAccessIdKey = rhs.fAccessIdKey;
      fSignature   = rhs.fSignature;
   }
   return *this;
}

//______________________________________________________________________________
TString THTTPMessage::Sign()
{
   // Message Signature according to:
   //    http://awsdocs.s3.amazonaws.com/S3/latest/s3-qrc.pdf
   // and
   //    http://code.google.com/apis/storage/docs/reference/v1/developer-guidev1.html#authentication

   TString sign;
   sign += HTTPVerbToTString() + "\n";
   sign += "\n"; // GetContentMD5()
   sign += "\n"; // GetContentType()
   sign += DatimeToTString() + "\n";

   if (GetAuthPrefix() == "GOOG1") {
      sign += "x-goog-api-version:1\n";
   }

   sign += "/"+GetBucket()+GetPath();
   char digest[SHA_DIGEST_LENGTH] = {0};
   TString key = GetAccessIdKey();

#if defined(MAC_OS_X_VERSION_10_7)
   CCHmac(kCCHmacAlgSHA1, key.Data(), key.Length() , (unsigned char *) sign.Data(), sign.Length(), (unsigned char *) digest);
#else
   unsigned int *sd = NULL;
   HMAC(EVP_sha1(), key.Data(), key.Length() , (unsigned char *) sign.Data(), sign.Length(), (unsigned char *) digest, sd);
#endif

   return TBase64::Encode((const char *) digest, SHA_DIGEST_LENGTH);
}

//______________________________________________________________________________
TString THTTPMessage::HTTPVerbToTString() const
{
   EHTTP_Verb mverb = GetHTTPVerb();
   switch(mverb){
      case kGET:    return TString("GET");
      case kPOST:   return TString("POST");
      case kPUT:    return TString("PUT");
      case kDELETE: return TString("DELETE");
      case kHEAD:   return TString("HEAD");
      case kCOPY:   return TString("COPY");
      default:      return TString("");
   }
}

//______________________________________________________________________________
TString THTTPMessage::DatimeToTString() const
{
   // Generates a Date TString according to:
   //   http://code.google.com/apis/storage/docs/reference-headers.html#date

   time_t date_temp;
   struct tm *date_format;
   char date_out[128];

   time(&date_temp);
   date_format = gmtime(&date_temp);
   strftime(date_out, 128, "%a, %d %b %Y %H:%M:%S GMT", date_format);

   return TString(date_out);
}

//______________________________________________________________________________
TString THTTPMessage::CreateHead() const
{
   return HTTPVerbToTString() + " " + GetPath() + " HTTP/1.1";
}

//______________________________________________________________________________
TString THTTPMessage::CreateHost() const
{
   return (fBucket.EqualTo("")) ? "Host: "+GetHost() : "Host: "+GetBucket()+"."+GetHost();
}

//______________________________________________________________________________
TString THTTPMessage::CreateDate() const
{
   return "Date: " + GetDatime();
}

//______________________________________________________________________________
TString THTTPMessage::CreateAuth() const
{
   if (GetAuthPrefix() == "AWS") {
      return "Authorization: " + GetAuthPrefix() + " " + GetAccessId()+":"+GetSignature();
   } else {
      return "x-goog-api-version: 1\r\nAuthorization: " + GetAuthPrefix() + " " +
      GetAccessId() + ":" + GetSignature();
   }
}

//______________________________________________________________________________
TString THTTPMessage::GetRequest()
{
   // Generates a TString with the HTTP Request.

   TString msg;
   msg  = CreateHead()+"\r\n";
   msg += CreateHost()+"\r\n";
   msg += CreateDate()+"\r\n";

   Int_t n = 0;
   if(HasRange()){
      msg += "Range: bytes=";
      for (Int_t i = 0; i < fNumBuf; i++) {
         if (n) msg += ",";
         msg += fInitByte[i] + fOffset;
         msg += "-";
         msg += fInitByte[i] + fOffset + fLen[i] - 1;
         fLength += fLen[i];
         n += fLen[i];
         fCurrentBuf++;
         if (msg.Length() > 8000) {
            break;      
         }
      }
      msg += "\r\n";   
   }

   msg += CreateAuth()+"\r\n";
   msg += "\r\n\r\n";;
   return msg;
}
