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
   fLength       = 0;

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
   fLength      = 0;

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
      fLength      = rhs.fLength;
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

   sign += "/" + GetBucket() + GetPath();
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

   time_t now = time(NULL);
   {
      // TODO: remove this block which was introduced to turnaround the
      // time skew problem of UDS
      // now -= 16*60;
   }
   char result[128];
   struct tm date_format;
   strftime(result, sizeof(result), "%a, %d %b %Y %H:%M:%S GMT", gmtime_r(&now, &date_format));
   return TString(result);
}

//______________________________________________________________________________
TString THTTPMessage::CreateHead() const
{
   // Returns the first line of a HTTP request for this object. Note that since
   // we don't use the virtual host syntax which is supported by Amazon, we
   // must include the bucket name in thr resource. For example, we don't use
   // http://mybucket.s3.amazonaws.com/path/to/my/file but instead
   // http://s3.amazonaws.com/mybucket/path/to/my/file so the HTTP request
   // will be of the form "GET /mybucket/path/to/my/file HTTP/1.1"
   // Also note that the path must include the leading '/'.

   return TString::Format("%s /%s%s HTTP/1.1",
      (const char*)HTTPVerbToTString(), (const char*)GetBucket(), (const char*)GetPath());
}

//______________________________________________________________________________
TString THTTPMessage::CreateHost() const
{
   // Returns the 'Host' header to include in the HTTP request.

   return "Host: " + GetHost();
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
      return "Authorization: " + GetAuthPrefix() + " " + GetAccessId() + ":" + GetSignature();
   } else {
      return "x-goog-api-version: 1\r\nAuthorization: " + GetAuthPrefix() + " " +
      GetAccessId() + ":" + GetSignature();
   }
}

//______________________________________________________________________________
TString THTTPMessage::GetRequest(Bool_t appendCRLF)
{
   // Returns the HTTP request

   TString msg = TString::Format("%s\r\n%s\r\n%s\r\n%s\r\n", 
      (const char*)CreateHead(), (const char*)CreateHost(),
      (const char*)CreateDate(), (const char*)CreateAuth());

   if (HasRange()) {
      Int_t n = 0;
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
   if (appendCRLF)
      msg += "\r\n";
   return msg;
}
