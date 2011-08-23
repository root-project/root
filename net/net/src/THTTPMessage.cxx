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
#include <openssl/sha.h>
#include <openssl/hmac.h>
#include <openssl/evp.h>
#include <openssl/bio.h>
#include <openssl/buffer.h>
#include <stdio.h>
#include <time.h>
#include <string.h>

ClassImp(THTTPMessage)

//______________________________________________________________________________
THTTPMessage::THTTPMessage(EHTTP_Verb mverb, TString mpath, TString mbucket,
                           TString mhost, TString maprefix, TString maid,
                           TString maidkey)
{
   // THTTPMessage for HTTP requests without the Range attribute.

   verb           = mverb;
   path           = mpath;
   bucket         = mbucket;
   host           = mhost;
   date           = DatimeToTString();
   auth_prefix    = maprefix;
   access_id      = maid;
   access_id_key  = maidkey;
   has_range      = false;
   init_byte      = 0;
   final_byte     = 0;
   signature      = Sign();
}

//______________________________________________________________________________
THTTPMessage::THTTPMessage(EHTTP_Verb mverb, TString mpath, TString mbucket,
                           TString mhost, TString maprefix, TString maid,
                           TString maidkey, Int_t ibyte, Int_t fbyte)
{
   // THTTPMessage for HTTP Get Requests with Range.

   verb           = mverb;
   path           = mpath;
   bucket         = mbucket;
   host           = mhost;
   date           = DatimeToTString();
   auth_prefix    = maprefix;
   access_id      = maid;
   access_id_key  = maidkey;
   has_range      = true;
   init_byte       = ibyte;
   final_byte      = fbyte;
   signature      = Sign();
}

//______________________________________________________________________________
THTTPMessage &THTTPMessage::operator=(const THTTPMessage &rhs)
{
   // Copy ctor.

   if (this != &rhs){
      TObject::operator=(rhs);
      verb           = rhs.verb;
      path           = rhs.path;
      bucket         = rhs.bucket;
      host           = rhs.host;
      date           = rhs.date;
      has_range      = rhs.has_range;
      init_byte      = rhs.init_byte;
      final_byte     = rhs.final_byte;
      auth_prefix    = rhs.auth_prefix;
      access_id      = rhs.access_id;
      access_id_key  = rhs.access_id_key;
      signature      = rhs.signature;
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

   if(GetAuthPrefix() == "GOOG1"){
      sign += "x-goog-api-version:1\n";
   }

   sign += "/"+GetBucket()+GetPath();
   char digest[SHA_DIGEST_LENGTH] = {0};
   unsigned int *sd = NULL;

   TString key = GetAccessIdKey();

   HMAC(EVP_sha1(), key.Data(), key.Length() , (unsigned char *) sign.Data(), sign.Length(), (unsigned char *) digest, sd);

   return TBase64::Encode((const char *) digest, SHA_DIGEST_LENGTH);
}

//______________________________________________________________________________
TString THTTPMessage::HTTPVerbToTString() const
{
   EHTTP_Verb mverb = GetHTTPVerb();
   switch(mverb){
      case GET:    return TString("GET");
      case POST:   return TString("POST");
      case PUT:    return TString("PUT");
      case DELETE: return TString("DELETE");
      case HEAD:   return TString("HEAD");
      case COPY:   return TString("COPY");
      default:     return TString("");
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
   return (bucket.EqualTo("")) ? "Host: "+GetHost() : "Host: "+GetBucket()+"."+GetHost();
}

//______________________________________________________________________________
TString THTTPMessage::CreateDate() const
{
   return "Date: " + GetDatime();
}

//______________________________________________________________________________
TString THTTPMessage::CreateRange() const
{
   TString range;
   range.Form("Range: bytes=%d-%d", GetInitByte(), GetFinalByte());
   return range;
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
TString THTTPMessage::GetRequest() const
{
   // Generates a TString with the HTTP Request.

   TString msg;
   msg  = CreateHead()+"\r\n";
   msg += CreateHost()+"\r\n";
   msg += CreateDate()+"\r\n";
   if(HasRange()) { msg += CreateRange()+"\r\n"; }
   msg += CreateAuth()+"\r\n";
   msg += "\r\n\r\n";;
   return msg;
}
