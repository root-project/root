// @(#)root/net:$Id$
// Author: Marcelo Sousa   23/08/2011

/*************************************************************************
 * Copyright (C) 1995-2011, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_THTTPMessage
#define ROOT_THTTPMessage

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

#ifndef ROOT_TObject
#include "TObject.h"
#endif

#ifndef ROOT_TString
#include "TString.h"
#endif

enum EHTTP_Verb {
   GET,
   POST,
   PUT,
   DELETE,
   HEAD,
   COPY
};

class THTTPMessage : public TObject{

private:
   enum EHTTP_Verb verb;
   TString path;
   TString bucket;
   TString host;
   TString date;
   TString auth_prefix;
   TString access_id;
   TString access_id_key;
   Bool_t  has_range;
   Int_t   init_byte;
   Int_t   final_byte;

   TString signature;

protected:
   TString Sign();

public:
   THTTPMessage(EHTTP_Verb mverb, TString mpath, TString mbucket, TString mhost,
                TString maprefix, TString maid, TString maidkey);
   THTTPMessage(EHTTP_Verb mverb, TString mpath, TString mbucket, TString mhost,
                TString maprefix, TString maid, TString maidkey, Int_t ibyte, Int_t fbyte);
   THTTPMessage() { }
   virtual ~THTTPMessage() { }
   THTTPMessage &operator=(const THTTPMessage& rhs);

   EHTTP_Verb GetHTTPVerb() const { return verb; }
   TString    GetPath() const { return path; }
   TString    GetBucket() const { return bucket; }
   TString    GetHost() const { return host; }
   TString    GetDatime() const { return date; }
   TString    GetAuthPrefix() const { return auth_prefix; }
   TString    GetAccessId() const { return access_id; }
   TString    GetAccessIdKey() const { return access_id_key; }
   Int_t      GetInitByte() const { return init_byte; }
   Int_t      GetFinalByte() const { return final_byte; }
   TString    GetSignature() const { return signature; }

   Bool_t     HasRange() const { return has_range; }

   TString DatimeToTString() const;
   TString HTTPVerbToTString() const;

   TString CreateHead() const;
   TString CreateHost() const;
   TString CreateDate() const;
   TString CreateRange() const;
   TString CreateAuth() const;

   TString GetRequest() const;

   ClassDef(THTTPMessage, 0)  // Create generic HTTP request for Amazon S3 and Google Storage services
};

#endif
