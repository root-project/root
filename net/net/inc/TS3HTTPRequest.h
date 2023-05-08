// @(#)root/net:$Id$
// Author: Fabio Hernandez 30/01/2013
//         based on an initial version by Marcelo Sousa (class THTTPMessage)

/*************************************************************************
 * Copyright (C) 1995-2011, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TS3HTTPRequest
#define ROOT_TS3HTTPRequest

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TS3HTTPRequest                                                       //
//                                                                      //
// An object of this class represents an HTTP request extended to be    //
// compatible with Amazon's S3 protocol.                                //
// Specifically, such a request contains an 'Authorization' header with //
// information used by the S3 server for authenticating this request.   //
// The authentication information is computed based on a pair of access //
// key and secret key which are both provided to the user by the S3     //
// service provider (e.g. Amazon, Google, etc.).                        //
// The secret key is used to compute a signature of selected fields in  //
// the request. The algorithm for computing the signature is documented //
// in:                                                                  //
//                                                                      //
// Google storage:                                                      //
// http://code.google.com/apis/storage/docs/reference/v1/developer-guidev1.html#authentication
//                                                                      //
// Amazon:                                                              //
// http://docs.aws.amazon.com/AmazonS3/latest/dev/S3_Authentication2.html
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TObject.h"

#include "TString.h"



class TS3HTTPRequest : public TObject {

public:

   enum EHTTPVerb { kGET, kPOST, kPUT, kDELETE, kHEAD, kCOPY };
   enum EAuthType { kNoAuth, kAmazon, kGoogle };

private:
   EHTTPVerb fVerb;        // HTTP Verb
   EAuthType fAuthType;    // Authentication type
   TString   fHost;        // Host name
   TString   fBucket;      // Bucket name
   TString   fObjectKey;   // Object key
   TString   fTimeStamp;   // Request time stamp
   TString   fAccessKey;   // Access key (for authentication)
   TString   fSecretKey;   // Secret key (for authentication)
   TString   fSessionToken; // Session token (for authentication)


protected:
   TString HTTPVerbToTString(EHTTPVerb httpVerb) const;
   TString MakeRequestLine(TS3HTTPRequest::EHTTPVerb httpVerb) const;
   TString MakeAuthHeader(TS3HTTPRequest::EHTTPVerb httpVerb) const;
   TString ComputeSignature(TS3HTTPRequest::EHTTPVerb httpVerb) const;
   TString MakeAuthPrefix() const;
   TString MakeHostHeader() const;
   TString MakeDateHeader() const;
   TString MakeTokenHeader() const;
   TS3HTTPRequest& SetTimeStamp();

public:

   TS3HTTPRequest();
   TS3HTTPRequest(EHTTPVerb httpVerb, const TString& host,
                const TString& bucket, const TString& objectKey,
                EAuthType authType, const TString& accessKey,
                const TString& secretKey);
   TS3HTTPRequest(const TS3HTTPRequest& m);
   virtual ~TS3HTTPRequest() {}

   EHTTPVerb       GetHTTPVerb() const { return fVerb; }
   const TString&  GetHost() const { return fHost; }
   const TString&  GetBucket() const { return fBucket; }
   const TString&  GetObjectKey() const { return fObjectKey; }
   const TString&  GetTimeStamp() const { return fTimeStamp; }
   const TString&  GetAccessKey() const { return fAccessKey; }
   const TString&  GetSecretKey() const { return fSecretKey; }
   TString         GetAuthType() const { return fAuthType; }
   TString         GetRequest(TS3HTTPRequest::EHTTPVerb httpVerb, Bool_t appendCRLF=kTRUE);

   TS3HTTPRequest& SetHost(const TString& host);
   TS3HTTPRequest& SetBucket(const TString& bucket);
   TS3HTTPRequest& SetObjectKey(const TString& objectKey);
   TS3HTTPRequest& SetAccessKey(const TString& accessKey);
   TS3HTTPRequest& SetSecretKey(const TString& secretKey);
   TS3HTTPRequest& SetAuthKeys(const TString& accessKey, const TString& secretKey);
   TS3HTTPRequest& SetAuthType(TS3HTTPRequest::EAuthType authType);
   TS3HTTPRequest& SetSessionToken(const TString& token);

   ClassDefOverride(TS3HTTPRequest, 0)  // Create generic HTTP request for Amazon S3 and Google Storage services
};


//////////////////////////////////////////////////////////////////////////
//                                                                      //
//  Inlines                                                             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

inline TS3HTTPRequest& TS3HTTPRequest::SetHost(const TString& host)
{
   fHost = host;
   return *this;
}

inline TS3HTTPRequest& TS3HTTPRequest::SetBucket(const TString& bucket)
{
   fBucket = bucket;
   return *this;
}

inline TS3HTTPRequest& TS3HTTPRequest::SetObjectKey(const TString& objectKey)
{
   fObjectKey = objectKey;
   return *this;
}

inline TS3HTTPRequest& TS3HTTPRequest::SetAuthKeys(const TString& accessKey, const TString& secretKey)
{
   fAccessKey = accessKey;
   fSecretKey = secretKey;
   return *this;
}

inline TS3HTTPRequest& TS3HTTPRequest::SetAuthType(TS3HTTPRequest::EAuthType authType)
{
   fAuthType = authType;
   return *this;
}

inline TS3HTTPRequest& TS3HTTPRequest::SetAccessKey(const TString& accessKey)
{
   fAccessKey = accessKey;
   return *this;
}

inline TS3HTTPRequest& TS3HTTPRequest::SetSecretKey(const TString& secretKey)
{
   fSecretKey = secretKey;
   return *this;
}

inline TS3HTTPRequest& TS3HTTPRequest::SetSessionToken(const TString& token)
{
   fSessionToken = token;
   return *this;
}

#endif
