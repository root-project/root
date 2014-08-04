// @(#)root/net:$Id: TS3WebFile.h$
// Author: Fabio Hernandez   22/01/2013
//         extending an initial version by Marcelo Sousa (class TAS3File)

/*************************************************************************
 * Copyright (C) 1995-2011, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TS3WebFile
#define ROOT_TS3WebFile

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TS3WebFile                                                           //
//                                                                      //
// A TS3WebFile is a TWebFile which retrieves the file contents from a  //
// web server implementing the REST API of the Amazon S3 protocol. This //
// class is meant to be as generic as possible to be used with files    //
// hosted not only by Amazon S3 servers but also by other providers     //
// implementing the core of the S3 protocol.                            //
//                                                                      //
// The S3 protocol works on top of HTTPS (and HTTP) and imposes that    //
// each HTTP request be signed using a specific convention: the request //
// must include an 'Authorization' header which contains the signature  //
// of a concatenation of selected request fields. For signing the       //
// request, an 'Access Key Id' and a 'Secret Access Key' need to be     //
// known. These keys are used by the S3 servers to identify the client  //
// and to authenticate the request as genuine.                          //
//                                                                      //
// As an end user, you must know the Access Key and Secret Access Key   //
// in order to access each S3 file. They are provided to you by your S3 //
// service provider. Those two keys can be provided to ROOT when        //
// initializing an object of this class by two means:                   //
// a) by using the environmental variables S3_ACCESS_KEY and            //
//    S3_SECRET_KEY, or                                                 //
// b) by specifying them as an argument when opening each file.         //
//                                                                      //
// The first method is convenient if all the S3 files you want to       //
// access are hosted by a single provider. The second one is more       //
// flexible as it allows you to specify which credentials to use        //
// on a per-file basis. See the documentation of the constructor of     //
// this class for details on the syntax.                                //
//                                                                      //
// For generating and signing the HTTP request, this class uses         //
// TS3HTTPRequest.                                                      //
//                                                                      //
// For more information on the details of S3 protocol please refer to:  //
// "Amazon Simple Storage Service Developer Guide":                     //
// http://docs.amazonwebservices.com/AmazonS3/latest/dev/Welcome.html   //
//                                                                      //
// "Amazon Simple Storage Service REST API Reference"                   //
//  http://docs.amazonwebservices.com/AmazonS3/latest/API/APIRest.html  //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TWebFile
#include "TWebFile.h"
#endif

#ifndef ROOT_TUrl
#include "TUrl.h"
#endif

#ifndef ROOT_TString
#include "TString.h"
#endif

#ifndef ROOT_TS3HTTPRequest
#include "TS3HTTPRequest.h"
#endif


class TS3WebFile: public TWebFile {

private:
   TS3WebFile();
   Bool_t ParseOptions(Option_t* options, TString& accessKey, TString& secretKey);
   Bool_t GetCredentialsFromEnv(const char* accessKeyEnv, const char* secretKeyEnv,
                                TString& outAccessKey, TString& outSecretKey);

protected:
   // Super-class methods extended by this class
   virtual Int_t GetHead();
   virtual void SetMsgReadBuffer10(const char* redirectLocation = 0, Bool_t tempRedirect = kFALSE);
   virtual void ProcessHttpHeader(const TString& headerLine);

   // Modifiers of data members (to be used mainly by subclasses)
   void SetAccessKey(const TString& accessKey) { fS3Request.SetAccessKey(accessKey); }
   void SetSecretKey(const TString& secretKey) { fS3Request.SetSecretKey(secretKey); }

   // Data members
   TS3HTTPRequest fS3Request;      // S3 HTTP request
   Bool_t         fUseMultiRange;  // Is the S3 server capable of serving multirange requests?

public:
   // Constructors & Destructor
   TS3WebFile(const char* url, Option_t* options="");
   virtual ~TS3WebFile() {}

   // Selectors
   const TString&  GetAccessKey() const { return fS3Request.GetAccessKey(); }
   const TString&  GetSecretKey() const { return fS3Request.GetSecretKey(); }
   const TString&  GetBucket() const { return fS3Request.GetBucket(); }
   const TString&  GetObjectKey() const { return fS3Request.GetObjectKey(); }
   const TUrl&     GetUrl() const { return fUrl; }

   // Modifiers
   virtual Bool_t ReadBuffers(char* buf, Long64_t* pos, Int_t* len, Int_t nbuf);

   ClassDef(TS3WebFile, 0)  // Read a ROOT file from a S3 server
};

#endif // ROOT_TS3WebFile
