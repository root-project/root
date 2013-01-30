// @(#)root/net:$Id$
// Author: Fabio Hernandez   22/01/2013
//         extending an initial version by Marcelo Sousa (class TAS3File)

/*************************************************************************
 * Copyright (C) 1995-2011, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

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
// b) by specifying them when opening each file.                        //
//                                                                      //
// The first method is convenient if all the S3 files you want to       //
// access are hosted by a single provider. The second one is more       //
// flexible as it allows you to specify which credentials to use        //
// on a per-file basis. See the documentation of the constructor of     //
// this class for details on the syntax.                                //
//                                                                      //
// For generating and signing the HTTP request, this class uses         //
// THTTPMessage.                                                        //
//                                                                      //
// For more information on the details of S3 protocol please refer to:  //
// "Amazon Simple Storage Service Developer Guide":                     //
// http://docs.amazonwebservices.com/AmazonS3/latest/dev/Welcome.html   //
//                                                                      //
// "Amazon Simple Storage Service REST API Reference"                   //
//  http://docs.amazonwebservices.com/AmazonS3/latest/API/APIRest.html  //
//////////////////////////////////////////////////////////////////////////

#include "TS3WebFile.h"
#include "TROOT.h"
#include "THTTPMessage.h"
#include "TError.h"
#include "TSystem.h"
#include "TPRegexp.h"
#include "TEnv.h"


ClassImp(TS3WebFile)

const TString TS3WebFile::fgAuthPrefix = "AWS";

//_____________________________________________________________________________
TS3WebFile::TS3WebFile(const char* path, Option_t* options)
           : TWebFile(path, "IO")
{
   // Construct a TS3WebFile object. The path argument is a URL of one of the
   // following forms:
   //
   //         s3://host.example.com/bucket/path/to/my/file
   //     s3http://host.example.com/bucket/path/to/my/file
   //    s3https://host.example.com/bucket/path/to/my/file
   //        as3://host.example.com/bucket/path/to/my/file
   // 
   // The 'as3' schema is accepted for backwards compatibility but its usage is
   // deprecated.
   //
   // The recommended way to create an instance of this class is through
   // TFile::Open, for instance:
   //
   // TFile* f = TFile::Open("s3://host.example.com/bucket/path/to/my/file")
   //
   // The specified schema (i.e. s3, s3http or s3https) determines the underlying
   // protocol to use for downloading the file contents, namely HTTP or HTTPS. The
   // 's3' and 's3https' schemas imply using HTTPS as the transport protocol.
   // The 's3http' and 'as3' schema imply using HTTP as the transport protocol.
   //
   // The 'options' argument can contain 'NOPROXY' if you want to bypass
   // the HTTP proxy when retrieving this file's contents. In addition, you can
   // provide the access key and secret key to be used for authentication purposes
   // for this file by using a string of the form "AUTH=myAccessKey:mySecretkey".
   //
   // If you need to specify both NOPROXY and AUTH separate them by ' '
   // (blank), for instance: 
   // "NOPROXY AUTH=F38XYZABCDeFgH4D0E1F:V+frt4re7J1euSNFnmaf8wwmI4AAAE7kzxZ/TTM+"
   //
   // Examples:
   //    TFile* f1 = TFile::Open("s3://host.example.com/bucket/path/to/my/file",
   //                            "NOPROXY AUTH=F38XYZABCDeFgH4D0E1F:V+frt4re7J1euSNFnmaf8wwmI4AAAE7kzxZ/TTM+");
   //    TFile* f2 = TFile::Open("s3://host.example.com/bucket/path/to/my/file",
   //                            "AUTH=F38XYZABCDeFgH4D0E1F:V+frt4re7J1euSNFnmaf8wwmI4AAAE7kzxZ/TTM+");
   //
   // If there is no authentication information in the 'options' argument
   // (i.e. not AUTH="....") the values of the environmental variables
   // S3_ACCESS_KEY and S3_SECRET_KEY (if set) are expected to contain
   // the access key id and the secret access key, respectively. You have
   // been provided with these credentials by your S3 service provider.

   // Make sure this is a valid S3 path. We accept 'as3' as a schema, for
   // backwards compatibility
   Bool_t doMakeZombie = kFALSE;
   TString errorMsg;
   TPMERegexp rex("^(s3|s3http[s]?|as3){1}://([^/]+)/([^/]+)/([^/].*)", "i");
   if (rex.Match(TString(path)) != 5) {
      errorMsg = TString::Format("invalid S3 path '%s'", path);
      doMakeZombie = kTRUE;
   }
   else if (!ParseOptions(options)) {
      errorMsg = TString::Format("could not parse options '%s'", options);
      doMakeZombie = kTRUE;
   }

   // Should we stop initializing this object?
   if (doMakeZombie) {
      Error("TS3WebFile", "%s", (const char*)errorMsg);
      MakeZombie();
      gDirectory = gROOT;
      return;      
   }

   // Set this S3 object's URL, the bucket name this file is located in
   // and the object key
   fBucket = rex[3]; // bucket name
   fObjectKey.Form("/%s", (const char*)rex[4]); // include '/' as the starting character in object key

   // Initialize super-classes data members (fUrl is a data member of
   // super-super class TFile)
   TString protocol = "https";
   if (rex[1].EqualTo("http", TString::kIgnoreCase) || 
       rex[1].EqualTo("as3", TString::kIgnoreCase))
      protocol = "http";
   fUrl.SetUrl(TString::Format("%s://%s/%s/%s", (const char*)protocol,
                                                (const char*)rex[2], 
                                                (const char*)rex[3],
                                                (const char*)rex[4]));
      
   // Set S3-specific data members. If the access and secret keys are not
   // provided in the 'options' argument we look in the environmental 
   // variables.
   if (fAccessKey.IsNull()) {
      const char* kAccessKeyEnv = "S3_ACCESS_KEY";
      const char* kSecretKeyEnv = "S3_SECRET_KEY";
      if (!SetCredentialsFromEnv(kAccessKeyEnv, kSecretKeyEnv)) {
         Error("TS3WebFile", "could not find authentication info in "\
               "'options' argument and at least one of the environment variables '%s' or '%s' is not set",
               kAccessKeyEnv, kSecretKeyEnv);
         MakeZombie();
         gDirectory = gROOT;
         return;
      }
   }
   
   // Assume this server does not serve multi-range HTTP GET requests. We
   // will detect this when the HTTP headers of this files are retrieved
   // later in the initialization process
   fUseMultiRange = kFALSE;
      
   // Call super-class initializer
   TWebFile::Init(kFALSE);
}


//_____________________________________________________________________________
Bool_t TS3WebFile::ParseOptions(const TString& options)
{
   // Extracts the S3 authentication key pair (access key and secret key)
   // from the options. The authentication credentials can be specified in
   // the options provided to the constructor of this class as a string
   // containing: "AUTH=<access key>:<secret key>" and can include other
   // options, for instance "NOPROXY" for not using the HTTP proxy for
   // accessing this file's contents.
   // For instance:
   // "NOPROXY AUTH=F38XYZABCDeFgH4D0E1F:V+frt4re7J1euSNFnmaf8wwmI4AAAE7kzxZ/TTM+"
   
   if (options.IsNull())
      return kTRUE;
      
   fNoProxy = kFALSE;
   if (options.Contains("NOPROXY", TString::kIgnoreCase))
      fNoProxy = kTRUE;
   CheckProxy();
   
   // Look in the options string for the authentication information.
   TPMERegexp rex("(^AUTH=|^.* AUTH=)([a-z0-9]+):([a-z0-9+/]+)[\\s]*.*$", "i");
   if (rex.Match(options) < 4) {
      Error("ParseOptions", "expecting options of the form \"AUTH=myAccessKey:mySecretKey\"");
      return kFALSE;
   }
   fAccessKey = rex[2];
   fSecretKey = rex[3];
   return kTRUE;
}


//_____________________________________________________________________________
Int_t TS3WebFile::GetHead()
{
   // Overwrites TWebFile::GetHead() for retrieving the HTTP headers of this
   // file. Uses THTTPMessage to generate an HTTP HEAD request which includes
   // the authorization header expected by the S3 server.

   THTTPMessage s3head = THTTPMessage(kHEAD, fObjectKey, fBucket,
                                      fUrl.GetHost(), fgAuthPrefix,
                                      fAccessKey, fSecretKey);
   fMsgGetHead = s3head.GetRequest();
   return TWebFile::GetHead();
}


//_____________________________________________________________________________
void TS3WebFile::SetMsgReadBuffer10(const char* redirectLocation, Bool_t tempRedirect)
{
   // Overwrites TWebFile::SetMsgReadBuffer10() for setting the HTTP GET
   // request compliant to the authentication mechanism used by the S3
   // protocol. The GET request must contain an "Authorization" header with
   // the signature of the request, generated using the user's secret access
   // key.

   TWebFile::SetMsgReadBuffer10(redirectLocation, tempRedirect);
   THTTPMessage s3get = THTTPMessage(kGET, fObjectKey, fBucket,
                                     fUrl.GetHost(), fgAuthPrefix,
                                     fAccessKey, fSecretKey);
   fMsgReadBuffer10 = s3get.GetRequest(kFALSE) + "Range: bytes=";
   return;
}


//_____________________________________________________________________________
Bool_t TS3WebFile::ReadBuffers(char* buf, Long64_t* pos, Int_t* len, Int_t nbuf)
{

   // Overwrites TWebFile::ReadBuffers() for reading specified byte ranges.
   // According to the kind of server this file is hosted by, we use a
   // single HTTP request with a muti-range header or we generate multiple
   // requests with a single range each.
   
   // Does this server support multi-range GET requests?
   if (fUseMultiRange)
      return TWebFile::ReadBuffers(buf, pos, len, nbuf);

   // Send multiple GET requests with a single range of bytes
   // Adapted from original version by Wang Lu
   for (Int_t i=0, offset=0; i < nbuf; i++) {
      THTTPMessage s3get = THTTPMessage(kGET, fObjectKey, fBucket,
                                        fUrl.GetHost(), fgAuthPrefix,
                                        fAccessKey, fSecretKey);
      TString rangeHeader = TString::Format("Range: bytes=%lld-%lld\r\n\r\n", 
                                             pos[i], pos[i] + len[i] - 1);
      TString s3Request = s3get.GetRequest(kFALSE) + rangeHeader;
      if (GetFromWeb10(&buf[offset], len[i], s3Request) == -1)
         return kTRUE;
      offset += len[i];
   }
   return kFALSE;
}


//_____________________________________________________________________________
void TS3WebFile::ProcessHttpHeader(const TString& headerLine)
{
   // This method is called by the super-class TWebFile when a HTTP header
   // for this file is retrieved. We scan the 'Server' header to detect the
   // type of S3 server this file is hosted on and to determine if it is
   // known to support multi-range HTTP GET requests. Some S3 servers (for
   // instance Amazon's) do not support that feature and when they
   // receive a multi-range request they sent back the whole file contents.
   // For this class, if the server do not support multirange requests
   // we issue multiple single-range requests instead.
   
   TPMERegexp rex("^Server: (.+)", "i");
   if (rex.Match(headerLine) != 2)
      return;

   // Extract the identity of this server and compare it to the
   // identify of the servers known to support multi-range requests.
   // The list of server identities is expected to be found in ROOT
   // configuration.
   TString serverId = rex[1].ReplaceAll("\r", "").ReplaceAll("\n", "");
   TString multirangeServers(gEnv->GetValue("TS3WebFile.Root.MultiRangeServer", ""));
   fUseMultiRange = multirangeServers.Contains(serverId, TString::kIgnoreCase) ? kTRUE : kFALSE;
}

//_____________________________________________________________________________
Bool_t TS3WebFile::SetCredentialsFromEnv(const char* accessKeyEnv, const char* secretKeyEnv)
{
   // Sets the access and secret keys from the environmental variables, if
   // they are both set.

   // Look first in the recommended environmental variables. Both variables 
   // must be set.
   TString accessKey = gSystem->Getenv(accessKeyEnv);
   TString secretKey = gSystem->Getenv(secretKeyEnv);
   if (!accessKey.IsNull() && !secretKey.IsNull()) {
      fAccessKey = accessKey;
      fSecretKey = secretKey;
      return kTRUE;
   }

   // Look now in the legacy environmental variables
   accessKey = gSystem->Getenv("S3_ACCESS_ID"); // Legacy access key
   secretKey = gSystem->Getenv("S3_ACCESS_KEY"); // Legacy secret key
   if (!accessKey.IsNull() && !secretKey.IsNull()) {
      Warning("SetAuthKeys", "usage of S3_ACCESS_ID and S3_ACCESS_KEY environmental variables is deprecated.");
      Warning("SetAuthKeys", "please use S3_ACCESS_KEY and S3_SECRET_KEY environmental variables.");
      fAccessKey = accessKey;
      fSecretKey = secretKey;
      return kTRUE;
   }

   return kFALSE;
}

