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
   kGET,
   kPOST,
   kPUT,
   kDELETE,
   kHEAD,
   kCOPY
};

class THTTPMessage : public TObject{

private:
   enum EHTTP_Verb fVerb;  //HTTP Verb
   TString fPath;          //Given path to be parsed        
   TString fBucket;        //Bucket associated with the file
   TString fHost;          //Server name
   TString fDate;          //Date
   TString fAuthPrefix;    //Authentication prefix to distinguish between GT and AWS3
   TString fAccessId;      //User id 
   TString fAccessIdKey;   //Secret key
   Bool_t  fHasRange;      //GET request with range
   Long64_t fOffset;       //Offset
   Long64_t *fInitByte;    //Init positions for the range
   Int_t *fLen;            //Range length
   Int_t fNumBuf;          //Number of buffers
   Int_t fCurrentBuf;      //For requests > 8000 we need to generate several requests

   Int_t fLength;          //Request length

   TString fSignature;     //Message signature

protected:
   TString Sign();

public:
   THTTPMessage(EHTTP_Verb mverb, TString mpath, TString mbucket, TString mhost,
                TString maprefix, TString maid, TString maidkey);
   THTTPMessage(EHTTP_Verb mverb, TString mpath, TString mbucket, TString mhost,
                TString maprefix, TString maid, TString maidkey, Long64_t offset, Long64_t *pos, Int_t *len, Int_t nbuf);
   THTTPMessage() { }
   virtual ~THTTPMessage() { }
   THTTPMessage &operator=(const THTTPMessage& rhs);

   EHTTP_Verb GetHTTPVerb() const { return fVerb; }
   TString    GetPath() const { return fPath; }
   TString    GetBucket() const { return fBucket; }
   TString    GetHost() const { return fHost; }
   TString    GetDatime() const { return fDate; }
   TString    GetAuthPrefix() const { return fAuthPrefix; }
   TString    GetAccessId() const { return fAccessId; }
   TString    GetAccessIdKey() const { return fAccessIdKey; }
   Long64_t   GetOffset() const { return fOffset; }
   Long64_t*  GetInitByte() const { return fInitByte; }
   Int_t*     GetRangeLength() const { return fLen; }
   Int_t      GetCurrentBuffer() const { return fCurrentBuf; }
   Int_t      GetNumBuffers() const { return fNumBuf; }
   Int_t      GetLength() const { return fLength; }
   TString    GetSignature() const { return fSignature; }

   Bool_t     HasRange() const { return fHasRange; }

   TString DatimeToTString() const;
   TString HTTPVerbToTString() const;

   TString CreateHead() const;
   TString CreateHost() const;
   TString CreateDate() const;
   TString CreateAuth() const;

   TString GetRequest();

   ClassDef(THTTPMessage, 0)  // Create generic HTTP request for Amazon S3 and Google Storage services
};

#endif
