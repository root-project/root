// $Id$
// Author: Sergey Linev   21/05/2015

/*************************************************************************
 * Copyright (C) 1995-2013, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_THttpCallArg
#define ROOT_THttpCallArg

#include "TObject.h"

#include "TString.h"

#include <condition_variable>

class THttpServer;
class TNamed;

class THttpCallArg : public TObject {

protected:
   friend class THttpServer;

   TString fTopName;  ///<! top item name
   TString fMethod;   ///<! request method like GET or POST
   TString fPathName; ///<! item path
   TString fFileName; ///<! file name
   TString fUserName; ///<! authenticated user name (if any)
   TString fQuery;    ///<! additional arguments

   void *fPostData;        ///<! binary data received with post request
   Long_t fPostDataLength; ///<! length of binary data

   TNamed *fWSHandle; ///<!  web-socket handle, derived from TNamed class
   UInt_t fWSId;      ///<! websocket identifier, used in web-socket related operations

   std::condition_variable fCond; ///<! condition used to wait for processing

   TString fContentType;   ///<! type of content
   TString fRequestHeader; ///<! complete header, provided with request
   TString fHeader;        ///<! response header like ContentEncoding, Cache-Control and so on
   TString fContent;       ///<! text content (if any)
   Int_t fZipping;         ///<! indicate if content should be zipped

   void *fBinData;        ///<! binary data, assigned with http call
   Long_t fBinDataLength; ///<! length of binary data

   Bool_t fNotifyFlag; ///<!  indicate that notification called

   Bool_t IsBinData() const { return fBinData && fBinDataLength > 0; }

   TString AccessHeader(TString &buf, const char *name, const char *value = 0, Bool_t doing_set = kFALSE);

   TString CountHeader(const TString &buf, Int_t number = -1111) const;

public:
   THttpCallArg();
   ~THttpCallArg();

   // these methods used to set http request arguments

   /** set request method kind like GET or POST */
   void SetMethod(const char *method) { fMethod = method; }

   /** set engine-specific top-name */
   void SetTopName(const char *topname) { fTopName = topname; }

   void SetPathAndFileName(const char *fullpath);

   /** set request path name */
   void SetPathName(const char *p) { fPathName = p; }

   /** set request file name */
   void SetFileName(const char *f) { fFileName = f; }

   /** set name of authenticated user */
   void SetUserName(const char *n) { fUserName = n; }

   /** set request query */
   void SetQuery(const char *q) { fQuery = q; }

   void SetPostData(void *data, Long_t length, Bool_t make_copy = kFALSE);

   void SetWSHandle(TNamed *handle);

   TNamed *TakeWSHandle();

   /** set web-socket id */
   void SetWSId(UInt_t id) { fWSId = id; }

   /** get web-socket id */
   UInt_t GetWSId() const { return fWSId; }

   /** set full set of request header */
   void SetRequestHeader(const char *h) { fRequestHeader = h ? h : ""; }

   /** returns number of fields in request header */
   Int_t NumRequestHeader() const { return CountHeader(fRequestHeader).Atoi(); }

   /** returns field name in request header */
   TString GetRequestHeaderName(Int_t number) const { return CountHeader(fRequestHeader, number); }

   /** get named field from request header */
   TString GetRequestHeader(const char *name) { return AccessHeader(fRequestHeader, name); }

   /** returns engine-specific top-name */
   const char *GetTopName() const { return fTopName.Data(); }

   /** returns request method like GET or POST */
   const char *GetMethod() const { return fMethod.Data(); }

   /** returns kTRUE if post method is used */
   Bool_t IsPostMethod() const { return fMethod.CompareTo("POST") == 0; }

   /** return pointer on posted with request data */
   void *GetPostData() const { return fPostData; }

   /** return length of posted with request data */
   Long_t GetPostDataLength() const { return fPostDataLength; }

   /** returns path name from request URL */
   const char *GetPathName() const { return fPathName.Data(); }

   /** returns file name from request URL */
   const char *GetFileName() const { return fFileName.Data(); }

   /** return authenticated user name (0 - when no authentication) */
   const char *GetUserName() const { return fUserName.Length() > 0 ? fUserName.Data() : 0; }

   /** returns request query (string after ? in request URL) */
   const char *GetQuery() const { return fQuery.Data(); }

   // these methods used in THttpServer to set results of request processing

   /** set content type like "text/xml" or "application/json" */
   void SetContentType(const char *typ) { fContentType = typ; }

   /** mark reply as 404 error - page/request not exists or refused */
   void Set404() { SetContentType("_404_"); }

   /** mark reply as postponed - submitting thread will not be inform */
   void SetPostponed() { SetContentType("_postponed_"); }

   /** indicate that http request should response with file content */
   void SetFile(const char *filename = 0)
   {
      SetContentType("_file_");
      if (filename != 0) fContent = filename;
   }

   /** set content type as XML */
   void SetXml() { SetContentType("text/xml"); }

   /** set content type as JSON */
   void SetJson() { SetContentType("application/json"); }

   void AddHeader(const char *name, const char *value);

   /** returns number of fields in header */
   Int_t NumHeader() const { return CountHeader(fHeader).Atoi(); }

   /** returns field name in header */
   TString GetHeaderName(Int_t number) const { return CountHeader(fHeader, number); }

   TString GetHeader(const char *name);

   /** Set Content-Encoding header like gzip */
   void SetEncoding(const char *typ) { AccessHeader(fHeader, "Content-Encoding", typ, kTRUE); }

   /** Set content directly */
   void SetContent(const char *c) { fContent = c; }

   Bool_t CompressWithGzip();

   /** Set kind of content zipping
     * 0 - none
     * 1 - only when supported in request header
     * 2 - if supported and content size bigger than 10K
     * 3 - always */
   void SetZipping(Int_t kind) { fZipping = kind; }

   /** return kind of content zipping */
   Int_t GetZipping() const { return fZipping; }

   /** add extra http header value to the reply */
   void SetExtraHeader(const char *name, const char *value) { AddHeader(name, value); }

   // Fill http header
   void FillHttpHeader(TString &buf, const char *header = 0);

   // these methods used to return results of http request processing

   Bool_t IsContentType(const char *typ) const { return fContentType == typ; }
   Bool_t Is404() const { return IsContentType("_404_"); }
   Bool_t IsFile() const { return IsContentType("_file_"); }
   Bool_t IsPostponed() const { return IsContentType("_postponed_"); }
   const char *GetContentType() const { return fContentType.Data(); }

   void SetBinData(void *data, Long_t length);

   Long_t GetContentLength() const { return IsBinData() ? fBinDataLength : fContent.Length(); }

   const void *GetContent() const { return IsBinData() ? fBinData : fContent.Data(); }

   void NotifyCondition();

   virtual void HttpReplied();

   ClassDef(THttpCallArg, 0) // Arguments for single HTTP call
};

#endif
