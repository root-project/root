// $Id$
// Author: Sergey Linev   21/05/2015

#ifndef ROOT_THttpCallArg
#define ROOT_THttpCallArg

#ifndef ROOT_TObject
#include "TObject.h"
#endif

#ifndef ROOT_TString
#include "TString.h"
#endif

#ifndef ROOT_TCondition
#include "TCondition.h"
#endif


class THttpServer;

class THttpCallArg : public TObject {

protected:
   friend class THttpServer;

   TString fTopName;            //! top item name
   TString fMethod;             //! request method like GET or POST
   TString fPathName;           //! item path
   TString fFileName;           //! file name
   TString fUserName;           //! authenticated user name (if any)
   TString fQuery;              //! additional arguments

   void *fPostData;              //! binary data received with post request
   Long_t fPostDataLength;       //! length of binary data

   TCondition fCond;            //! condition used to wait for processing

   TString fContentType;        //! type of content
   TString fRequestHeader;      //! complete header, provided with request
   TString fHeader;             //! response header like ContentEncoding, Cache-Control and so on
   TString fContent;            //! text content (if any)
   Int_t   fZipping;            //! indicate if content should be zipped

   void *fBinData;              //! binary data, assigned with http call
   Long_t fBinDataLength;       //! length of binary data

   Bool_t IsBinData() const
   {
      return fBinData && fBinDataLength > 0;
   }

   TString AccessHeader(TString& buf, const char* name, const char* value = 0, Bool_t doing_set = kFALSE);

   TString CountHeader(const TString& buf, Int_t number = -1111) const;

public:

   THttpCallArg();
   ~THttpCallArg();

   // these methods used to set http request arguments

   void SetMethod(const char *method)
   {
      // set request method kind like GET or POST

      fMethod = method;
   }

   void SetTopName(const char *topname)
   {
      // set engine-specific top-name

      fTopName = topname;
   }

   void SetPathAndFileName(const char *fullpath);

   void SetPathName(const char *p)
   {
      // set request path name

      fPathName = p;
   }

   void SetFileName(const char *f)
   {
      // set request file name

      fFileName = f;
   }

   void SetUserName(const char *n)
   {
      // set name of authenticated user

      fUserName = n;
   }

   void SetQuery(const char *q)
   {
      // set request query

      fQuery = q;
   }

   void SetPostData(void *data, Long_t length);

   void SetRequestHeader(const char* h)
   {
      // set full set of request header

      fRequestHeader = h ? h : "";
   }

   Int_t NumRequestHeader() const
   {
      // returns number of fields in request header

      return CountHeader(fRequestHeader).Atoi();
   }

   TString GetRequestHeaderName(Int_t number) const
   {
      // returns field name in request header

      return CountHeader(fRequestHeader, number);
   }

   TString GetRequestHeader(const char* name)
   {
      // get named field from request header

      return AccessHeader(fRequestHeader, name);
   }

   const char *GetTopName() const
   {
      // returns engine-specific top-name

      return fTopName.Data();
   }

   const char *GetMethod() const
   {
      // returns request method like GET or POST

      return fMethod.Data();
   }

   Bool_t IsPostMethod() const
   {
      // returns kTRUE if post method is used

      return fMethod.CompareTo("POST")==0;
   }

   void* GetPostData() const
   {
      // return pointer on posted with request data

      return fPostData;
   }

   Long_t GetPostDataLength() const
   {
      // return length of posted with request data

      return fPostDataLength;
   }

   const char *GetPathName() const
   {
      // returns path name from request URL

      return fPathName.Data();
   }

   const char *GetFileName() const
   {
      // returns file name from request URL

      return fFileName.Data();
   }

   const char *GetUserName() const
   {
      // return authenticated user name (0 - when no authentication)

      return fUserName.Length() > 0 ? fUserName.Data() : 0;
   }

   const char *GetQuery() const
   {
      // returns request query (string after ? in request URL)

      return fQuery.Data();
   }

   // these methods used in THttpServer to set results of request processing

   void SetContentType(const char *typ)
   {
      // set content type like "text/xml" or "application/json"

      fContentType = typ;
   }

   void Set404()
   {
      // mark reply as 404 error - page/request not exists

      SetContentType("_404_");
   }

   void SetFile(const char *filename = 0)
   {
      // indicate that http request should response with file content

      SetContentType("_file_");
      if (filename != 0) fContent = filename;
   }

   void SetXml()
   {
      // set content type as JSON

      SetContentType("text/xml");
   }

   void SetJson()
   {
      // set content type as JSON

      SetContentType("application/json");
   }

   void AddHeader(const char *name, const char *value);

   Int_t NumHeader() const
   {
      // returns number of fields in header

      return CountHeader(fHeader).Atoi();
   }

   TString GetHeaderName(Int_t number) const
   {
      // returns field name in header

      return CountHeader(fHeader, number);
   }

   TString GetHeader(const char* name);

   void SetEncoding(const char *typ)
   {
      // Set Content-Encoding header like gzip

      AccessHeader(fHeader, "Content-Encoding", typ, kTRUE);
   }

   void SetContent(const char *c)
   {
      // Set content directly

      fContent = c;
   }

   Bool_t CompressWithGzip();

   void SetZipping(Int_t kind)
   {
      // Set kind of content zipping
      // 0 - none
      // 1 - only when supported in request header
      // 2 - if supported and content size bigger than 10K
      // 3 - always

      fZipping = kind;
   }

   Int_t GetZipping() const
   {
      // return kind of content zipping

      return fZipping;
   }

   void SetExtraHeader(const char *name, const char *value)
   {
      AddHeader(name, value);
   }

   // Fill http header
   void FillHttpHeader(TString &buf, const char *header = 0);

   // these methods used to return results of http request processing

   Bool_t IsContentType(const char *typ) const
   {
      return fContentType == typ;
   }

   Bool_t Is404() const
   {
      return IsContentType("_404_");
   }

   Bool_t IsFile() const
   {
      return IsContentType("_file_");
   }

   const char *GetContentType() const
   {
      return fContentType.Data();
   }

   void SetBinData(void *data, Long_t length);

   Long_t GetContentLength() const
   {
      return IsBinData() ? fBinDataLength : fContent.Length();
   }

   const void *GetContent() const
   {
      return IsBinData() ? fBinData : fContent.Data();
   }

   ClassDef(THttpCallArg, 0) // Arguments for single HTTP call
};

#endif
