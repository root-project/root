// $Id$
// Author: Sergey Linev   21/12/2013

#ifndef ROOT_THttpServer
#define ROOT_THttpServer

#ifndef ROOT_TNamed
#include "TNamed.h"
#endif

#ifndef ROOT_TList
#include "TList.h"
#endif

#ifndef ROOT_TMutex
#include "TMutex.h"
#endif

#ifndef ROOT_TCondition
#include "TCondition.h"
#endif


// this class is used to deliver http request arguments to main process
// and also to return back results of processing


class THttpEngine;
class THttpTimer;
class TRootSniffer;
class THttpServer;


class THttpCallArg : public TObject {

protected:
   friend class THttpServer;

   TString fTopName;            //! top item name
   TString fPathName;           //! item path
   TString fFileName;           //! file name
   TString fQuery;              //! additional arguments

   TCondition fCond;            //! condition used to wait for processing

   TString fContentType;        //! type of content
   TString fHeader;             //! response header like ContentEncoding, Cache-Control and so on
   TString fContent;            //! text content (if any)
   Int_t   fZipping;            //! indicate if content should be zipped

   void *fBinData;              //! binary data, assigned with http call
   Long_t fBinDataLength;       //! length of binary data

   Bool_t IsBinData() const
   {
      return fBinData && fBinDataLength > 0;
   }

public:

   THttpCallArg();
   ~THttpCallArg();

   // these methods used to set http request arguments

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

   void SetQuery(const char *q)
   {
      // set request query

      fQuery = q;
   }

   const char *GetTopName() const
   {
      // returns engine-specific top-name

      return fTopName.Data();
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

   void AddHeader(const char *name, const char *value)
   {
      // Add name:value pair to reply header
      // Same header can be specified only once

      fHeader.Append(TString::Format("%s: %s\r\n", name, value));
   }

   void SetEncoding(const char *typ)
   {
      // Set Content-Encoding header like gzip

      AddHeader("Content-Encoding", typ);
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

// ______________________________________________________________________

class THttpServer : public TNamed {

protected:

   TList        fEngines;     //! engines which runs http server
   THttpTimer  *fTimer;       //! timer used to access main thread
   TRootSniffer *fSniffer;    //! sniffer provides access to ROOT objects hierarchy

   Long_t       fMainThrdId;  //! id of the main ROOT process

   TString      fJSROOTSYS;   //! location of local JSROOT files
   TString      fROOTSYS;     //! location of ROOT files
   TString      fTopName;     //! name of top folder, default - "ROOT"
   TString      fJSROOT;      //! location of external JSROOT files

   TString      fDefaultPage; //! file name for default page name
   TString      fDefaultPageCont; //! content of the file content
   TString      fDrawPage;    //! file name for drawing of single element
   TString      fDrawPageCont; //! content of draw page

   TMutex       fMutex;       //! mutex to protect list with arguments
   TList        fCallArgs;    //! submitted arguments

   // Here any request can be processed
   virtual void ProcessRequest(THttpCallArg *arg);

   static Bool_t VerifyFilePath(const char *fname);

public:

   THttpServer(const char *engine = "civetweb:8080");
   virtual ~THttpServer();

   Bool_t CreateEngine(const char *engine);

   TRootSniffer *GetSniffer() const
   {
      // returns pointer on objects sniffer

      return fSniffer;
   }

   void SetSniffer(TRootSniffer *sniff);

   Bool_t IsReadOnly() const;

   void SetReadOnly(Bool_t readonly);

   void SetTopName(const char *top)
   {
      // set name of top item in objects hierarchy
      fTopName = top;
   }

   const char *GetTopName() const
   {
      // returns name of top item in objects hierarchy
      return fTopName.Data();
   }

   void SetJSROOT(const char* location);

   void SetDefaultPage(const char* filename);

   void SetDrawPage(const char* filename);

   void SetTimer(Long_t milliSec = 100, Bool_t mode = kTRUE);

   /** Check if file is requested, thread safe */
   Bool_t  IsFileRequested(const char *uri, TString &res) const;

   /** Execute HTTP request */
   Bool_t ExecuteHttp(THttpCallArg *arg);

   /** Process submitted requests, must be called from main thread */
   void ProcessRequests();

   /** Register object in subfolder */
   Bool_t Register(const char *subfolder, TObject *obj);

   /** Unregister object */
   Bool_t Unregister(TObject *obj);

   Bool_t RegisterCommand(const char *cmdname, const char *method, const char *icon = 0);

   Bool_t Hide(const char *fullname, Bool_t hide = kTRUE);

   Bool_t SetIcon(const char *fullname, const char *iconname);

   Bool_t CreateItem(const char *fullname, const char *title);

   Bool_t SetItemField(const char *fullname, const char *name, const char *value);

   const char *GetItemField(const char *fullname, const char *name);

   /** Guess mime type base on file extension */
   static const char *GetMimeType(const char *path);

   /** Reads content of file from the disk */
   static char *ReadFileContent(const char *filename, Int_t &len);

   ClassDef(THttpServer, 0) // HTTP server for ROOT analysis
};

#endif
