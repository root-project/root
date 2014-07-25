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
   TString fContentEncoding;    //! type of content encoding
   TString fExtraHeader;        //! extra line which could be append to http response
   TString fContent;            //! text content (if any)

   void *fBinData;              //! binary data, assigned with http call
   Long_t fBinDataLength;       //! length of binary data

   Bool_t IsBinData() const
   {
      return fBinData && fBinDataLength > 0;
   }

   void SetBinData(void* data, Long_t length);

public:

   THttpCallArg();
   ~THttpCallArg();

   // these methods used to set http request arguments

   void SetTopName(const char *topname)
   {
      fTopName = topname;
   }
   void SetPathAndFileName(const char *fullpath);
   void SetPathName(const char *p)
   {
      fPathName = p;
   }
   void SetFileName(const char *f)
   {
      fFileName = f;
   }
   void SetQuery(const char *q)
   {
      fQuery = q;
   }

   const char *GetTopName() const
   {
      return fTopName.Data();
   }
   const char *GetPathName() const
   {
      return fPathName.Data();
   }
   const char *GetFileName() const
   {
      return fFileName.Data();
   }
   const char *GetQuery() const
   {
      return fQuery.Data();
   }

   // these methods used in THttpServer to set results of request processing

   void SetContentType(const char *typ)
   {
      fContentType = typ;
   }
   void Set404()
   {
      SetContentType("_404_");
   }
   void SetFile()
   {
      SetContentType("_file_");
   }
   void SetXml()
   {
      SetContentType("text/xml");
   }
   void SetJson()
   {
      SetContentType("application/json");
   }

   // Set encoding like gzip
   void SetEncoding(const char *typ)
   {
      fContentEncoding = typ;
   }

   void SetExtraHeader(const char* name, const char* value)
   {
      if ((name!=0) && (value!=0))
         fExtraHeader.Form("%s: %s", name, value);
      else
         fExtraHeader.Clear();
   }

   // Fill http header
   void FillHttpHeader(TString &buf, const char* header = 0);

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


class THttpServer : public TNamed {

protected:

   TList        fEngines;     //! engines which runs http server
   THttpTimer  *fTimer;       //! timer used to access main thread
   TRootSniffer *fSniffer;    //! sniffer provides access to ROOT objects hierarchy

   Long_t       fMainThrdId;  //! id of the main ROOT process

   TString      fHttpSys;     //! location of http plugin, need to read special files
   TString      fRootSys;     //! location of ROOT (if any)
   TString      fJSRootIOSys; //! location of JSRootIO (if any)
   TString      fTopName;     //! name of top folder, default - "ROOT"

   TString      fDefaultPage; //! file name for default page name
   TString      fDrawPage;    //! file name for drawing of single element

   TMutex       fMutex;       //! mutex to protect list with arguments
   TList        fCallArgs;    //! submitted arguments

   // Here any request can be processed
   void ProcessRequest(THttpCallArg *arg);

public:

   THttpServer(const char *engine = "civetweb:8080");
   virtual ~THttpServer();

   Bool_t CreateEngine(const char *engine);

   TRootSniffer *GetSniffer() const
   {
      return fSniffer;
   }

   void SetSniffer(TRootSniffer *sniff);

   void SetTopName(const char *top)
   {
      fTopName = top;
   }
   const char *GetTopName() const
   {
      return fTopName.Data();
   }

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

   /** Guess mime type base on file extension */
   static const char *GetMimeType(const char *path);

   ClassDef(THttpServer, 0) // HTTP server for ROOT analysis
};

#endif
