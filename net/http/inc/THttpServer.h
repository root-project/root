// $Id$
// Author: Sergey Linev   21/12/2013

#ifndef ROOT_THttpServer
#define ROOT_THttpServer

#ifndef ROOT_TObject
#include "TObject.h"
#endif

#ifndef ROOT_TList
#include "TList.h"
#endif

#ifndef ROOT_TMutex
#include "TMutex.h"
#endif

#ifndef ROOT_THttpCallArg
#include "THttpCallArg.h"
#endif


class THttpEngine;
class THttpTimer;
class TRootSniffer;


class THttpServer : public TNamed {

protected:

   TList        fEngines;     //! engines which runs http server
   THttpTimer  *fTimer;       //! timer used to access main thread
   TRootSniffer *fSniffer;    //! sniffer provides access to ROOT objects hierarchy

   Long_t       fMainThrdId;  //! id of the main ROOT process

   TString      fJSROOTSYS;   //! location of local JSROOT files
   TString      fTopName;     //! name of top folder, default - "ROOT"
   TString      fJSROOT;      //! location of external JSROOT files
   TList        fLocations;   //! list of local directories, which could be accessed via server

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

   Bool_t IsAnyEngine() const { return fEngines.GetSize() > 0; }

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

   void SetJSROOT(const char *location);

   void AddLocation(const char *prefix, const char *path);

   void SetDefaultPage(const char *filename);

   void SetDrawPage(const char *filename);

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

   /** Restrict access to specified object */
   void Restrict(const char *path, const char* options);

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
