// $Id$
// Author: Sergey Linev   20/10/2017

/*************************************************************************
 * Copyright (C) 1995-2017, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_THttpWSHandler
#define ROOT_THttpWSHandler

#include "TNamed.h"
#include "THttpCallArg.h"

#include <vector>
#include <memory>
#include <mutex>

class THttpWSEngine;
class THttpServer;

class THttpWSHandler : public TNamed {

friend class THttpServer;

private:
   Bool_t fSyncMode{kTRUE};  ///<! is handler runs in synchronous mode (default, no multi-threading)
   Bool_t fDisabled{kFALSE}; ///<!  when true, all further operations will be ignored
   Int_t fSendCnt{0};        ///<! counter for completed send operations
   std::mutex fMutex;        ///<!  protect list of engines
   std::vector<std::shared_ptr<THttpWSEngine>> fEngines; ///<!  list of active WS engines (connections)

   std::shared_ptr<THttpWSEngine> FindEngine(UInt_t id, Bool_t book_send = kFALSE);

   Bool_t HandleWS(std::shared_ptr<THttpCallArg> &arg);

   Int_t RunSendingThrd(std::shared_ptr<THttpWSEngine> engine);

   Int_t PerformSend(std::shared_ptr<THttpWSEngine> engine);

   void RemoveEngine(std::shared_ptr<THttpWSEngine> &engine, Bool_t terminate = kFALSE);

   Int_t CompleteSend(std::shared_ptr<THttpWSEngine> &engine);

protected:

   THttpWSHandler(const char *name, const char *title, Bool_t syncmode = kTRUE);

   /// Method called when multi-threaded send operation is completed
   virtual void CompleteWSSend(UInt_t) {}

   /// Method used to accept or reject root_batch_holder.js request
   virtual Bool_t ProcessBatchHolder(std::shared_ptr<THttpCallArg> &) { return kFALSE; }

   /// Method called when default page content is prepared for use
   /// By default no-cache header is provided
   virtual void VerifyDefaultPageContent(std::shared_ptr<THttpCallArg> &arg) { arg->AddNoCacheHeader(); }

public:
   virtual ~THttpWSHandler();

   /// Returns processing mode of WS handler
   /// If sync mode is TRUE (default), all event processing and data sending performed in main thread
   /// All send functions are blocking and must be performed from main thread
   /// If sync mode is false, WS handler can be used from different threads and starts its own sending threads
   Bool_t IsSyncMode() const { return fSyncMode; }

   /// Provides content of default web page for registered web-socket handler
   /// Can be content of HTML page or file name, where content should be taken
   /// For instance, file:/home/user/test.htm or file:$jsrootsys/files/canvas.htm
   /// If not specified, default index.htm page will be shown
   /// Used by the webcanvas
   virtual TString GetDefaultPageContent() { return ""; }

   /// If returns kTRUE, allows to serve files from subdirectories where page content is situated
   virtual Bool_t CanServeFiles() const { return kFALSE; }

   /// Allow processing of WS requests in arbitrary thread
   virtual Bool_t AllowMTProcess() const { return kFALSE; }

   /// Allow send operations in separate threads (when supported by websocket engine)
   virtual Bool_t AllowMTSend() const { return kFALSE; }

   /// Returns true when processing of websockets is disabled, set shortly before handler need to be destroyed
   Bool_t IsDisabled() const { return fDisabled; }

   /// Disable all processing of websockets, normally called shortly before destructor
   void SetDisabled() { fDisabled = kTRUE; }

   /// Return kTRUE if websocket with given ID exists
   Bool_t HasWS(UInt_t wsid) { return !!FindEngine(wsid); }

   /// Returns current number of websocket connections
   Int_t GetNumWS();

   UInt_t GetWS(Int_t num = 0);

   void CloseWS(UInt_t wsid);

   Int_t SendWS(UInt_t wsid, const void *buf, int len);

   Int_t SendHeaderWS(UInt_t wsid, const char *hdr, const void *buf, int len);

   Int_t SendCharStarWS(UInt_t wsid, const char *str);

   virtual Bool_t ProcessWS(THttpCallArg *arg) = 0;

   ClassDefOverride(THttpWSHandler, 0) // abstract class for handling websocket requests
};

#endif
