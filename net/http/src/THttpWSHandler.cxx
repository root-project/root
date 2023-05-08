// $Id$
// Author: Sergey Linev   20/10/2017

/*************************************************************************
 * Copyright (C) 1995-2013, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "THttpWSHandler.h"

#include "THttpWSEngine.h"
#include "THttpCallArg.h"
#include "TSystem.h"

#include <thread>
#include <chrono>

/** \class THttpWSHandler
\ingroup http

Class for user-side handling of websocket with THttpServer

Approximate how-to:

1. Create derived from  THttpWSHandler class and implement
   ProcessWS() method, where all web sockets request handled.

2. Register instance of derived class to running THttpServer

       TUserWSHandler *handler = new TUserWSHandler("name1","title");
       THttpServer *server = new THttpServer("http:8090");
       server->Register("/subfolder", handler)

3. Now server can accept web socket connection from outside.
   For instance, from JavaScirpt one can connect to it with code:

       let ws = new WebSocket("ws://hostname:8090/subfolder/name1/root.websocket");

4. In the ProcessWS(THttpCallArg *arg) method following code should be implemented:

       if (arg->IsMethod("WS_CONNECT")) {
          return true;  // to accept incoming request
       }

       if (arg->IsMethod("WS_READY")) {
           fWSId = arg->GetWSId(); // fWSId should be member of the user class
           return true; // connection established
       }

       if (arg->IsMethod("WS_CLOSE")) {
            fWSId = 0;
            return true; // confirm close of socket
        }

       if (arg->IsMethod("WS_DATA")) {
            std::string str((const char *)arg->GetPostData(), arg->GetPostDataLength());
            std::cout << "got string " << str << std::endl;
            SendCharStarWS(fWSId, "our reply");
            return true;
        }

5. See in `$ROOTSYS/tutorials/http/ws.C` and `$ROOTSYS/tutorials/http/ws.htm` functional example
*/


ClassImp(THttpWSHandler);

////////////////////////////////////////////////////////////////////////////////
/// normal constructor

THttpWSHandler::THttpWSHandler(const char *name, const char *title, Bool_t syncmode) : TNamed(name, title), fSyncMode(syncmode)
{
}

////////////////////////////////////////////////////////////////////////////////
/// destructor
/// Make sure that all sending threads are stopped correctly

THttpWSHandler::~THttpWSHandler()
{
   SetDisabled();

   std::vector<std::shared_ptr<THttpWSEngine>> clr;

   {
      std::lock_guard<std::mutex> grd(fMutex);
      std::swap(clr, fEngines);
   }

   for (auto &eng : clr) {
      eng->fDisabled = true;
      if (eng->fHasSendThrd) {
         eng->fHasSendThrd = false;
         if (eng->fWaiting)
            eng->fCond.notify_all();
         eng->fSendThrd.join();
      }
      eng->ClearHandle(kTRUE); // terminate connection before starting destructor
   }
}

/// Returns current number of websocket connections
Int_t THttpWSHandler::GetNumWS()
{
   std::lock_guard<std::mutex> grd(fMutex);
   return fEngines.size();
}

////////////////////////////////////////////////////////////////////////////////
/// Return websocket id with given sequential number
/// Number of websockets returned with GetNumWS() method

UInt_t THttpWSHandler::GetWS(Int_t num)
{
   std::lock_guard<std::mutex> grd(fMutex);
   auto iter = fEngines.begin() + num;
   return (*iter)->GetId();
}

////////////////////////////////////////////////////////////////////////////////
/// Find websocket connection handle with given id
/// If book_send parameter specified, have to book send operation under the mutex

std::shared_ptr<THttpWSEngine> THttpWSHandler::FindEngine(UInt_t wsid, Bool_t book_send)
{
   if (IsDisabled())
      return nullptr;

   std::lock_guard<std::mutex> grd(fMutex);

   for (auto &eng : fEngines)
      if (eng->GetId() == wsid) {

         // not allow to work with disabled engine
         if (eng->fDisabled)
            return nullptr;

         if (book_send) {
            if (eng->fMTSend) {
               Error("FindEngine", "Try to book next send operation before previous completed");
               return nullptr;
            }
            eng->fMTSend = kTRUE;
         }
         return eng;
      }

   return nullptr;
}

////////////////////////////////////////////////////////////////////////////////
/// Remove and destroy WS connection

void THttpWSHandler::RemoveEngine(std::shared_ptr<THttpWSEngine> &engine, Bool_t terminate)
{
   if (!engine) return;

   {
      std::lock_guard<std::mutex> grd(fMutex);

      for (auto iter = fEngines.begin(); iter != fEngines.end(); iter++)
         if (*iter == engine) {
            if (engine->fMTSend)
               Error("RemoveEngine", "Trying to remove WS engine during send operation");

            engine->fDisabled = true;
            fEngines.erase(iter);
            break;
         }
   }

   engine->ClearHandle(terminate);

   if (engine->fHasSendThrd) {
      engine->fHasSendThrd = false;
      if (engine->fWaiting)
         engine->fCond.notify_all();
      engine->fSendThrd.join();
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Process request to websocket
/// Different kind of requests coded into THttpCallArg::Method:
///
///     "WS_CONNECT" - connection request
///     "WS_READY" - connection ready
///     "WS_CLOSE" - connection closed
///
/// All other are normal data, which are delivered to users

Bool_t THttpWSHandler::HandleWS(std::shared_ptr<THttpCallArg> &arg)
{
   if (IsDisabled())
      return kFALSE;

   if (!arg->GetWSId())
      return ProcessWS(arg.get());

   // normally here one accept or reject connection requests
   if (arg->IsMethod("WS_CONNECT"))
      return ProcessWS(arg.get());

   auto engine = FindEngine(arg->GetWSId());

   if (arg->IsMethod("WS_READY")) {

      if (engine) {
         Error("HandleWS", "WS engine with similar id exists %u", arg->GetWSId());
         RemoveEngine(engine, kTRUE);
      }

      engine = arg->TakeWSEngine();
      {
         std::lock_guard<std::mutex> grd(fMutex);
         fEngines.emplace_back(engine);
      }

      if (!ProcessWS(arg.get())) {
         // if connection refused, remove engine again
         RemoveEngine(engine, kTRUE);
         return kFALSE;
      }

      return kTRUE;
   }

   if (arg->IsMethod("WS_CLOSE")) {
      // connection is closed, one can remove handle

      RemoveEngine(engine);

      return ProcessWS(arg.get());
   }

   if (engine && engine->PreProcess(arg)) {
      PerformSend(engine);
      return kTRUE;
   }

   Bool_t res = ProcessWS(arg.get());

   if (engine)
      engine->PostProcess(arg);

   return res;
}

////////////////////////////////////////////////////////////////////////////////
/// Close connection with given websocket id

void THttpWSHandler::CloseWS(UInt_t wsid)
{
   auto engine = FindEngine(wsid);

   RemoveEngine(engine, kTRUE);
}

////////////////////////////////////////////////////////////////////////////////
/// Send data stored in the buffer. Returns:
///
/// * 0 - when operation was executed immediately
/// * 1 - when send operation will be performed in different thread

Int_t THttpWSHandler::RunSendingThrd(std::shared_ptr<THttpWSEngine> engine)
{
   if (IsSyncMode() || !engine->SupportSendThrd()) {
      // this is case of longpoll engine, no extra thread is required for it
      if (engine->CanSendDirectly())
         return PerformSend(engine);

      // handling will be performed in following http request handler

      if (!IsSyncMode()) return 1;

      // now we should wait until next polling requests is processed
      // or when connection is closed or handler is shutdown

      Int_t sendcnt = fSendCnt, loopcnt(0);

      while (!IsDisabled() && !engine->fDisabled) {
         gSystem->ProcessEvents();
         // if send counter changed - current send operation is completed
         if (sendcnt != fSendCnt)
            return 0;
         if (loopcnt++ > 1000) {
            loopcnt = 0;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
         }
      }

      return -1;
   }

   // probably this thread can continuously run
   std::thread thrd([this, engine] {
      while (!IsDisabled() && !engine->fDisabled) {
         PerformSend(engine);
         if (IsDisabled() || engine->fDisabled) break;
         std::unique_lock<std::mutex> lk(engine->fMutex);
         if (engine->fKind == THttpWSEngine::kNone) {
            engine->fWaiting = true;
            engine->fCond.wait(lk);
            engine->fWaiting = false;
         }
      }
   });

   engine->fSendThrd.swap(thrd);

   engine->fHasSendThrd = true;

   return 1;
}


////////////////////////////////////////////////////////////////////////////////
/// Perform send operation, stored in buffer

Int_t THttpWSHandler::PerformSend(std::shared_ptr<THttpWSEngine> engine)
{
   {
      std::lock_guard<std::mutex> grd(engine->fMutex);

      // no need to do something - operation was processed already by somebody else
      if (engine->fKind == THttpWSEngine::kNone)
         return 0;

      if (engine->fSending)
         return 1;
      engine->fSending = true;
   }

   if (IsDisabled() || engine->fDisabled)
      return 0;

   switch (engine->fKind) {
   case THttpWSEngine::kData:
      engine->Send(engine->fData.data(), engine->fData.length());
      break;
   case THttpWSEngine::kHeader:
      engine->SendHeader(engine->fHdr.c_str(), engine->fData.data(), engine->fData.length());
      break;
   case THttpWSEngine::kText:
      engine->SendCharStar(engine->fData.c_str());
      break;
   default:
      break;
   }

   engine->fData.clear();
   engine->fHdr.clear();

   {
      std::lock_guard<std::mutex> grd(engine->fMutex);
      engine->fSending = false;
      engine->fKind = THttpWSEngine::kNone;
   }

   return CompleteSend(engine);
}


////////////////////////////////////////////////////////////////////////////////
/// Complete current send operation

Int_t THttpWSHandler::CompleteSend(std::shared_ptr<THttpWSEngine> &engine)
{
   fSendCnt++;
   engine->fMTSend = false; // probably we do not need to lock mutex to reset flag
   CompleteWSSend(engine->GetId());
   return 0; // indicates that operation is completed
}


////////////////////////////////////////////////////////////////////////////////
/// Send binary data via given websocket id. Returns:
///
/// * -1 - in case of error
/// *  0 - when operation was executed immediately
/// *  1 - when send operation will be performed in different thread

Int_t THttpWSHandler::SendWS(UInt_t wsid, const void *buf, int len)
{
   auto engine = FindEngine(wsid, kTRUE);
   if (!engine) return -1;

   if ((IsSyncMode() || !AllowMTSend()) && engine->CanSendDirectly()) {
      engine->Send(buf, len);
      return CompleteSend(engine);
   }

   bool notify = false;

   // now we indicate that there is data and any thread can access it
   {
      std::lock_guard<std::mutex> grd(engine->fMutex);

      if (engine->fKind != THttpWSEngine::kNone) {
         Error("SendWS", "Data kind is not empty - something screwed up");
         return -1;
      }

      notify = engine->fWaiting;

      engine->fKind = THttpWSEngine::kData;

      engine->fData.resize(len);
      std::copy((const char *)buf, (const char *)buf + len, engine->fData.begin());
   }

   if (engine->fHasSendThrd) {
      if (notify) engine->fCond.notify_all();
      return 1;
   }

   return RunSendingThrd(engine);
}


////////////////////////////////////////////////////////////////////////////////
/// Send binary data with text header via given websocket id. Returns:
///
/// * -1 - in case of error,
/// *  0 - when operation was executed immediately,
/// *  1 - when send operation will be performed in different thread,

Int_t THttpWSHandler::SendHeaderWS(UInt_t wsid, const char *hdr, const void *buf, int len)
{
   auto engine = FindEngine(wsid, kTRUE);
   if (!engine) return -1;

   if ((IsSyncMode() || !AllowMTSend()) && engine->CanSendDirectly()) {
      engine->SendHeader(hdr, buf, len);
      return CompleteSend(engine);
   }

   bool notify = false;

   // now we indicate that there is data and any thread can access it
   {
      std::lock_guard<std::mutex> grd(engine->fMutex);

      if (engine->fKind != THttpWSEngine::kNone) {
         Error("SendWS", "Data kind is not empty - something screwed up");
         return -1;
      }

      notify = engine->fWaiting;

      engine->fKind = THttpWSEngine::kHeader;

      engine->fHdr = hdr;
      engine->fData.resize(len);
      std::copy((const char *)buf, (const char *)buf + len, engine->fData.begin());
   }

   if (engine->fHasSendThrd) {
      if (notify) engine->fCond.notify_all();
      return 1;
   }

   return RunSendingThrd(engine);
}

////////////////////////////////////////////////////////////////////////////////
/// Send string via given websocket id. Returns:
///
/// * -1 - in case of error,
/// *  0 - when operation was executed immediately,
/// *  1 - when send operation will be performed in different thread,

Int_t THttpWSHandler::SendCharStarWS(UInt_t wsid, const char *str)
{
   auto engine = FindEngine(wsid, kTRUE);
   if (!engine) return -1;

   if ((IsSyncMode() || !AllowMTSend()) && engine->CanSendDirectly()) {
      engine->SendCharStar(str);
      return CompleteSend(engine);
   }

   bool notify = false;

   // now we indicate that there is data and any thread can access it
   {
      std::lock_guard<std::mutex> grd(engine->fMutex);

      if (engine->fKind != THttpWSEngine::kNone) {
         Error("SendWS", "Data kind is not empty - something screwed up");
         return -1;
      }

      notify = engine->fWaiting;

      engine->fKind = THttpWSEngine::kText;
      engine->fData = str;
   }

   if (engine->fHasSendThrd) {
      if (notify) engine->fCond.notify_all();
      return 1;
   }

   return RunSendingThrd(engine);
}
