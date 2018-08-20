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

#include <thread>

/////////////////////////////////////////////////////////////////////////
///
/// THttpWSHandler
///
/// Class for user-side handling of websocket with THttpServer
/// 1. Create derived from  THttpWSHandler class and implement
///     ProcessWS() method, where all web sockets request handled.
/// 2. Register instance of derived class to running THttpServer
///
///        TUserWSHandler *handler = new TUserWSHandler("name1","title");
///        THttpServer *server = new THttpServer("http:8090");
///        server->Register("/subfolder", handler)
///
/// 3. Now server can accept web socket connection from outside.
///    For instance, from JavaScirpt one can connect to it with code:
///
///        var ws = new WebSocket("ws://hostname:8090/subfolder/name1/root.websocket")
///
/// 4. In the ProcessWS(THttpCallArg *arg) method following code should be implemented:
///
///     if (arg->IsMethod("WS_CONNECT")) {
///         return true;  // to accept incoming request
///      }
///
///      if (arg->IsMethod("WS_READY")) {
///          fWSId = arg->GetWSId(); // fWSId should be member of the user class
///          return true; // connection established
///      }
///
///     if (arg->IsMethod("WS_CLOSE")) {
///         fWSId = 0;
///         return true; // confirm close of socket
///     }
///
///     if (arg->IsMethod("WS_DATA")) {
///         // received data stored as POST data
///         std::string str((const char *)arg->GetPostData(), arg->GetPostDataLength());
///         std::cout << "got string " << str << std::endl;
///         // immediately send data back using websocket id
///         SendCharStarWS(fWSId, "our reply");
///         return true;
///     }
///
///////////////////////////////////////////////////////////////////////////

ClassImp(THttpWSHandler);

////////////////////////////////////////////////////////////////////////////////
/// normal constructor

THttpWSHandler::THttpWSHandler(const char *name, const char *title) : TNamed(name, title), fEngines(), fDisabled(kFALSE)
{
}

////////////////////////////////////////////////////////////////////////////////
/// destructor
/// Delete all websockets handles

THttpWSHandler::~THttpWSHandler()
{
   SetDisabled();
}

////////////////////////////////////////////////////////////////////////////////
/// Return websocket id with given sequential number
/// Number of websockets return with GetNumWS() method

UInt_t THttpWSHandler::GetWS(Int_t num) const
{
   auto iter = fEngines.begin() + num;
   return (*iter)->GetId();
}

////////////////////////////////////////////////////////////////////////////////
/// Find websocket connection handle with given id

std::shared_ptr<THttpWSEngine> THttpWSHandler::FindEngine(UInt_t wsid) const
{
   if (IsDisabled()) return nullptr;

   for (auto &eng : fEngines)
      if (eng->GetId() == wsid)
         return eng;

   return nullptr;
}

////////////////////////////////////////////////////////////////////////////////
/// Remove and destroy WS connection

void THttpWSHandler::RemoveEngine(std::shared_ptr<THttpWSEngine> &engine)
{
   for (auto iter = fEngines.begin(); iter != fEngines.end(); iter++)
      if (*iter == engine) {
         engine->ClearHandle();
         fEngines.erase(iter);
         break;
      }
}

////////////////////////////////////////////////////////////////////////////////
/// Process request to websocket
/// Different kind of requests coded into THttpCallArg::Method
///  "WS_CONNECT" - connection request
///  "WS_READY" - connection ready
///  "WS_CLOSE" - connection closed
/// All other are normal data, which are delivered to users

Bool_t THttpWSHandler::HandleWS(std::shared_ptr<THttpCallArg> &arg)
{
   if (IsDisabled()) return kFALSE;

   if (!arg->GetWSId())
      return ProcessWS(arg.get());

   // normally here one accept or reject connection requests
   if (arg->IsMethod("WS_CONNECT"))
      return ProcessWS(arg.get());

   auto engine = FindEngine(arg->GetWSId());

   if (arg->IsMethod("WS_READY")) {

      if (engine) {
         Error("HandleWS", "WS engine with similar id exists %u", arg->GetWSId());
         RemoveEngine(engine);
      }

      engine = arg->TakeWSEngine();
      fEngines.push_back(engine);

      if (!ProcessWS(arg.get())) {
         // if connection refused, remove engine again
         RemoveEngine(engine);
         return kFALSE;
      }

      return kTRUE;
   }

   if (arg->IsMethod("WS_CLOSE")) {
      // connection is closed, one can remove handle

      if (engine) {
         engine->ClearHandle();
         RemoveEngine(engine);
      }

      return ProcessWS(arg.get());
   }

   if (engine && engine->PreviewData(arg))
      return kTRUE;

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

   if (engine)
      RemoveEngine(engine);
}

////////////////////////////////////////////////////////////////////////////////
/// Send binary data via given websocket id
/// Returns -1 - in case of error,
///          0 - when operation was executed immediately,
///          1 - when send operation will be performed in different thread,

Int_t THttpWSHandler::SendWS(UInt_t wsid, const void *buf, int len)
{
   auto engine = FindEngine(wsid);
   if (!engine) return -1;

   if (engine->fMTSend) {
      Error("SendWS", "Call next send operation before previous is completed");
      return -1;
   }

   if (!AllowMTSend() || !engine->SupportMT()) {
      engine->Send(buf, len);
      return 0;
   }

   engine->fMTSend = true;

   std::string argbuf;
   argbuf.resize(len);
   std::copy((const char *)buf, (const char *)buf + len, argbuf.begin());

   std::thread thrd([this, argbuf, engine] {
      engine->Send(argbuf.data(), argbuf.length());
      engine->fMTSend = false;
      if (!IsDisabled()) CompleteMTSend(engine->GetId());
   });

   thrd.detach(); // let continue thread execution without thread handle

   return 1;
}

////////////////////////////////////////////////////////////////////////////////
/// Send binary data with text header via given websocket id
/// Returns -1 - in case of error,
///          0 - when operation was executed immediately,
///          1 - when send operation will be performed in different thread,

Int_t THttpWSHandler::SendHeaderWS(UInt_t wsid, const char *hdr, const void *buf, int len)
{
   auto engine = FindEngine(wsid);
   if (!engine) return -1;

   if (engine->fMTSend) {
      Error("SendHeaderWS", "Call next send operation before previous is completed");
      return -1;
   }

   if (!AllowMTSend() || !engine->SupportMT()) {
      engine->SendHeader(hdr, buf, len);
      return 0;
   }

   engine->fMTSend = true;

   std::string arghdr(hdr), argbuf;
   argbuf.resize(len);
   std::copy((const char *)buf, (const char *)buf + len, argbuf.begin());

   std::thread thrd([this, arghdr, argbuf, engine] {
      engine->SendHeader(arghdr.c_str(), argbuf.data(), argbuf.length());
      engine->fMTSend = false;
      if (!IsDisabled()) CompleteMTSend(engine->GetId());
   });

   thrd.detach(); // let continue thread execution without thread handle

   return 1;
}

////////////////////////////////////////////////////////////////////////////////
/// Send string via given websocket id
/// Returns -1 - in case of error,
///          0 - when operation was executed immediately,
///          1 - when send operation will be performed in different thread,

Int_t THttpWSHandler::SendCharStarWS(UInt_t wsid, const char *str)
{
   auto engine = FindEngine(wsid);
   if (!engine) return -1;

   if (engine->fMTSend) {
      Error("SendCharStarWS", "Call next send operation before previous is completed");
      return -1;
   }

   if (!AllowMTSend() || !engine->SupportMT()) {
      engine->SendCharStar(str);
      return 0;
   }

   engine->fMTSend = true;

   std::string arg(str);

   std::thread thrd([this, arg, engine] {
      engine->SendCharStar(arg.c_str());
      engine->fMTSend = false;
      if (!IsDisabled()) CompleteMTSend(engine->GetId());
   });

   thrd.detach(); // let continue thread execution without thread handle

   return 1;
}
