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

THttpWSHandler::THttpWSHandler(const char *name, const char *title) : TNamed(name, title), fEngines()
{
}

////////////////////////////////////////////////////////////////////////////////
/// destructor
/// Delete all websockets handles

THttpWSHandler::~THttpWSHandler()
{
   for (auto iter = fEngines.begin(); iter != fEngines.end(); iter++) {
      THttpWSEngine *engine = *iter;
      engine->ClearHandle();
      delete engine;
   }

   fEngines.clear();
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

THttpWSEngine *THttpWSHandler::FindEngine(UInt_t wsid) const
{
   for (auto iter = fEngines.begin(); iter != fEngines.end(); iter++)
      if ((*iter)->GetId() == wsid)
         return *iter;

   return nullptr;
}

////////////////////////////////////////////////////////////////////////////////
/// Remove and destroy WS connection

void THttpWSHandler::RemoveEngine(THttpWSEngine *engine)
{
   for (auto iter = fEngines.begin(); iter != fEngines.end(); iter++)
      if (*iter == engine) {
         fEngines.erase(iter);
         break;
      }

   delete engine;
}

////////////////////////////////////////////////////////////////////////////////
/// Process request to websocket
/// Different kind of requests coded into THttpCallArg::Method
///  "WS_CONNECT" - connection request
///  "WS_READY" - connection ready
///  "WS_CLOSE" - connection closed
/// All other are normal data, which are delivered to users

Bool_t THttpWSHandler::HandleWS(THttpCallArg *arg)
{
   if (!arg->GetWSId())
      return ProcessWS(arg);

   // normally here one accept or reject connection requests
   if (arg->IsMethod("WS_CONNECT"))
      return ProcessWS(arg);

   THttpWSEngine *engine = FindEngine(arg->GetWSId());

   if (arg->IsMethod("WS_READY")) {

      if (engine) {
         Error("HandleWS", "WS engine with similar id exists %u\n", arg->GetWSId());
         RemoveEngine(engine);
      }

      engine = arg->TakeWSHandle();

      fEngines.push_back(engine);

      if (!ProcessWS(arg)) {
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

      return ProcessWS(arg);
   }

   if (engine && engine->PreviewData(*arg))
      return kTRUE;

   Bool_t res = ProcessWS(arg);

   if (engine)
      engine->PostProcess(*arg);

   return res;
}

////////////////////////////////////////////////////////////////////////////////
/// Close connection with given websocket id

void THttpWSHandler::CloseWS(UInt_t wsid)
{
   THttpWSEngine *engine = FindEngine(wsid);

   if (engine)
      RemoveEngine(engine);
}

////////////////////////////////////////////////////////////////////////////////
/// Send binary data via given websocket id

void THttpWSHandler::SendWS(UInt_t wsid, const void *buf, int len)
{
   THttpWSEngine *engine = FindEngine(wsid);

   if (engine)
      engine->Send(buf, len);
}

////////////////////////////////////////////////////////////////////////////////
/// Send string via given websocket id

void THttpWSHandler::SendCharStarWS(UInt_t wsid, const char *str)
{
   THttpWSEngine *engine = FindEngine(wsid);

   if (engine)
      engine->SendCharStar(str);
}
