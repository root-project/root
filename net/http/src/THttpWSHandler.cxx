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

THttpWSHandler::THttpWSHandler(const char *name, const char *title) :
   TNamed(name, title), fEngines()
{
}

THttpWSHandler::~THttpWSHandler()
{
   TIter iter(&fEngines);
   THttpWSEngine *engine = 0;

   while ((engine = (THttpWSEngine *)iter()) != 0)
      engine->ClearHandle();

   fEngines.Delete();
}


THttpWSEngine *THttpWSHandler::FindEngine(UInt_t id) const
{
   TIter iter(&fEngines);
   THttpWSEngine *engine = 0;

   while ((engine = (THttpWSEngine *)iter()) != 0) {
      if (engine->GetId() == id) return engine;
   }

   return 0;
}

Bool_t THttpWSHandler::HandleWS(THttpCallArg *arg)
{
   if (!arg->GetWSId()) return ProcessWS(arg);

   THttpWSEngine* engine = FindEngine(arg->GetWSId());

   if (arg->IsMethod("WS_CONNECT")) {
      // accept all requests, in future one could limit number of connections
      return ProcessWS(arg);
   }

   if (arg->IsMethod("WS_READY")) {

      if (engine) {
         Error("HandleWS","WS engine with similar id exists %u\n", arg->GetWSId());
         fEngines.Remove(engine);
         delete engine;
      }

      THttpWSEngine *wshandle = dynamic_cast<THttpWSEngine *>(arg->TakeWSHandle());

      fEngines.Add(wshandle);

      if (!ProcessWS(arg)) {
         // if connection refused, remove engine again
         fEngines.Remove(wshandle);
         delete wshandle;
         return kFALSE;
      }

      return kTRUE;
   }

   if (arg->IsMethod("WS_CLOSE")) {
      // connection is closed, one can remove handle

      if (engine) {
         engine->ClearHandle();
         fEngines.Remove(engine);
         delete engine;
      }

      return ProcessWS(arg);
   }

   if (engine && engine->PreviewData(arg)) return kTRUE;

   return ProcessWS(arg);
}

void THttpWSHandler::CloseWS(UInt_t wsid)
{
   THttpWSEngine* engine = FindEngine(wsid);

   if (engine) {
      fEngines.Remove(engine);
      delete engine;
   }
}

void THttpWSHandler::SendWS(UInt_t wsid, const void *buf, int len)
{
   THttpWSEngine* engine = FindEngine(wsid);

   if (engine) engine->Send(buf, len);
}

void THttpWSHandler::SendCharStarWS(UInt_t wsid, const char *str)
{
   THttpWSEngine* engine = FindEngine(wsid);

   if (engine) engine->SendCharStar(str);
}
