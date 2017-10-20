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


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// THttpWSHandler                                                       //
//                                                                      //
// Abstract class for processing websocket requests                     //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

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

Bool_t THttpWSHandler::DirecltyHandle(THttpCallArg *arg)
{
   if (!arg->GetWSId()) return ProcessWS(arg);

   THttpWSEngine* engine = FindEngine(arg->GetWSId());

   if (strcmp(arg->GetMethod(), "WS_CONNECT") == 0) {
      // accept all requests, in future one could limit number of connections
      return ProcessWS(arg);
   }

   if (strcmp(arg->GetMethod(), "WS_READY") == 0) {

      if (engine) {
         Error("DirecltyHandle","WS engine with similar id exists %u\n", arg->GetWSId());
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

   if (strcmp(arg->GetMethod(), "WS_CLOSE") == 0) {
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
