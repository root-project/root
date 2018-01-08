// $Id$
// Author: Sergey Linev   8/01/2018

/*************************************************************************
 * Copyright (C) 1995-2013, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "THttpLongPollEngine.h"

#include "THttpCallArg.h"
#include <TSystem.h>

#include <string.h>

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// THttpLongPollEngine                                                  //
//                                                                      //
// Emulation of websocket with long poll requests                       //
// Allows to send data from server to client without explicit request   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

const char *THttpLongPollEngine::gLongPollNope = "<<nope>>";

//////////////////////////////////////////////////////////////////////////
/// returns ID of the engine, created from this pointer

UInt_t THttpLongPollEngine::GetId() const
{
   const void *ptr = (const void *)this;
   return TString::Hash((void *)&ptr, sizeof(void *));
}

//////////////////////////////////////////////////////////////////////////
/// clear request, waiting for next portion of data

void THttpLongPollEngine::ClearHandle()
{
   if (fPoll) {
      fPoll->Set404();
      fPoll->NotifyCondition();
      fPoll = nullptr;
   }
}

//////////////////////////////////////////////////////////////////////////
/// Send binary data via connection - not supported

void THttpLongPollEngine::Send(const void * /*buf*/, int /*len*/)
{
   Error("Send", "Binary send is not supported, use only text");
}

//////////////////////////////////////////////////////////////////////////
/// Send const char data
/// Either do it immediately or keep in internal buffer

void THttpLongPollEngine::SendCharStar(const char *buf)
{
   if (fPoll) {
      fPoll->SetContentType("text/plain");
      fPoll->SetContent(buf);
      fPoll->NotifyCondition();
      fPoll = nullptr;
   } else {
      fBuf.push_back(std::string(buf));
      if (fBuf.size() > 100)
         Error("SendCharStar", "Too many send operations %d, check algorithms", (int)fBuf.size());
   }
}

//////////////////////////////////////////////////////////////////////////////
/// Preview data for given socket
/// function called in the user code before processing correspondent websocket data
/// returns kTRUE when user should ignore such http request - it is for internal use

Bool_t THttpLongPollEngine::PreviewData(THttpCallArg *arg)
{
   if (!strstr(arg->GetQuery(), "&dummy")) {
      // this is normal request, deliver and process it as any other
      // put dummy content, it can be overwritten in the future
      arg->SetContentType("text/plain");
      arg->SetContent(gLongPollNope);
      return kFALSE;
   }

   if (arg == fPoll) {
      Error("PreviewData", "NEVER SHOULD HAPPEN");
      gSystem->Exit(12);
   }

   if (fPoll) {
      Info("PreviewData", "Get dummy request when previous not completed");
      // if there are pending request, reply it immediately
      fPoll->SetContentType("text/plain");
      fPoll->SetContent(gLongPollNope); // normally should never happen
      fPoll->NotifyCondition();
      fPoll = nullptr;
   }

   if (fBuf.size() > 0) {
      arg->SetContentType("text/plain");
      arg->SetContent(fBuf.front().c_str());
      fBuf.pop_front();
   } else {
      arg->SetPostponed();
      fPoll = arg;
   }

   // if arguments has "&dummy" string, user should not process it
   return kTRUE;
}

//////////////////////////////////////////////////////////////////////////////
/// Normally requests from client does not replied directly
/// Therefore one can use it to send data with it

void THttpLongPollEngine::PostProcess(THttpCallArg *arg)
{
   if ((fBuf.size() > 0) && arg->IsContentType("text/plain") &&
       (arg->GetContentLength() == (Long_t)strlen(gLongPollNope)) &&
       (strcmp((const char *)arg->GetContent(), gLongPollNope) == 0)) {
      arg->SetContent(fBuf.front().c_str());
      fBuf.pop_front();
   }
}
