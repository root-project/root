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

#include "TError.h"
#include "THttpCallArg.h"

#include <cstring>
#include <cstdlib>

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// THttpLongPollEngine                                                  //
//                                                                      //
// Emulation of websocket with long poll requests                       //
// Allows to send data from server to client without explicit request   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

const std::string THttpLongPollEngine::gLongPollNope = "<<nope>>";

//////////////////////////////////////////////////////////////////////////
/// constructor

THttpLongPollEngine::THttpLongPollEngine(bool raw) : THttpWSEngine(), fRaw(raw)
{
}

//////////////////////////////////////////////////////////////////////////
/// returns ID of the engine, created from this pointer

UInt_t THttpLongPollEngine::GetId() const
{
   const void *ptr = (const void *)this;
   return TString::Hash((void *)&ptr, sizeof(void *));
}

//////////////////////////////////////////////////////////////////////////
/// clear request, normally called shortly before destructor

void THttpLongPollEngine::ClearHandle(Bool_t)
{
   std::shared_ptr<THttpCallArg> poll;

   {
      std::lock_guard<std::mutex> grd(fMutex);
      poll = std::move(fPoll);
   }

   if (poll) {
      poll->Set404();
      poll->NotifyCondition();
   }
}

//////////////////////////////////////////////////////////////////////////
/// Create raw buffer which should be send as reply
/// For the raw mode all information must be send via binary response

std::string THttpLongPollEngine::MakeBuffer(const void *buf, int len, const char *hdr)
{
   std::string res;

   if (!fRaw) {
      res.resize(len);
      std::copy((const char *)buf, (const char *)buf + len, res.begin());
      return res;
   }

   int hdrlen = hdr ? strlen(hdr) : 0;
   std::string hdrstr = "bin:";
   hdrstr.append(std::to_string(hdrlen));

   while ((hdrstr.length() + 1 + hdrlen) % 8 != 0)
      hdrstr.append(" ");
   hdrstr.append(":");
   if (hdrlen > 0)
      hdrstr.append(hdr);

   res.resize(hdrstr.length() + len);
   std::copy(hdrstr.begin(), hdrstr.begin() + hdrstr.length(), res.begin());
   std::copy((const char *)buf, (const char *)buf + len, res.begin() + hdrstr.length());

   return res;
}

//////////////////////////////////////////////////////////////////////////
/// Send binary data via connection

void THttpLongPollEngine::Send(const void *buf, int len)
{
   std::shared_ptr<THttpCallArg> poll;

   {
      std::lock_guard<std::mutex> grd(fMutex);
      poll = std::move(fPoll);
   }

   if(!poll) {
      Error("Send", "Operation invoked before polling request obtained");
      return;
   }

   std::string buf2 = MakeBuffer(buf, len);

   poll->SetBinaryContent(std::move(buf2));
   poll->NotifyCondition();
}

//////////////////////////////////////////////////////////////////////////
/// Send binary data with text header via connection

void THttpLongPollEngine::SendHeader(const char *hdr, const void *buf, int len)
{
   std::shared_ptr<THttpCallArg> poll;

   {
      std::lock_guard<std::mutex> grd(fMutex);
      poll = std::move(fPoll);
   }

   if(!poll) {
      Error("SendHeader", "Operation invoked before polling request obtained");
      return;
   }

   std::string buf2 = MakeBuffer(buf, len, hdr);

   poll->SetBinaryContent(std::move(buf2));
   if (!fRaw)
      poll->SetExtraHeader("LongpollHeader", hdr);
   poll->NotifyCondition();
}

//////////////////////////////////////////////////////////////////////////
/// Send const char data
/// Either do it immediately or keep in internal buffer

void THttpLongPollEngine::SendCharStar(const char *buf)
{
   std::shared_ptr<THttpCallArg> poll;

   {
      std::lock_guard<std::mutex> grd(fMutex);
      poll = std::move(fPoll);
   }

   if(!poll) {
      Error("SendCharStart", "Operation invoked before polling request obtained");
      return;
   }

   std::string sendbuf(fRaw ? "txt:" : "");
   sendbuf.append(buf);

   if (fRaw) poll->SetBinaryContent(std::move(sendbuf));
        else poll->SetTextContent(std::move(sendbuf));
   poll->NotifyCondition();
}

//////////////////////////////////////////////////////////////////////////////
/// Preview data for given socket
/// Method called by WS handler before processing websocket data
/// Returns kTRUE when user should ignore such http request - it is for internal use

Bool_t THttpLongPollEngine::PreProcess(std::shared_ptr<THttpCallArg> &arg)
{
   if (!strstr(arg->GetQuery(), "&dummy"))
      return kFALSE;

   arg->SetPostponed(); // mark http request as pending, http server should wait for notification

   std::shared_ptr<THttpCallArg> poll;

   {
      std::lock_guard<std::mutex> grd(fMutex);
      poll = std::move(fPoll);
      fPoll = arg;         // keep reference on polling request
   }

   if (arg == poll)
      Fatal("PreviewData", "Submit same THttpCallArg object once again");

   if (poll) {
      Error("PreviewData", "Get next dummy request when previous not completed");
      // if there are pending request, reply it immediately
      // normally should never happen
      if (fRaw) poll->SetBinaryContent(std::string("txt:") + gLongPollNope);
           else poll->SetTextContent(std::string(gLongPollNope));
      poll->NotifyCondition();         // inform http server that request is processed
   }

   // if arguments has "&dummy" string, user should not process it
   return kTRUE;
}

//////////////////////////////////////////////////////////////////////////////
/// Normally requests from client does not replied directly for longpoll socket
/// Therefore one can use such request to send data, which was submitted before to the queue

void THttpLongPollEngine::PostProcess(std::shared_ptr<THttpCallArg> &arg)
{
   if (fRaw) arg->SetBinaryContent(std::string("txt:") + gLongPollNope);
        else arg->SetTextContent(std::string(gLongPollNope));
}

//////////////////////////////////////////////////////////////////////////////
/// Indicate that polling requests is there and can be immediately invoked

Bool_t THttpLongPollEngine::CanSendDirectly()
{
   std::lock_guard<std::mutex> grd(fMutex);
   return fPoll ? kTRUE : kFALSE;
}
