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
#include "TSystem.h"

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

const char *THttpLongPollEngine::gLongPollNope = "<<nope>>";


//////////////////////////////////////////////////////////////////////////
/// constructor

THttpLongPollEngine::THttpLongPollEngine(std::shared_ptr<THttpCallArg> arg, bool raw) : THttpWSEngine(arg), fRaw(raw)
{
   arg->SetWSId(GetId());
}

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
      fPoll.reset();
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
      memcpy((void *) res.data(), buf, len);
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

   memcpy((void *)res.data(), hdrstr.data(), hdrstr.length());
   memcpy((char *)res.data() + hdrstr.length(), buf, len);

   return res;
}

//////////////////////////////////////////////////////////////////////////
/// Send binary data via connection - not supported

void THttpLongPollEngine::Send(const void *buf, int len)
{
   std::string buf2 = MakeBuffer(buf, len);

   if (fPoll) {
      fPoll->SetBinaryContent(std::move(buf2));
      fPoll->NotifyCondition();
      fPoll.reset();
   } else {
      fQueue.emplace_back(true, std::move(buf2));
      if (fQueue.size() > 100)
         Error("Send", "Too many send operations %u in the queue, check algorithms", (unsigned) fQueue.size());
   }
}

//////////////////////////////////////////////////////////////////////////
/// Send binary data with text header via connection - not supported

void THttpLongPollEngine::SendHeader(const char *hdr, const void *buf, int len)
{
   std::string buf2 = MakeBuffer(buf, len, hdr);

   if (fPoll) {
      fPoll->SetBinaryContent(std::move(buf2));
      if (!fRaw)
         fPoll->SetExtraHeader("LongpollHeader", hdr);
      fPoll->NotifyCondition();
      fPoll.reset();
   } else {
      fQueue.emplace_back(true, std::move(buf2), hdr);
      if (fQueue.size() > 100)
         Error("SendHeader", "Too many send operations %u in the queue, check algorithms", (unsigned) fQueue.size());
   }
}

//////////////////////////////////////////////////////////////////////////
/// Send const char data
/// Either do it immediately or keep in internal buffer

void THttpLongPollEngine::SendCharStar(const char *buf)
{
   std::string sendbuf(fRaw ? "txt:" : "");
   sendbuf.append(buf);

   if (fPoll) {
      if (fRaw) fPoll->SetBinaryContent(std::move(sendbuf));
           else fPoll->SetTextContent(std::move(sendbuf));
      fPoll->NotifyCondition();
      fPoll.reset();
   } else {
      fQueue.emplace_back(false, std::move(sendbuf));
      if (fQueue.size() > 100)
         Error("SendCharStar", "Too many send operations %u in the queue, check algorithms", (unsigned) fQueue.size());
   }
}

//////////////////////////////////////////////////////////////////////////////
/// Preview data for given socket
/// function called in the user code before processing correspondent websocket data
/// returns kTRUE when user should ignore such http request - it is for internal use

Bool_t THttpLongPollEngine::PreviewData(std::shared_ptr<THttpCallArg> &arg)
{
   if (!strstr(arg->GetQuery(), "&dummy")) {
      // this is normal request, deliver and process it as any other
      // put dummy content, it can be overwritten in the future
      arg->SetTextContent(std::string(gLongPollNope));
      return kFALSE;
   }

   if (arg == fPoll)
      Fatal("PreviewData", "Submit same THttpCallArg object once again");

   if (fPoll) {
      Error("PreviewData", "Get next dummy request when previous not completed");
      // if there are pending request, reply it immediately
      if (fRaw) fPoll->SetBinaryContent(std::string("txt:") + gLongPollNope);
           else fPoll->SetTextContent(std::string(gLongPollNope)); // normally should never happen
      fPoll->NotifyCondition();         // inform http server that request is processed
      fPoll.reset();
   }

   if (fQueue.size() > 0) {
      QueueItem &item = fQueue.front();
      if (item.fBinary) {
         arg->SetBinaryContent(std::move(item.fData));
         if (!fRaw && !item.fHdr.empty())
            arg->SetExtraHeader("LongpollHeader", item.fHdr.c_str());
      } else {
         arg->SetTextContent(std::move(item.fData));
      }
      fQueue.erase(fQueue.begin());
   } else {
      arg->SetPostponed(); // mark http request as pending, http server should wait for notification
      fPoll = arg;         // keep reference on polling request
   }

   // if arguments has "&dummy" string, user should not process it
   return kTRUE;
}

//////////////////////////////////////////////////////////////////////////////
/// Normally requests from client does not replied directly
/// Therefore one can use it to send data with it

void THttpLongPollEngine::PostProcess(std::shared_ptr<THttpCallArg> &arg)
{
   if (arg->IsText() && (arg->GetContentLength() == (Long_t)strlen(gLongPollNope)) &&
       (strcmp((const char *)arg->GetContent(), gLongPollNope) == 0)) {
      if (fQueue.size() > 0) {
         QueueItem &item = fQueue.front();
         if (item.fBinary) {
            arg->SetBinaryContent(std::move(item.fData));
            if (!fRaw && !item.fHdr.empty())
               arg->SetExtraHeader("LongpollHeader", item.fHdr.c_str());
         } else {
            arg->SetTextContent(std::move(item.fData));
         }
         fQueue.erase(fQueue.begin());
      } else if (fRaw) {
         arg->SetContent(std::string("txt:") + gLongPollNope);
      }
   }
}
