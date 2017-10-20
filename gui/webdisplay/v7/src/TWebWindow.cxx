/// \file TWebWindow.cxx
/// \ingroup WebGui ROOT7
/// \author Sergey Linev <s.linev@gsi.de>
/// \date 2017-10-16
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2017, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/TWebWindow.hxx"

#include "ROOT/TWebWindowsManager.hxx"

#include "THttpEngine.h"
#include "THttpCallArg.h"

#include <cassert>
#include <cstring>
#include <cstdlib>

namespace ROOT {
namespace Experimental {

/// just wrapper to deliver WS call-backs to the TWebWindow class

class TWebWindowWSHandler : public THttpWSHandler {
public:
   TWebWindow *fDispl; ///<! back-pointer to display

   TWebWindowWSHandler(TWebWindow *displ) : THttpWSHandler("name","title"), fDispl(displ) {}
   ~TWebWindowWSHandler() { fDispl = nullptr; }

   virtual TString GetDefaultPageContent() override
   {
      return "file:$jsrootsys/files/canvas.htm";
   }

   virtual Bool_t ProcessWS(THttpCallArg *arg) override
   {
      return fDispl->ProcessWS(arg);
   }

};


} // namespace Experimental
} // namespace ROOT

ROOT::Experimental::TWebWindow::~TWebWindow()
{
   Cleanup();
}


void ROOT::Experimental::TWebWindow::Cleanup()
{
   for (auto iter = fConn.begin(); iter != fConn.end(); ++iter) {
      if (iter->fHandle) {
         iter->fHandle->ClearHandle();
         delete iter->fHandle;
         iter->fHandle = nullptr;
      }
   }

   fConn.clear();


   if (fMgr)
      fMgr->CloseDisplay(this);

   if (fWSHandler) {
      delete fWSHandler;
      fWSHandler = nullptr;
   }
}


// TODO: add callback which executed when exactly this window is opened and connection is established

bool ROOT::Experimental::TWebWindow::Show(const std::string &where)
{
   if (!fMgr) return false;

   bool first_time = false;

   if (!fWSHandler) {
      fWSHandler = new TWebWindowWSHandler(this);
      fWSHandler->SetName(Form("win%u",GetId()));
      first_time = true;
   }

   return fMgr->Show(this, where, first_time);
}


bool ROOT::Experimental::TWebWindow::ProcessWS(THttpCallArg *arg)
{
   if (!arg)
      return kTRUE;

   // try to identify connection for given WS request
   WebConn *conn = 0;
   auto iter = fConn.begin();
   while (iter != fConn.end()) {
      if (iter->fHandle && (iter->fHandle->GetId() == arg->GetWSId()) && arg->GetWSId()) {
         conn = &(*iter);
         break;
      }
      ++iter;
   }

   if (strcmp(arg->GetMethod(), "WS_CONNECT") == 0) {

      // accept all requests, in future one could limit number of connections
      // arg->Set404(); // refuse connection
      return true;
   }

   if (strcmp(arg->GetMethod(), "WS_READY") == 0) {
      THttpWSEngine *wshandle = dynamic_cast<THttpWSEngine *>(arg->TakeWSHandle());

      assert(conn == 0  && "WSHandle with given websocket id exists!!!");

      WebConn newconn;
      newconn.fHandle = wshandle;
      newconn.fConnId = ++fConnCnt; // unique connection id
      fConn.push_back(newconn);

      // CheckDataToSend();

      return true;
   }

   if (strcmp(arg->GetMethod(), "WS_CLOSE") == 0) {
      // connection is closed, one can remove handle

      // UInt_t connid = 0;

      if (conn && fDataCallback)
         fDataCallback(conn->fConnId, "CONN_CLOSED");

      if (conn && conn->fHandle) {
         // connid = conn->fHandle->GetId();
         conn->fHandle->ClearHandle();
         delete conn->fHandle;
         conn->fHandle = 0;
      }

      if (conn)
         fConn.erase(iter);

      // if there are no other connections - cancel all submitted commands
      // CancelCommands((fWebConn.size() == 0), connid);

      // CheckDataToSend(); // check if data should be send via other connections

      return true;
   }

   assert(strcmp(arg->GetMethod(), "WS_DATA") == 0 && "WS_DATA request expected!");

   assert(conn != 0 && "Get websocket data without valid connection - ignore!!!");

   // TODO: move to THttpServer, workaround for LongPolling socket
   if (conn->fHandle->PreviewData(arg))
      return true;

   if (arg->GetPostDataLength() <= 0)
      return true;

   // here processing of received data should be performed
   // this is task for the implemented windows

   const char *buf = (const char *)arg->GetPostData();
   char *str_end = 0;

   printf("Get portion of data %d %s\n", arg->GetPostDataLength(), buf);

   unsigned long ackn_oper = std::strtoul(buf, &str_end, 10);
   assert(str_end != 0 && *str_end != ':' && "missing number of acknowledged operations");

   unsigned long can_send = std::strtoul(str_end+1, &str_end, 10);
   assert(str_end != 0 && *str_end != ':' && "missing can_send counter");

   unsigned long nchannel = std::strtoul(str_end+1, &str_end, 10);
   assert(str_end != 0 && *str_end != ':' && "missing channel number");

   unsigned processed_len = (str_end + 1 - buf);

   assert(processed_len <= arg->GetPostDataLength() && "corrupted buffer");

   std::string cdata(str_end+1, arg->GetPostDataLength() - processed_len);

   conn->fSendCredits += ackn_oper;
   conn->fRecvCount++;
   conn->fClientCredits = (int) can_send;

   if (nchannel == 0) {
      // special default channel for basic communications
      if (cdata == "READY") {
         if (!conn->fReady) fDataCallback(conn->fConnId, "CONN_READY");
         conn->fReady = true;
      }
   } else if (nchannel == 1) {
      fDataCallback(conn->fConnId, cdata);
   } else if (nchannel > 1) {
      conn->fCallBack(conn->fConnId, cdata);
   }

   CheckDataToSend();

   return true;
}

void ROOT::Experimental::TWebWindow::SendDataViaConnection(ROOT::Experimental::TWebWindow::WebConn &conn, int chid, const std::string &data)
{
   if (!conn.fHandle) return;

   assert(conn.fSendCredits>0 && "No credits to send data via connection");

   std::string buf;
   buf.reserve(data.length() + 100);

   buf.append(std::to_string(conn.fRecvCount));
   buf.append(":");
   buf.append(std::to_string(conn.fSendCredits));
   buf.append(":");
   conn.fRecvCount = 0; // we confirm how many packages was received
   conn.fSendCredits--;

   if (chid >= 0) {
      buf.append(std::to_string(chid));
      buf.append(":");
   }

   // TODO: should we add extra : just as placeholder for any kind of custom data??
   buf.append(data);

   conn.fHandle->SendCharStar(buf.c_str());
}

/// Check if data to any connection can be send
void ROOT::Experimental::TWebWindow::CheckDataToSend(bool only_once)
{
   bool isany = false;

   do {
      isany = false;

      for (auto iter = fConn.begin(); iter != fConn.end(); ++iter) {
         if (iter->fSendCredits <= 0) continue;

         if (iter->fQueue.size() > 0) {
            SendDataViaConnection(*iter, -1, iter->fQueue.front());
            iter->fQueue.pop_front();
            isany = true;
         } else if ((iter->fClientCredits < 3) && (iter->fRecvCount > 1)) {
            // give more credits to the client
            printf("Send keep alive to client recv:%d client:%d\n", iter->fRecvCount, iter->fClientCredits);
            SendDataViaConnection(*iter, 0, "KEEPALIVE");
            isany = true;
         }

      }

   } while (isany && !only_once);
}

// returns true if sending via specified connection can be performed
// if direct==true, requires that direct sending (without queue) is possible

bool ROOT::Experimental::TWebWindow::CanSend(unsigned connid, bool direct) const
{
   for (auto iter = fConn.begin(); iter != fConn.end(); ++iter) {
      if (connid && connid != iter->fConnId) continue;

      if (direct && ((iter->fQueue.size()>0) || (iter->fSendCredits==0))) return false;

      if (iter->fQueue.size() >= fMaxQueueLength) return false;
   }

   return true;
}


/// Sends data to specified connection
/// If connid==0, data will be send to all connections for channel 1
void ROOT::Experimental::TWebWindow::Send(const std::string &data, unsigned connid, unsigned chid)
{
   for (auto iter = fConn.begin(); iter != fConn.end(); ++iter) {
      if (connid && connid != iter->fConnId) continue;

      if ((iter->fQueue.size()==0) && (iter->fSendCredits>0)) {
         SendDataViaConnection(*iter, chid, data);
      } else {
         assert(iter->fQueue.size() < fMaxQueueLength && "Maximum queue length achieved");
         iter->fQueue.push_back(std::to_string(chid)+":" + data);
      }
   }

   CheckDataToSend();
}
