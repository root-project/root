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
#include <ROOT/TLogger.hxx>

#include "THttpCallArg.h"
#include "THttpWSHandler.h"

#include <cassert>
#include <cstring>
#include <cstdlib>

namespace ROOT {
namespace Experimental {

/// just wrapper to deliver WS call-backs to the TWebWindow class

class TWebWindowWSHandler : public THttpWSHandler {
public:
   TWebWindow *fDispl; ///<! back-pointer to display

   TWebWindowWSHandler(TWebWindow *displ) : THttpWSHandler("name", "title"), fDispl(displ) {}
   ~TWebWindowWSHandler() { fDispl = nullptr; }

   virtual TString GetDefaultPageContent() override { return fDispl->fDefaultPage.c_str(); }

   virtual Bool_t ProcessWS(THttpCallArg *arg) override { return fDispl->ProcessWS(arg); }
};

} // namespace Experimental
} // namespace ROOT

ROOT::Experimental::TWebWindow::~TWebWindow()
{
   fConn.clear();

   if (fMgr)
      fMgr->CloseWindow(this);

   if (fWSHandler) {
      delete fWSHandler;
      fWSHandler = nullptr;
   }
}

void ROOT::Experimental::TWebWindow::SetPanelName(const std::string &name)
{
   assert(fConn.size() == 0 && "Cannot configure panel when connection exists");

   fPanelName = name;

   SetDefaultPage("file:$jsrootsys/files/panel.htm");

   fConnLimit = 1;
}

// TODO: add callback which executed when exactly this window is opened and connection is established

void ROOT::Experimental::TWebWindow::CreateWSHandler()
{
   if (!fWSHandler) {
      fWSHandler = new TWebWindowWSHandler(this);
      fWSHandler->SetName(Form("win%u", GetId()));
   }
}

bool ROOT::Experimental::TWebWindow::Show(const std::string &where)
{
   bool res = fMgr->Show(this, where);
   if (res)
      fShown = true;
   return res;
}

bool ROOT::Experimental::TWebWindow::ProcessWS(THttpCallArg *arg)
{
   if (!arg || (arg->GetWSId() == 0))
      return kTRUE;

   // try to identify connection for given WS request
   WebConn *conn = 0;
   auto iter = fConn.begin();
   while (iter != fConn.end()) {
      if (iter->fWSId == arg->GetWSId()) {
         conn = &(*iter);
         break;
      }
      ++iter;
   }

   if (arg->IsMethod("WS_CONNECT")) {

      // refuse connection when limit exceed limit
      if (fConnLimit && (fConn.size() >= fConnLimit))
         return false;

      return true;
   }

   if (arg->IsMethod("WS_READY")) {
      assert(conn == 0 && "WSHandle with given websocket id exists!!!");

      WebConn newconn;
      newconn.fWSId = arg->GetWSId();
      newconn.fConnId = ++fConnCnt; // unique connection id
      fConn.push_back(newconn);

      // CheckDataToSend();

      return true;
   }

   if (arg->IsMethod("WS_CLOSE")) {
      // connection is closed, one can remove handle

      if (conn && fDataCallback)
         fDataCallback(conn->fConnId, "CONN_CLOSED");

      if (conn)
         fConn.erase(iter);

      return true;
   }

   assert(arg->IsMethod("WS_DATA") && "WS_DATA request expected!");

   assert(conn != 0 && "Get websocket data without valid connection - ignore!!!");

   if (arg->GetPostDataLength() <= 0)
      return true;

   // here processing of received data should be performed
   // this is task for the implemented windows

   const char *buf = (const char *)arg->GetPostData();
   char *str_end = 0;

   printf("Get portion of data %d %.30s\n", (int)arg->GetPostDataLength(), buf);

   unsigned long ackn_oper = std::strtoul(buf, &str_end, 10);
   assert(str_end != 0 && *str_end == ':' && "missing number of acknowledged operations");

   unsigned long can_send = std::strtoul(str_end + 1, &str_end, 10);
   assert(str_end != 0 && *str_end == ':' && "missing can_send counter");

   unsigned long nchannel = std::strtoul(str_end + 1, &str_end, 10);
   assert(str_end != 0 && *str_end == ':' && "missing channel number");

   unsigned processed_len = (str_end + 1 - buf);

   assert(processed_len <= arg->GetPostDataLength() && "corrupted buffer");

   std::string cdata(str_end + 1, arg->GetPostDataLength() - processed_len);

   conn->fSendCredits += ackn_oper;
   conn->fRecvCount++;
   conn->fClientCredits = (int)can_send;

   if (nchannel == 0) {
      // special default channel for basic communications
      if ((cdata == "READY") && !conn->fReady) {
         if (fPanelName.length()) {
            // initialization not yet finished, appropriate panel should be started
            Send(std::string("SHOWPANEL:") + fPanelName, conn->fConnId);
            conn->fReady = 5;
         } else {
            fDataCallback(conn->fConnId, "CONN_READY");
            conn->fReady = 10;
         }
      }
   } else if (fPanelName.length() && (conn->fReady < 10)) {
      if (cdata == "PANEL_READY") {
         printf("Get panel ready %s !!!\n", fPanelName.c_str());
         fDataCallback(conn->fConnId, "CONN_READY");
         conn->fReady = 10;
      } else {
         fDataCallback(conn->fConnId, "CONN_CLOSED");
         fConn.erase(iter);
      }
   } else if (nchannel == 1) {
      fDataCallback(conn->fConnId, cdata);
   } else if (nchannel > 1) {
      conn->fCallBack(conn->fConnId, cdata);
   }

   CheckDataToSend();

   return true;
}

void ROOT::Experimental::TWebWindow::SendDataViaConnection(ROOT::Experimental::TWebWindow::WebConn &conn, int chid,
                                                           const std::string &data)
{
   if (!conn.fWSId || !fWSHandler)
      return;

   assert(conn.fSendCredits > 0 && "No credits to send data via connection");

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

   fWSHandler->SendCharStarWS(conn.fWSId, buf.c_str());
}

/// Check if data to any connection can be send
void ROOT::Experimental::TWebWindow::CheckDataToSend(bool only_once)
{
   bool isany = false;

   do {
      isany = false;

      for (auto iter = fConn.begin(); iter != fConn.end(); ++iter) {
         if (iter->fSendCredits <= 0)
            continue;

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

/// Returns relative address for the window to use when establish connection from the client

std::string ROOT::Experimental::TWebWindow::RelativeAddr(std::shared_ptr<TWebWindow> &win)
{
   if (fMgr != win->fMgr) {
      R__ERROR_HERE("RelativeAddr") << "Same web window manager should be used";
      return "";
   }

   std::string res("../");
   res.append(win->fWSHandler->GetName());
   res.append("/");
   return res;
}

// returns true if sending via specified connection can be performed
// if direct==true, requires that direct sending (without queue) is possible

bool ROOT::Experimental::TWebWindow::CanSend(unsigned connid, bool direct) const
{
   for (auto iter = fConn.begin(); iter != fConn.end(); ++iter) {
      if (connid && connid != iter->fConnId)
         continue;

      if (direct && ((iter->fQueue.size() > 0) || (iter->fSendCredits == 0)))
         return false;

      if (iter->fQueue.size() >= fMaxQueueLength)
         return false;
   }

   return true;
}

/// Sends data to specified connection
/// If connid==0, data will be send to all connections
void ROOT::Experimental::TWebWindow::Send(const std::string &data, unsigned connid, unsigned chid)
{
   for (auto iter = fConn.begin(); iter != fConn.end(); ++iter) {
      if (connid && connid != iter->fConnId)
         continue;

      if ((iter->fQueue.size() == 0) && (iter->fSendCredits > 0)) {
         SendDataViaConnection(*iter, chid, data);
      } else {
         assert(iter->fQueue.size() < fMaxQueueLength && "Maximum queue length achieved");
         iter->fQueue.push_back(std::to_string(chid) + ":" + data);
      }
   }

   CheckDataToSend();
}

/////////////////////////////////////////////////////////////////////////////////
/// Waits until provided check function returns non-zero value
/// Runs application mainloop in background
/// timelimit (in seconds) defines how long to wait (0 - for ever)

bool ROOT::Experimental::TWebWindow::WaitFor(WebWindowWaitFunc_t check, double timelimit)
{
   return fMgr->WaitFor(check, timelimit);
}
