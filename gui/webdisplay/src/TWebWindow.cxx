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

#pragma GCC diagnostic ignored "-Wall"
#pragma GCC diagnostic ignored "-Wunused-parameter"

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

/// just wrapper to deliver websockets call-backs to the TWebWindow class

class TWebWindowWSHandler : public THttpWSHandler {
public:
   TWebWindow &fDispl; ///<! display reference

   /// constructor
   TWebWindowWSHandler(TWebWindow &displ, const char *name)
      : THttpWSHandler(name, "TWebWindow websockets handler"), fDispl(displ)
   {
   }

   /// returns content of default web-page
   /// THttpWSHandler interface
   virtual TString GetDefaultPageContent() override { return fDispl.fDefaultPage.c_str(); }

   /// Process websocket request
   /// THttpWSHandler interface
   virtual Bool_t ProcessWS(THttpCallArg *arg) override { return arg ? fDispl.ProcessWS(*arg) : kFALSE; }
};

} // namespace Experimental
} // namespace ROOT

/** \class ROOT::Experimental::TWebWindow
\ingroup webdisplay

Represents web window, which can be shown in web browser or any other supported environment

Window can be configured to run either in the normal or in the batch (headless) mode.
In second case no any graphical elements will be created. For the normal window one can configure geometry
(width and height), which are applied when window shown.

Each window can be shown several times (if allowed) in different places - either as the
CEF (chromium embdedded) window or in the standard web browser. When started, window will open and show
HTML page, configured with TWebWindow::SetDefaultPage() method.

Typically (but not necessarily) clients open web socket connection to the window and one can exchange data,
using TWebWindow::Send() method and call-back function assigned via TWebWindow::SetDataCallBack().

*/

//////////////////////////////////////////////////////////////////////////////////////////
/// TWebWindow constructor
/// Should be defined here because of std::unique_ptr<TWebWindowWSHandler>

ROOT::Experimental::TWebWindow::TWebWindow() = default;

//////////////////////////////////////////////////////////////////////////////////////////
/// TWebWindow destructor
/// Closes all connections and remove window from manager

ROOT::Experimental::TWebWindow::~TWebWindow()
{
   fConn.clear();

   if (fMgr)
      fMgr->Unregister(*this);
}

//////////////////////////////////////////////////////////////////////////////////////////
/// Configure window to show some of existing JSROOT panels
/// It uses "file:$jsrootsys/files/panel.htm" as default HTML page
/// At the moment only FitPanel is existing

void ROOT::Experimental::TWebWindow::SetPanelName(const std::string &name)
{
   assert(fConn.size() == 0 && "Cannot configure panel when connection exists");

   fPanelName = name;

   SetDefaultPage("file:$jsrootsys/files/panel.htm");
}

//////////////////////////////////////////////////////////////////////////////////////////
/// Creates websocket handler, used for communication with the clients

void ROOT::Experimental::TWebWindow::CreateWSHandler()
{
   if (!fWSHandler)
      fWSHandler = std::make_unique<TWebWindowWSHandler>(*this, Form("win%u", GetId()));
}

//////////////////////////////////////////////////////////////////////////////////////////
/// Show window in specified location
/// See ROOT::Experimental::TWebWindowsManager::Show() docu for more info

bool ROOT::Experimental::TWebWindow::Show(const std::string &where)
{
   bool res = fMgr->Show(*this, where);
   if (res)
      fShown = true;
   return res;
}

//////////////////////////////////////////////////////////////////////////////////////////
/// Processing of websockets call-backs, invoked from TWebWindowWSHandler

bool ROOT::Experimental::TWebWindow::ProcessWS(THttpCallArg &arg)
{
   if (arg.GetWSId() == 0)
      return kTRUE;

   // try to identify connection for given WS request
   WebConn *conn = 0;
   auto iter = fConn.begin();
   while (iter != fConn.end()) {
      if (iter->fWSId == arg.GetWSId()) {
         conn = &(*iter);
         break;
      }
      ++iter;
   }

   if (arg.IsMethod("WS_CONNECT")) {

      // refuse connection when limit exceed limit
      if (fConnLimit && (fConn.size() >= fConnLimit))
         return false;

      return true;
   }

   if (arg.IsMethod("WS_READY")) {
      assert(conn == 0 && "WSHandle with given websocket id exists!!!");

      WebConn newconn;
      newconn.fWSId = arg.GetWSId();
      newconn.fConnId = ++fConnCnt; // unique connection id
      fConn.push_back(newconn);

      // CheckDataToSend();

      return true;
   }

   if (arg.IsMethod("WS_CLOSE")) {
      // connection is closed, one can remove handle

      if (conn && fDataCallback)
         fDataCallback(conn->fConnId, "CONN_CLOSED");

      if (conn)
         fConn.erase(iter);

      return true;
   }

   assert(arg.IsMethod("WS_DATA") && "WS_DATA request expected!");

   assert(conn != 0 && "Get websocket data without valid connection - ignore!!!");

   if (arg.GetPostDataLength() <= 0)
      return true;

   // here processing of received data should be performed
   // this is task for the implemented windows

   const char *buf = (const char *)arg.GetPostData();
   char *str_end = 0;

   printf("Get portion of data %d %.30s\n", (int)arg.GetPostDataLength(), buf);

   unsigned long ackn_oper = std::strtoul(buf, &str_end, 10);
   assert(str_end != 0 && *str_end == ':' && "missing number of acknowledged operations");

   unsigned long can_send = std::strtoul(str_end + 1, &str_end, 10);
   assert(str_end != 0 && *str_end == ':' && "missing can_send counter");

   unsigned long nchannel = std::strtoul(str_end + 1, &str_end, 10);
   assert(str_end != 0 && *str_end == ':' && "missing channel number");

   unsigned processed_len = (str_end + 1 - buf);

   assert(processed_len <= arg.GetPostDataLength() && "corrupted buffer");

   std::string cdata(str_end + 1, arg.GetPostDataLength() - processed_len);

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

//////////////////////////////////////////////////////////////////////////////////////////
/// Sends data via specified connection (internal use only)
/// Takes care about message prefix and account for send/recv credits

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

//////////////////////////////////////////////////////////////////////////////////////////
/// Checks if new data can be send (internal use only)
/// If necessary, provide credits to the client

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

///////////////////////////////////////////////////////////////////////////////////
/// Returns relative URL address for the specified window
/// Address can be requried if one needs to access data from one window into another window
/// Used for instance when inserting panel into canvas

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

///////////////////////////////////////////////////////////////////////////////////
/// returns true if sending via specified connection can be performed
/// if direct==true, checks if direct sending (without queuing) is possible
/// if connid==0, all existing connections are checked

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

///////////////////////////////////////////////////////////////////////////////////
/// Sends data to specified connection/channel
/// If connid==0, data will be send to all connections
/// Normally chid==1 (chid==0 is reserved for internal communications)

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
/// Set call-back function for data, received from the clients via websocket
/// Function should have signature like void func(unsigned connid, const std::string &data)
/// First argument identifies connection (unique for each window), second argument is received data
/// There are predefined values for the data:
///     "CONN_READY"  - appears when new connection is established
///     "CONN_CLOSED" - when connection closed, no more data will be send/received via connection
/// Most simple way to assign call-back - use of c++11 lambdas like:
/// ~~~ {.cpp}
/// std::shared_ptr<TWebWindow> win = TWebWindowsManager::Instance()->CreateWindow();
/// win->SetDefaultPage("file:./page.htm");
/// win->SetDataCallBack([](unsigned connid, const std::string &data) { printf("Conn:%u data:%s\n", connid,
/// data.c_str()); });
/// win->Show("opera");
/// ~~~

void ROOT::Experimental::TWebWindow::SetDataCallBack(WebWindowDataCallback_t func)
{
   fDataCallback = func;
}

/////////////////////////////////////////////////////////////////////////////////
/// Waits until provided check function or lambdas returns non-zero value
/// Runs application mainloop and short sleeps in-between
/// timelimit (in seconds) defines how long to wait (0 - forever)
/// Function has following signature: int func(double spent_tm)
/// Parameter spent_tm is time in seconds, which already spent inside function
/// Waiting will be continued, if function returns zero.
/// First non-zero value breaks waiting loop and result is returned (or 0 if time is expired).

int ROOT::Experimental::TWebWindow::WaitFor(WebWindowWaitFunc_t check, double timelimit)
{
   return fMgr->WaitFor(check, timelimit);
}
