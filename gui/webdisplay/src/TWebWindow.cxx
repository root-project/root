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

#include <cstring>
#include <cstdlib>
#include <utility>

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
CEF (chromium embedded) window or in the standard web browser. When started, window will open and show
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
   if (fMgr) {

      auto arr = GetConnections();

      for (auto &&conn : arr)
         if (conn->fActive) {
            conn->fActive = false;
            fMgr->HaltClient(conn->fProcId);
         }

      {
         std::lock_guard<std::mutex> grd(fConnMutex);
         fConn.clear(); // remove all connections
      }

      fMgr->Unregister(*this);
   }
}

//////////////////////////////////////////////////////////////////////////////////////////
/// Configure window to show some of existing JSROOT panels
/// It uses "file:$jsrootsys/files/panel.htm" as default HTML page
/// At the moment only FitPanel is existing

void ROOT::Experimental::TWebWindow::SetPanelName(const std::string &name)
{
   {
      std::lock_guard<std::mutex> grd(fConnMutex);
      if (!fConn.empty()) {
         R__ERROR_HERE("webgui") << "Cannot configure panel when connection exists";
         return;
      }
   }

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
/// Return URL string to access web window
/// If remote flag is specified, real HTTP server will be started automatically

std::string ROOT::Experimental::TWebWindow::GetUrl(bool remote)
{
   return fMgr->GetUrl(*this, remote);
}

//////////////////////////////////////////////////////////////////////////////////////////
/// Return THttpServer instance serving requests to the window

THttpServer *ROOT::Experimental::TWebWindow::GetServer()
{
   return fMgr->GetServer();
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
/// Find connection with given websocket id
/// Connection mutex should be locked before method calling

std::shared_ptr<ROOT::Experimental::TWebWindow::WebConn> ROOT::Experimental::TWebWindow::_FindConnection(unsigned wsid)
{
   // std::lock_guard<std::mutex> grd(fConnMutex);

   for (auto &&conn : fConn)
      if (conn->fWSId == wsid)
         return conn;

   return nullptr;
}

//////////////////////////////////////////////////////////////////////////////////////////
/// Remove connection with given websocket id

std::shared_ptr<ROOT::Experimental::TWebWindow::WebConn> ROOT::Experimental::TWebWindow::RemoveConnection(unsigned wsid)
{
   std::lock_guard<std::mutex> grd(fConnMutex);

   for (size_t n=0; n<fConn.size();++n)
      if (fConn[n]->fWSId == wsid) {
         auto res = std::move(fConn[n]);
         fConn.erase(fConn.begin() + n);
         res->fActive = false;
         return res;
      }

   return nullptr;
}

//////////////////////////////////////////////////////////////////////////////////////////
/// Processing of websockets call-backs, invoked from TWebWindowWSHandler

bool ROOT::Experimental::TWebWindow::ProcessWS(THttpCallArg &arg)
{
   if (arg.GetWSId() == 0)
      return true;

   if (arg.IsMethod("WS_CONNECT")) {

      std::lock_guard<std::mutex> grd(fConnMutex);

      // refuse connection when number of connections exceed limit
      if (fConnLimit && (fConn.size() >= fConnLimit))
         return false;

      return true;
   }

   if (arg.IsMethod("WS_READY")) {

      // mutex should remain locked longer to ensure that no other connection with given id appears
      std::lock_guard<std::mutex> grd(fConnMutex);

      auto conn = _FindConnection(arg.GetWSId());

      if (conn) {
         R__ERROR_HERE("webgui") << "WSHandle with given websocket id " << arg.GetWSId() << " already exists";
         return false;
      }

      // first value is unique connection id inside window
      fConn.push_back(std::make_shared<WebConn>(++fConnCnt, arg.GetWSId()));

      // CheckDataToSend();

      return true;
   }

   if (arg.IsMethod("WS_CLOSE")) {
      // connection is closed, one can remove handle

      auto conn = RemoveConnection(arg.GetWSId());

      if (conn) {
         if (fDataCallback)
            fDataCallback(conn->fConnId, "CONN_CLOSED");

         fMgr->HaltClient(conn->fProcId);
      }

      return true;
   }

   if (!arg.IsMethod("WS_DATA")) {
      R__ERROR_HERE("webgui") << "only WS_DATA request expected!";
      return false;
   }

   std::shared_ptr<WebConn> conn;

   {
      // probably mutex should remain locked longer to ensure that
      std::lock_guard<std::mutex> grd(fConnMutex);

      conn = _FindConnection(arg.GetWSId());
   }

   if (!conn) {
      R__ERROR_HERE("webgui") << "Get websocket data without valid connection - ignore!!!";
      return false;
   }

   if (arg.GetPostDataLength() <= 0)
      return true;

   // here processing of received data should be performed
   // this is task for the implemented windows

   const char *buf = (const char *)arg.GetPostData();
   char *str_end = nullptr;

   // printf("Get portion of data %d %.30s\n", (int)arg.GetPostDataLength(), buf);

   unsigned long ackn_oper = std::strtoul(buf, &str_end, 10);
   if (!str_end || *str_end != ':') {
      R__ERROR_HERE("webgui") << "missing number of acknowledged operations";
      return false;
   }

   unsigned long can_send = std::strtoul(str_end + 1, &str_end, 10);
   if (!str_end || *str_end != ':') {
      R__ERROR_HERE("webgui") << "missing can_send counter";
      return false;
   }

   unsigned long nchannel = std::strtoul(str_end + 1, &str_end, 10);
   if (!str_end || *str_end != ':') {
      R__ERROR_HERE("webgui") << "missing channel number";
      return false;
   }

   unsigned processed_len = (str_end + 1 - buf);

   if (processed_len > arg.GetPostDataLength()) {
      R__ERROR_HERE("webgui") << "corrupted buffer";
      return false;
   }

   std::string cdata(str_end + 1, arg.GetPostDataLength() - processed_len);

   conn->fSendCredits += ackn_oper;
   conn->fRecvCount++;
   conn->fClientCredits = (int)can_send;

   if (nchannel == 0) {
      // special system channel
      if ((cdata.find("READY=") == 0) && !conn->fReady) {
         std::string key = cdata.substr(6);

         if (!HasKey(key) && IsNativeOnlyConn()) {
            if (conn) RemoveConnection(conn->fWSId);

            return false;
         }

         if (HasKey(key)) {
            conn->fProcId = fKeys[key];
            R__DEBUG_HERE("webgui") << "Find key " << key << " for process " << conn->fProcId;
            fKeys.erase(key);
         }

         if (fPanelName.length()) {
            // initialization not yet finished, appropriate panel should be started
            Send(conn->fConnId, std::string("SHOWPANEL:") + fPanelName);
            conn->fReady = 5;
         } else {
            fDataCallback(conn->fConnId, "CONN_READY");
            conn->fReady = 10;
         }
      }
   } else if (fPanelName.length() && (conn->fReady < 10)) {
      if (cdata == "PANEL_READY") {
         R__DEBUG_HERE("webgui") << "Get panel ready " << fPanelName;
         fDataCallback(conn->fConnId, "CONN_READY");
         conn->fReady = 10;
      } else {
         fDataCallback(conn->fConnId, "CONN_CLOSED");
         RemoveConnection(conn->fWSId);
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
/// Prepare text part of send data
/// Should be called under locked connection mutex

std::string ROOT::Experimental::TWebWindow::_MakeSendHeader(std::shared_ptr<WebConn> &conn, bool txt, const std::string &data, int chid)
{
   std::string buf;

   if (!conn->fWSId || !fWSHandler) {
      R__ERROR_HERE("webgui") << "try to send text data when connection not established";
      return buf;
   }

   if (conn->fSendCredits <= 0) {
      R__ERROR_HERE("webgui") << "No credits to send text data via connection";
      return buf;
   }

   if (txt)
      buf.reserve(data.length() + 100);

   buf.append(std::to_string(conn->fRecvCount));
   buf.append(":");
   buf.append(std::to_string(conn->fSendCredits));
   buf.append(":");
   conn->fRecvCount = 0; // we confirm how many packages was received
   conn->fSendCredits--;

   buf.append(std::to_string(chid));
   buf.append(":");

   if (txt) {
      buf.append(data);
   } else {
      buf.append("$$binary$$");
   }
   return buf;
}


//////////////////////////////////////////////////////////////////////////////////////////
/// Checks if new data can be send (internal use only)
/// If necessary, provide credits to the client

void ROOT::Experimental::TWebWindow::CheckDataToSend(bool only_once)
{
   bool isany = false;

   // make copy of all connections to be independent later
   auto arr = GetConnections();

   std::string hdr, data;

   do {
      isany = false;

      for (auto &&conn : arr) {
         if (!conn->fActive || (conn->fSendCredits <= 0))
            continue;

         if (!conn->fQueue.empty()) {
            QueueItem &item = conn->fQueue.front();
            hdr = _MakeSendHeader(conn, item.fText, item.fData, item.fChID);
            if (!hdr.empty() && !item.fText)
               data = std::move(item.fData);
            conn->fQueue.pop();
         } else if ((conn->fClientCredits < 3) && (conn->fRecvCount > 1)) {
            // give more credits to the client
            R__DEBUG_HERE("webgui") << "Send keep alive to client";
            hdr = _MakeSendHeader(conn, true, "KEEPALIVE", 0);
         }

         if (!hdr.empty()) {
            isany = true;

            if (data.empty()) {
               fWSHandler->SendCharStarWS(conn->fWSId, hdr.c_str());
            } else {
               fWSHandler->SendHeaderWS(conn->fWSId, hdr.c_str(), data.data(), data.length());
               data.clear();
            }
            hdr.clear();
         }
      }

   } while (isany && !only_once);
}

///////////////////////////////////////////////////////////////////////////////////
/// Returns relative URL address for the specified window
/// Address can be required if one needs to access data from one window into another window
/// Used for instance when inserting panel into canvas

std::string ROOT::Experimental::TWebWindow::RelativeAddr(std::shared_ptr<TWebWindow> &win)
{
   if (fMgr != win->fMgr) {
      R__ERROR_HERE("WebDisplay") << "Same web window manager should be used";
      return "";
   }

   std::string res("../");
   res.append(win->fWSHandler->GetName());
   res.append("/");
   return res;
}

/// Returns current number of active clients connections
int ROOT::Experimental::TWebWindow::NumConnections()
{
   std::lock_guard<std::mutex> grd(fConnMutex);
   return fConn.size();
}

///////////////////////////////////////////////////////////////////////////////////
/// returns connection for specified connection number
/// Total number of connections can be retrieved with NumConnections() method

unsigned ROOT::Experimental::TWebWindow::GetConnectionId(int num)
{
   std::lock_guard<std::mutex> grd(fConnMutex);
   if (num>=(int)fConn.size() || !fConn[num]->fActive) return 0;
   return fConn[num]->fConnId;
}

///////////////////////////////////////////////////////////////////////////////////
/// Closes all connection to clients
/// Normally leads to closing of all correspondent browser windows
/// Some browsers (like firefox) do not allow by default to close window

void ROOT::Experimental::TWebWindow::CloseConnections()
{
   SubmitData(0, true, "CLOSE", 0);
}

///////////////////////////////////////////////////////////////////////////////////
/// Close specified connection
/// Connection id usually appears in the correspondent call-backs

void ROOT::Experimental::TWebWindow::CloseConnection(unsigned connid)
{
   if (connid)
      SubmitData(connid, true, "CLOSE", 0);
}

///////////////////////////////////////////////////////////////////////////////////
/// returns connection (or all active connections)

std::vector<std::shared_ptr<ROOT::Experimental::TWebWindow::WebConn>> ROOT::Experimental::TWebWindow::GetConnections(unsigned connid)
{
   std::vector<std::shared_ptr<WebConn>> arr;

   std::lock_guard<std::mutex> grd(fConnMutex);

   if (!connid) {
      arr = fConn;
   } else {
      for (auto &&conn : fConn)
         if ((conn->fConnId == connid) && conn->fActive)
            arr.push_back(conn);
   }

   return arr;
}


///////////////////////////////////////////////////////////////////////////////////
/// returns true if sending via specified connection can be performed
/// if direct==true, checks if direct sending (without queuing) is possible
/// if connid==0, all existing connections are checked

bool ROOT::Experimental::TWebWindow::CanSend(unsigned connid, bool direct)
{
   auto arr = GetConnections(connid);

   for (auto &&conn : arr) {

      std::lock_guard<std::mutex> grd(conn->fMutex);

      if (direct && (!conn->fQueue.empty() || (conn->fSendCredits == 0)))
         return false;

      if (conn->fQueue.size() >= fMaxQueueLength)
         return false;
   }

   return true;
}

///////////////////////////////////////////////////////////////////////////////////
/// returns send queue length for specified connection
/// if connid==0, maximal value for all connections are returned
/// If wrong connection is specified, -1 is return

int ROOT::Experimental::TWebWindow::SendQueueLength(unsigned connid)
{
   auto arr = GetConnections(connid);

   int maxq = -1;

   for (auto &&conn : arr) {
      std::lock_guard<std::mutex> grd(conn->fMutex);
      int len = conn->fQueue.size();
      if (len > maxq) maxq = len;
   }

   return maxq;
}


///////////////////////////////////////////////////////////////////////////////////
/// Internal method to send data
/// Allows to specify channel. chid==1 is normal communication, chid==0 for internal with higher priority
/// If connid==0, data will be send to all connections

void ROOT::Experimental::TWebWindow::SubmitData(unsigned connid, bool txt, std::string &&data, int chid)
{
   auto arr = GetConnections(connid);

   auto cnt = arr.size();

   for (auto &&conn : arr) {

      std::string sendhdr, senddata;

      {
         std::lock_guard<std::mutex> grd(conn->fMutex);

         if (conn->fQueue.empty() && (conn->fSendCredits > 0)) {
            sendhdr = _MakeSendHeader(conn, txt, data, chid);
            if (!sendhdr.empty() && !txt) {
               if (--cnt)
                  senddata = std::string(data);
               else
                  senddata = std::move(data);
            }
         } else if (conn->fQueue.size() < fMaxQueueLength) {
            if (--cnt)
               conn->fQueue.emplace(chid, txt, std::string(data)); // make copy
            else
               conn->fQueue.emplace(chid, txt, std::move(data)); // move content
         } else {
            R__ERROR_HERE("webgui") << "Maximum queue length achieved";
         }
      }

      // move out code out of locking area
      if (!sendhdr.empty()) {

         if (senddata.empty()) {
            fWSHandler->SendCharStarWS(conn->fWSId, sendhdr.c_str());
         } else {
            fWSHandler->SendHeaderWS(conn->fWSId, sendhdr.c_str(), senddata.data(), senddata.length());
            senddata.clear();
         }
         sendhdr.clear();
      }
   }

   CheckDataToSend();
}

///////////////////////////////////////////////////////////////////////////////////
/// Sends data to specified connection
/// If connid==0, data will be send to all connections

void ROOT::Experimental::TWebWindow::Send(unsigned connid, const std::string &data)
{
   SubmitData(connid, true, std::string(data), 1);
}

///////////////////////////////////////////////////////////////////////////////////
/// Send binary data to specified connection
/// If connid==0, data will be sent to all connections

void ROOT::Experimental::TWebWindow::SendBinary(unsigned connid, std::string &&data)
{
   SubmitData(connid, false, std::move(data), 1);
}

///////////////////////////////////////////////////////////////////////////////////
/// Send binary data to specified connection
/// If connid==0, data will be sent to all connections

void ROOT::Experimental::TWebWindow::SendBinary(unsigned connid, const void *data, std::size_t len)
{
   std::string buf;
   buf.resize(len);
   std::copy((const char *)data, (const char *)data + len, buf.begin());
   SubmitData(connid, false, std::move(buf), 1);
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
/// win->SetDataCallBack(
///          [](unsigned connid, const std::string &data) {
///                  printf("Conn:%u data:%s\n", connid, data.c_str());
///           }
///       );
/// win->Show("opera");
/// ~~~

void ROOT::Experimental::TWebWindow::SetDataCallBack(WebWindowDataCallback_t func)
{
   fDataCallback = func;
}

/////////////////////////////////////////////////////////////////////////////////
/// Waits until provided check function or lambdas returns non-zero value
/// Runs application mainloop and short sleeps in-between
/// timelimit (in seconds) defines how long to wait (0 - forever, negative - default value)
/// Function has following signature: int func(double spent_tm)
/// Parameter spent_tm is time in seconds, which already spent inside function
/// Waiting will be continued, if function returns zero.
/// First non-zero value breaks waiting loop and result is returned (or 0 if time is expired).

int ROOT::Experimental::TWebWindow::WaitFor(WebWindowWaitFunc_t check, double timelimit)
{
   return fMgr->WaitFor(check, timelimit);
}
