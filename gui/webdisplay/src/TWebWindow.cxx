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

#include <ROOT/TWebWindow.hxx>

#include <ROOT/TWebWindowsManager.hxx>
#include <ROOT/TLogger.hxx>

#include "TWebWindowWSHandler.hxx"
#include "THttpCallArg.h"

#include <cstring>
#include <cstdlib>
#include <utility>

namespace ROOT {
namespace Experimental {


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
   if (fWSHandler)
      fWSHandler->SetDisabled();

   if (fMgr) {

      for (auto &&conn : GetConnections())
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
   if (!fWSHandler) {
      fSendMT = fMgr->IsUseSenderThreads();
      fWSHandler = std::make_shared<TWebWindowWSHandler>(*this, Form("win%u", GetId()));
   }
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

std::shared_ptr<ROOT::Experimental::TWebWindow::WebConn> ROOT::Experimental::TWebWindow::FindConnection(unsigned wsid, bool make_new)
{
   std::lock_guard<std::mutex> grd(fConnMutex);

   for (auto &&conn : fConn) {
      if (conn->fWSId == wsid)
         return conn;
   }

   // put code to create new connection here to stay under same locked mutex
   if (make_new)
      fConn.push_back(std::make_shared<WebConn>(++fConnCnt, wsid));

   return nullptr;
}

//////////////////////////////////////////////////////////////////////////////////////////
/// Remove connection with given websocket id

std::shared_ptr<ROOT::Experimental::TWebWindow::WebConn> ROOT::Experimental::TWebWindow::RemoveConnection(unsigned wsid)
{
   std::lock_guard<std::mutex> grd(fConnMutex);

   for (size_t n=0; n<fConn.size();++n)
      if (fConn[n]->fWSId == wsid) {
         std::shared_ptr<WebConn> res = std::move(fConn[n]);
         fConn.erase(fConn.begin() + n);
         res->fActive = false;
         return res;
      }

   return nullptr;
}

//////////////////////////////////////////////////////////////////////////////////////////
/// Provide data to user callback
/// User callback must be executed in the window thread

void ROOT::Experimental::TWebWindow::ProvideData(unsigned connid, std::string &&arg)
{
   {
      std::lock_guard<std::mutex> grd(fDataMutex);
      fDataQueue.emplace(connid, std::move(arg));
   }

   InvokeCallbacks();
}

//////////////////////////////////////////////////////////////////////////////////////////
/// Invoke callbacks with existing data
/// Must be called from appropriate thread

void ROOT::Experimental::TWebWindow::InvokeCallbacks(bool force)
{
   if ((fDataThrdId != std::this_thread::get_id()) && !force)
      return;

   while (fDataCallback) {
      std::string arg;
      unsigned connid;

      {
         std::lock_guard<std::mutex> grd(fDataMutex);
         if (fDataQueue.size() == 0)
            return;
         DataEntry &entry = fDataQueue.front();
         connid = entry.fConnId;
         arg = std::move(entry.fData);
         fDataQueue.pop();
      }

      fDataCallback(connid, arg);
   }
}


//////////////////////////////////////////////////////////////////////////////////////////
/// Processing of websockets call-backs, invoked from TWebWindowWSHandler
/// Method invoked from http server thread, therefore appropriate mutex must be used on all relevant data

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

      auto conn = FindConnection(arg.GetWSId(), true);

      if (conn) {
         R__ERROR_HERE("webgui") << "WSHandle with given websocket id " << arg.GetWSId() << " already exists";
         return false;
      }

      return true;
   }

   if (arg.IsMethod("WS_CLOSE")) {
      // connection is closed, one can remove handle

      auto conn = RemoveConnection(arg.GetWSId());

      if (conn) {
         ProvideData(conn->fConnId, "CONN_CLOSED");

         fMgr->HaltClient(conn->fProcId);
      }

      return true;
   }

   if (!arg.IsMethod("WS_DATA")) {
      R__ERROR_HERE("webgui") << "only WS_DATA request expected!";
      return false;
   }

   auto conn =  FindConnection(arg.GetWSId());

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

   {
      std::lock_guard<std::mutex> grd(conn->fMutex);

      conn->fSendCredits += ackn_oper;
      conn->fRecvCount++;
      conn->fClientCredits = (int)can_send;
   }

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
            ProvideData(conn->fConnId, "CONN_READY");
            conn->fReady = 10;
         }
      }
   } else if (fPanelName.length() && (conn->fReady < 10)) {
      if (cdata == "PANEL_READY") {
         R__DEBUG_HERE("webgui") << "Get panel ready " << fPanelName;
         ProvideData(conn->fConnId, "CONN_READY");
         conn->fReady = 10;
      } else {
         ProvideData(conn->fConnId, "CONN_CLOSED");
         RemoveConnection(conn->fWSId);
      }
   } else if (nchannel == 1) {
      ProvideData(conn->fConnId, std::move(cdata));
   } else if (nchannel > 1) {
      // add processing of extra channels later
      // conn->fCallBack(conn->fConnId, cdata);
   }

   CheckDataToSend();

   return true;
}

void ROOT::Experimental::TWebWindow::CompleteWSSend(unsigned wsid)
{
   auto conn = FindConnection(wsid);

   if (!conn)
      return;

   {
      std::lock_guard<std::mutex> grd(conn->fMutex);
      conn->fDoingSend = false;
   }

   CheckDataToSend(conn);
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

   if (conn->fDoingSend) {
      R__ERROR_HERE("webgui") << "Previous send operation not completed yet";
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
/// Checks if one should send data for specified connection
/// Returns true when send operation was performed

bool ROOT::Experimental::TWebWindow::CheckDataToSend(std::shared_ptr<WebConn> &conn)
{
   std::string hdr, data;

   {
      std::lock_guard<std::mutex> grd(conn->fMutex);

      if (!conn->fActive || (conn->fSendCredits <= 0) || conn->fDoingSend) return false;

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

      if (hdr.empty()) return false;

      conn->fDoingSend = true;
   }

   int res = 0;

   if (data.empty()) {
      res = fWSHandler->SendCharStarWS(conn->fWSId, hdr.c_str());
   } else {
      res = fWSHandler->SendHeaderWS(conn->fWSId, hdr.c_str(), data.data(), data.length());
   }

   // submit operation, will be processed
   if (res >=0) return true;


   // failure, clear sending flag
   std::lock_guard<std::mutex> grd(conn->fMutex);
   conn->fDoingSend = false;
   return false;
}


//////////////////////////////////////////////////////////////////////////////////////////
/// Checks if new data can be send (internal use only)
/// If necessary, provide credits to the client

void ROOT::Experimental::TWebWindow::CheckDataToSend(bool only_once)
{
   // make copy of all connections to be independent later
   auto arr = GetConnections();

   do {
      bool isany = false;

      for (auto &&conn : arr)
         if (CheckDataToSend(conn))
            isany = true;

      if (!isany) break;

   } while (!only_once);
}

///////////////////////////////////////////////////////////////////////////////////
/// Special method to process all internal activity when window runs in separate thread

void ROOT::Experimental::TWebWindow::Sync()
{
   InvokeCallbacks();

   CheckDataToSend();
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

      if (direct && (!conn->fQueue.empty() || (conn->fSendCredits == 0) || conn->fDoingSend))
         return false;

      if (conn->fQueue.size() >= fMaxQueueLength)
         return false;
   }

   return true;
}

///////////////////////////////////////////////////////////////////////////////////
/// returns send queue length for specified connection
/// if connid==0, maximal value for all connections is returned
/// If wrong connection is specified, -1 is return

int ROOT::Experimental::TWebWindow::GetSendQueueLength(unsigned connid)
{
   int maxq = -1;

   for (auto &&conn : GetConnections(connid)) {
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
      std::lock_guard<std::mutex> grd(conn->fMutex);

      if (conn->fQueue.size() < fMaxQueueLength) {
         if (--cnt)
            conn->fQueue.emplace(chid, txt, std::string(data)); // make copy
         else
            conn->fQueue.emplace(chid, txt, std::move(data));  // move content
      } else {
         R__ERROR_HERE("webgui") << "Maximum queue length achieved";
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
///
/// Function should have signature like void func(unsigned connid, const std::string &data)
/// First argument identifies connection (unique for each window), second argument is received data
/// There are predefined values for the data:
///     "CONN_READY"  - appears when new connection is established
///     "CONN_CLOSED" - when connection closed, no more data will be send/received via connection
///
/// At the moment when callback is assigned, TWebWindow working thread is detected.
/// If called not from main application thread, TWebWindow::Run() function must be regularly called from that thread.
///
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
   fDataThrdId = std::this_thread::get_id();
   if (!TWebWindowsManager::IsMainThrd()) {
      fProcessMT = true;
   } else if (fMgr->IsUseHttpThread()) {
      // special thread is used by the manager, but main thread used for the canvas - not supported
      R__ERROR_HERE("webgui") << "create web window from main thread when THttpServer created with special thread - not supported";
   }
}

/////////////////////////////////////////////////////////////////////////////////
/// Waits until provided check function or lambdas returns non-zero value
/// Check function has following signature: int func(double spent_tm)
/// Waiting will be continued, if function returns zero.
/// Parameter spent_tm is time in seconds, which already spent inside the function
/// First non-zero value breaks loop and result is returned.
/// Runs application mainloop and short sleeps in-between

int ROOT::Experimental::TWebWindow::WaitFor(WebWindowWaitFunc_t check)
{
   return fMgr->WaitFor(*this, check);
}

/////////////////////////////////////////////////////////////////////////////////
/// Waits until provided check function or lambdas returns non-zero value
/// Check function has following signature: int func(double spent_tm)
/// Waiting will be continued, if function returns zero.
/// Parameter spent_tm is time in seconds, which already spent inside the function
/// First non-zero value breaks waiting loop and result is returned (or 0 if time is expired).
/// Runs application mainloop and short sleeps in-between
/// timelimit (in seconds) defines how long to wait (if value <=0, WebGui.WaitForTmout parameter will be used)

int ROOT::Experimental::TWebWindow::WaitForTimed(WebWindowWaitFunc_t check, double timelimit)
{
   return fMgr->WaitFor(*this, check, true, timelimit);
}


/////////////////////////////////////////////////////////////////////////////////
/// Run window functionality for specified time
/// If no action can be performed - just sleep specified time

void ROOT::Experimental::TWebWindow::Run(double tm)
{
   if (fDataThrdId != std::this_thread::get_id()) {
      R__WARNING_HERE("webgui") << "Change thread id where TWebWindow is executed";
      fDataThrdId = std::this_thread::get_id();
   }

   if (tm <= 0) {
      Sync();
   } else {
      WaitForTimed([](double) { return 0; }, tm);
   }
}
