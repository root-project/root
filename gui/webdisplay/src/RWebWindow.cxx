// Author: Sergey Linev <s.linev@gsi.de>
// Date: 2017-10-16
// Warning: This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/RWebWindow.hxx>

#include <ROOT/RWebWindowsManager.hxx>
#include <ROOT/RLogger.hxx>

#include "RWebWindowWSHandler.hxx"
#include "THttpCallArg.h"
#include "TUrl.h"
#include "TError.h"
#include "TROOT.h"
#include "TSystem.h"

#include <cstring>
#include <cstdlib>
#include <utility>
#include <assert.h>
#include <algorithm>
#include <fstream>

// must be here because of defines
#include "../../../core/foundation/res/ROOT/RSha256.hxx"

using namespace ROOT;
using namespace std::string_literals;

//////////////////////////////////////////////////////////////////////////////////////////
/// Destructor for WebConn
/// Notify special HTTP request which blocks headless browser from exit

RWebWindow::WebConn::~WebConn()
{
   if (fHold) {
      fHold->SetTextContent("console.log('execute holder script');  if (window) setTimeout (window.close, 1000); if (window) window.close();");
      fHold->NotifyCondition();
      fHold.reset();
   }
}


std::string RWebWindow::gJSROOTsettings = "";


/** \class ROOT::RWebWindow
\ingroup webdisplay

Represents web window, which can be shown in web browser or any other supported environment

Window can be configured to run either in the normal or in the batch (headless) mode.
In second case no any graphical elements will be created. For the normal window one can configure geometry
(width and height), which are applied when window shown.

Each window can be shown several times (if allowed) in different places - either as the
CEF (chromium embedded) window or in the standard web browser. When started, window will open and show
HTML page, configured with RWebWindow::SetDefaultPage() method.

Typically (but not necessarily) clients open web socket connection to the window and one can exchange data,
using RWebWindow::Send() method and call-back function assigned via RWebWindow::SetDataCallBack().

*/


//////////////////////////////////////////////////////////////////////////////////////////
/// RWebWindow constructor
/// Should be defined here because of std::unique_ptr<RWebWindowWSHandler>

RWebWindow::RWebWindow()
{
   fRequireAuthKey = RWebWindowWSHandler::GetBoolEnv("WebGui.OnetimeKey", 1) == 1; // does authentication key really required
}

//////////////////////////////////////////////////////////////////////////////////////////
/// RWebWindow destructor
/// Closes all connections and remove window from manager

RWebWindow::~RWebWindow()
{
   StopThread();

   if (fMaster) {
      std::vector<MasterConn> lst;
      {
         std::lock_guard<std::mutex> grd(fConnMutex);
         std::swap(lst, fMasterConns);
      }

      for (auto &entry : lst)
         fMaster->RemoveEmbedWindow(entry.connid, entry.channel);
      fMaster.reset();
   }

   if (fWSHandler)
      fWSHandler->SetDisabled();

   if (fMgr) {

      // make copy of all connections
      auto lst = GetWindowConnections();

      {
         // clear connections vector under mutex
         std::lock_guard<std::mutex> grd(fConnMutex);
         fConn.clear();
         fPendingConn.clear();
      }

      for (auto &conn : lst) {
         conn->fActive = false;
         for (auto &elem: conn->fEmbed)
            elem.second->RemoveMasterConnection();
         conn->fEmbed.clear();
      }

      fMgr->Unregister(*this);
   }
}

//////////////////////////////////////////////////////////////////////////////////////////
/// Configure window to show some of existing JSROOT panels
/// It uses "file:rootui5sys/panel/panel.html" as default HTML page
/// At the moment only FitPanel is existing

void RWebWindow::SetPanelName(const std::string &name)
{
   {
      std::lock_guard<std::mutex> grd(fConnMutex);
      if (!fConn.empty()) {
         R__LOG_ERROR(WebGUILog()) << "Cannot configure panel when connection exists";
         return;
      }
   }

   fPanelName = name;
   SetDefaultPage("file:rootui5sys/panel/panel.html");
   if (fPanelName.find("localapp.") == 0)
      SetUseCurrentDir(true);
}

//////////////////////////////////////////////////////////////////////////////////////////
/// Assigns manager reference, window id and creates websocket handler, used for communication with the clients

std::shared_ptr<RWebWindowWSHandler>
RWebWindow::CreateWSHandler(std::shared_ptr<RWebWindowsManager> mgr, unsigned id, double tmout)
{
   fMgr = mgr;
   fId = id;
   fOperationTmout = tmout;

   fSendMT = fMgr->IsUseSenderThreads();
   fWSHandler = std::make_shared<RWebWindowWSHandler>(*this, Form("win%u", GetId()));

   return fWSHandler;
}

//////////////////////////////////////////////////////////////////////////////////////////
/// Return URL string to connect web window
/// URL typically includes extra parameters required for connection with the window like
/// `http://localhost:9635/win1/?key=<connection_key>#<session_key>`
/// When \param remote is true, real HTTP server will be started automatically and
/// widget can be connected from the web browser. If \param remote is false,
/// HTTP server will not be started and window can be connected only from ROOT application itself.
/// !!! WARNING - do not invoke this method without real need, each URL consumes resources in widget and in http server

std::string RWebWindow::GetUrl(bool remote)
{
   return fMgr->GetUrl(*this, remote);
}

//////////////////////////////////////////////////////////////////////////////////////////
/// Return THttpServer instance serving requests to the window

THttpServer *RWebWindow::GetServer()
{
   return fMgr->GetServer();
}

//////////////////////////////////////////////////////////////////////////////////////////
/// Show window in specified location
/// \see ROOT::RWebWindowsManager::Show for more info
/// \return (future) connection id (or 0 when fails)

unsigned RWebWindow::Show(const RWebDisplayArgs &args)
{
   return fMgr->ShowWindow(*this, args);
}

//////////////////////////////////////////////////////////////////////////////////////////
/// Start headless browser for specified window
/// Normally only single instance is used, but many can be created
/// See ROOT::RWebWindowsManager::Show() docu for more info
/// returns (future) connection id (or 0 when fails)

unsigned RWebWindow::MakeHeadless(bool create_new)
{
   unsigned connid = 0;
   if (!create_new)
      connid = FindHeadlessConnection();
   if (!connid) {
      RWebDisplayArgs args;
      args.SetHeadless(true);
      connid = fMgr->ShowWindow(*this, args);
   }
   return connid;
}

//////////////////////////////////////////////////////////////////////////////////////////
/// Returns connection id of window running in headless mode
/// This can be special connection which may run picture production jobs in background
/// Connection to that job may not be initialized yet
/// If connection does not exists, returns 0

unsigned RWebWindow::FindHeadlessConnection()
{
   std::lock_guard<std::mutex> grd(fConnMutex);

   for (auto &entry : fPendingConn) {
      if (entry->fHeadlessMode)
         return entry->fConnId;
   }

   for (auto &conn : fConn) {
      if (conn->fHeadlessMode)
         return conn->fConnId;
   }

   return 0;
}

//////////////////////////////////////////////////////////////////////////////////////////
/// Returns first connection id where window is displayed
/// It could be that connection(s) not yet fully established - but also not timed out
/// Batch jobs will be ignored here
/// Returns 0 if connection not exists

unsigned RWebWindow::GetDisplayConnection() const
{
   std::lock_guard<std::mutex> grd(fConnMutex);

   for (auto &entry : fPendingConn) {
      if (!entry->fHeadlessMode)
         return entry->fConnId;
   }

   for (auto &conn : fConn) {
      if (!conn->fHeadlessMode)
         return conn->fConnId;
   }

   return 0;
}

//////////////////////////////////////////////////////////////////////////////////////////
/// Find connection with given websocket id

std::shared_ptr<RWebWindow::WebConn> RWebWindow::FindConnection(unsigned wsid)
{
   std::lock_guard<std::mutex> grd(fConnMutex);

   for (auto &conn : fConn) {
      if (conn->fWSId == wsid)
         return conn;
   }

   return nullptr;
}

//////////////////////////////////////////////////////////////////////////////////////////
/// Signal that connection is closing

void RWebWindow::ClearConnection(std::shared_ptr<WebConn> &conn, bool provide_signal)
{
   if (!conn)
      return;

   if (provide_signal)
      ProvideQueueEntry(conn->fConnId, kind_Disconnect, ""s);
   for (auto &elem: conn->fEmbed) {
      if (provide_signal)
         elem.second->ProvideQueueEntry(conn->fConnId, kind_Disconnect, ""s);
      elem.second->RemoveMasterConnection(conn->fConnId);
   }

   conn->fEmbed.clear();
}

//////////////////////////////////////////////////////////////////////////////////////////
/// Remove connection with given websocket id

std::shared_ptr<RWebWindow::WebConn> RWebWindow::RemoveConnection(unsigned wsid, bool provide_signal)
{
   std::shared_ptr<WebConn> res;

   {
      std::lock_guard<std::mutex> grd(fConnMutex);

      for (size_t n = 0; n < fConn.size(); ++n)
         if (fConn[n]->fWSId == wsid) {
            res = std::move(fConn[n]);
            fConn.erase(fConn.begin() + n);
            res->fActive = false;
            res->fWasFirst = (n == 0);
            break;
         }
   }

   ClearConnection(res, provide_signal);

   return res;
}

//////////////////////////////////////////////////////////////////////////////////////////
/// Add new master connection
/// If there are many connections - only same master is allowed

void RWebWindow::AddMasterConnection(std::shared_ptr<RWebWindow> window, unsigned connid, int channel)
{
   if (fMaster && fMaster != window)
      R__LOG_ERROR(WebGUILog()) << "Cannot configure different masters at the same time";

   fMaster = window;

   std::lock_guard<std::mutex> grd(fConnMutex);

   fMasterConns.emplace_back(connid, channel);
}

//////////////////////////////////////////////////////////////////////////////////////////
/// Get list of master connections

std::vector<RWebWindow::MasterConn> RWebWindow::GetMasterConnections(unsigned connid) const
{
   std::vector<MasterConn> lst;
   if (!fMaster)
      return lst;

   std::lock_guard<std::mutex> grd(fConnMutex);

   for (auto & entry : fMasterConns)
      if (!connid || entry.connid == connid)
         lst.emplace_back(entry);

   return lst;
}

//////////////////////////////////////////////////////////////////////////////////////////
/// Remove master connection - if any

void RWebWindow::RemoveMasterConnection(unsigned connid)
{
   if (!fMaster) return;

   bool isany = false;

   {
      std::lock_guard<std::mutex> grd(fConnMutex);

      if (connid == 0) {
         fMasterConns.clear();
      } else {
         for (auto iter = fMasterConns.begin(); iter != fMasterConns.end(); ++iter)
            if (iter->connid == connid) {
               fMasterConns.erase(iter);
               break;
            }
      }

      isany = fMasterConns.size() > 0;
   }

   if (!isany)
      fMaster.reset();
}

//////////////////////////////////////////////////////////////////////////////////////////
/// Process special http request, used to hold headless browser running
/// Such requests should not be replied for the long time
/// Be aware that function called directly from THttpServer thread, which is not same thread as window

bool RWebWindow::ProcessBatchHolder(std::shared_ptr<THttpCallArg> &arg)
{
   std::string query = arg->GetQuery();

   if (query.compare(0, 4, "key=") != 0)
      return false;

   std::string key = query.substr(4);

   std::shared_ptr<THttpCallArg> prev;

   bool found_key = false;

   // use connection mutex to access hold request
   {
      std::lock_guard<std::mutex> grd(fConnMutex);
      for (auto &entry : fPendingConn) {
         if (entry->fKey == key)  {
            assert(!found_key); // indicate error if many same keys appears
            found_key = true;
            prev = std::move(entry->fHold);
            entry->fHold = arg;
         }
      }

      for (auto &conn : fConn) {
         if (conn->fKey == key) {
            assert(!found_key); // indicate error if many same keys appears
            prev = std::move(conn->fHold);
            conn->fHold = arg;
            found_key = true;
         }
      }
   }

   if (prev) {
      prev->SetTextContent("console.log('execute holder script'); if (window) window.close();");
      prev->NotifyCondition();
   }

   return found_key;
}

//////////////////////////////////////////////////////////////////////////////////////////
/// Provide data to user callback
/// User callback must be executed in the window thread

void RWebWindow::ProvideQueueEntry(unsigned connid, EQueueEntryKind kind, std::string &&arg)
{
   {
      std::lock_guard<std::mutex> grd(fInputQueueMutex);
      fInputQueue.emplace(connid, kind, std::move(arg));
   }

   // if special python mode is used, process events called from special thread
   // there is no other way to get regular calls in main python thread,
   // therefore invoke widgets callbacks directly - which potentially can be dangerous
   InvokeCallbacks(fUseProcessEvents);
}

//////////////////////////////////////////////////////////////////////////////////////////
/// Invoke callbacks with existing data
/// Must be called from appropriate thread

void RWebWindow::InvokeCallbacks(bool force)
{
   if (fCallbacksThrdIdSet && (fCallbacksThrdId != std::this_thread::get_id()) && !force)
      return;

   while (true) {
      unsigned connid;
      EQueueEntryKind kind;
      std::string arg;

      {
         std::lock_guard<std::mutex> grd(fInputQueueMutex);
         if (fInputQueue.size() == 0)
            return;
         auto &entry = fInputQueue.front();
         connid = entry.fConnId;
         kind = entry.fKind;
         arg = std::move(entry.fData);
         fInputQueue.pop();
      }

      switch (kind) {
      case kind_None: break;
      case kind_Connect:
         if (fConnCallback)
            fConnCallback(connid);
         break;
      case kind_Data:
         if (fDataCallback)
            fDataCallback(connid, arg);
         break;
      case kind_Disconnect:
         if (fDisconnCallback)
            fDisconnCallback(connid);
         break;
      }
   }
}

//////////////////////////////////////////////////////////////////////////////////////////
/// Add display handle and associated key
/// Key is large random string generated when starting new window
/// When client is connected, key should be supplied to correctly identify it

unsigned RWebWindow::AddDisplayHandle(bool headless_mode, const std::string &key, std::unique_ptr<RWebDisplayHandle> &handle)
{
   std::lock_guard<std::mutex> grd(fConnMutex);

   for (auto &entry : fPendingConn) {
      if (entry->fKey == key) {
         entry->fHeadlessMode = headless_mode;
         std::swap(entry->fDisplayHandle, handle);
         return entry->fConnId;
      }
   }

   auto conn = std::make_shared<WebConn>(++fConnCnt, headless_mode, key);

   std::swap(conn->fDisplayHandle, handle);

   fPendingConn.emplace_back(conn);

   return fConnCnt;
}


//////////////////////////////////////////////////////////////////////////////////////////
/// Check if provided hash, ntry parameters from the connection request could be accepted
/// \param hash - provided hash value which should match with HMAC hash for generated before connection key
/// \param ntry - connection attempt number provided together with request, must come in increasing order
/// \param remote - boolean flag indicating if request comming from remote (via real http),
///                 for local displays like Qt5 or CEF simpler connection rules are applied
/// \param test_first_time - true if hash/ntry tested for the first time, false appears only with
///                          websocket when connection accepted by server

bool RWebWindow::_CanTrustIn(std::shared_ptr<WebConn> &conn, const std::string &hash, const std::string &ntry, bool remote, bool test_first_time)
{
   if (!conn)
      return false;

   int intry = ntry.empty() ? -1 : std::stoi(ntry);

   auto msg = TString::Format("attempt_%s", ntry.c_str());
   auto expected = HMAC(conn->fKey, fMgr->fUseSessionKey && remote ? fMgr->fSessionKey : ""s, msg.Data(), msg.Length());

   if (!IsRequireAuthKey())
      return (conn->fKey.empty() && hash.empty()) || (hash == conn->fKey) || (hash == expected);

   // for local connection simple key can be used
   if (!remote && ((hash == conn->fKey) || (hash == expected)))
      return true;

   if (hash == expected) {
      if (test_first_time) {
         if (conn->fKeyUsed >= intry) {
            // this is indication of main in the middle, already checked hashed value was shown again!!!
            // client sends id with increasing counter, if previous value is presented it is BAD
            R__LOG_ERROR(WebGUILog()) << "Detect connection hash send before, possible replay attack!!!";
            return false;
         }
         // remember counter, it should prevent trying previous hash values
         conn->fKeyUsed = intry;
      } else {
         if (conn->fKeyUsed != intry) {
            // this is rather error condition, should never happen
            R__LOG_ERROR(WebGUILog()) << "Connection failure with HMAC signature check";
            return false;
         }
      }
      return true;
   }

   return false;
}


//////////////////////////////////////////////////////////////////////////////////////////
/// Returns true if provided key value already exists (in processes map or in existing connections)
/// In special cases one also can check if key value exists as newkey

bool RWebWindow::HasKey(const std::string &key, bool also_newkey) const
{
   if (key.empty())
      return false;

   std::lock_guard<std::mutex> grd(fConnMutex);

   for (auto &entry : fPendingConn) {
      if (entry->fKey == key)
         return true;
   }

   for (auto &conn : fConn) {
      if (conn->fKey == key)
         return true;
      if (also_newkey && (conn->fNewKey == key))
         return true;
   }

   return false;
}

//////////////////////////////////////////////////////////////////////////////////////////
/// Removes all connections with the key

void RWebWindow::RemoveKey(const std::string &key)
{
   ConnectionsList_t lst;

   {
      std::lock_guard<std::mutex> grd(fConnMutex);

      auto pred = [&](std::shared_ptr<WebConn> &e) {
         if (e->fKey == key) {
            lst.emplace_back(e);
            return true;
         }
         return false;
      };

      fPendingConn.erase(std::remove_if(fPendingConn.begin(), fPendingConn.end(), pred), fPendingConn.end());
      fConn.erase(std::remove_if(fConn.begin(), fConn.end(), pred), fConn.end());
   }

   for (auto &conn : lst)
      ClearConnection(conn, conn->fActive);
}


//////////////////////////////////////////////////////////////////////////////////////////
/// Generate new unique key for the window

std::string RWebWindow::GenerateKey() const
{
   auto key = RWebWindowsManager::GenerateKey(IsRequireAuthKey() ? 32 : 4);

   R__ASSERT((!IsRequireAuthKey() || (!HasKey(key) && (key != fMgr->fSessionKey))) && "Fail to generate window connection key");

   return key;
}

//////////////////////////////////////////////////////////////////////////////////////////
/// Check if started process(es) establish connection. After timeout such processed will be killed
/// Method invoked from http server thread, therefore appropriate mutex must be used on all relevant data

void RWebWindow::CheckPendingConnections()
{
   if (!fMgr) return;

   timestamp_t stamp = std::chrono::system_clock::now();

   float tmout = fMgr->GetLaunchTmout();

   ConnectionsList_t selected;

   {
      std::lock_guard<std::mutex> grd(fConnMutex);

      auto pred = [&](std::shared_ptr<WebConn> &e) {
         std::chrono::duration<double> diff = stamp - e->fSendStamp;

         if (diff.count() > tmout) {
            R__LOG_DEBUG(0, WebGUILog()) << "Remove pending connection " << e->fKey << " after " << diff.count() << " sec";
            selected.emplace_back(e);
            return true;
         }

         return false;
      };

      fPendingConn.erase(std::remove_if(fPendingConn.begin(), fPendingConn.end(), pred), fPendingConn.end());
   }
}


//////////////////////////////////////////////////////////////////////////////////////////
/// Check if there are connection which are inactive for longer time
/// For instance, batch browser will be stopped if no activity for 30 sec is there

void RWebWindow::CheckInactiveConnections()
{
   timestamp_t stamp = std::chrono::system_clock::now();

   double batch_tmout = 20.;

   std::vector<std::shared_ptr<WebConn>> clr;

   {
      std::lock_guard<std::mutex> grd(fConnMutex);

      auto pred = [&](std::shared_ptr<WebConn> &conn) {
         std::chrono::duration<double> diff = stamp - conn->fSendStamp;
         // introduce large timeout
         if ((diff.count() > batch_tmout) && conn->fHeadlessMode) {
            conn->fActive = false;
            clr.emplace_back(conn);
            return true;
         }
         return false;
      };

      fConn.erase(std::remove_if(fConn.begin(), fConn.end(), pred), fConn.end());
   }

   for (auto &entry : clr)
      ClearConnection(entry, true);
}

/////////////////////////////////////////////////////////////////////////
/// Configure maximal number of allowed connections - 0 is unlimited
/// Will not affect already existing connections
/// Default is 1 - the only client is allowed
/// Because of security reasons setting number of allowed connections is not sufficient now.
/// To enable multi-connection mode, one also has to call
/// `ROOT::RWebWindowsManager::SetSingleConnMode(false);`
/// before creating of the RWebWindow instance

void RWebWindow::SetConnLimit(unsigned lmt)
{
   bool single_conn_mode = RWebWindowWSHandler::GetBoolEnv("WebGui.SingleConnMode", 1) == 1;

   std::lock_guard<std::mutex> grd(fConnMutex);

   fConnLimit = single_conn_mode ? 1 : lmt;
}

/////////////////////////////////////////////////////////////////////////
/// returns configured connections limit (0 - default)

unsigned RWebWindow::GetConnLimit() const
{
   std::lock_guard<std::mutex> grd(fConnMutex);

   return fConnLimit;
}

/////////////////////////////////////////////////////////////////////////
/// Configures connection token (default none)
/// When specified, in URL of webpage such token should be provided as &token=value parameter,
/// otherwise web window will refuse connection

void RWebWindow::SetConnToken(const std::string &token)
{
   std::lock_guard<std::mutex> grd(fConnMutex);

   fConnToken = token;
}

/////////////////////////////////////////////////////////////////////////
/// Returns configured connection token

std::string RWebWindow::GetConnToken() const
{
   std::lock_guard<std::mutex> grd(fConnMutex);

   return fConnToken;
}

//////////////////////////////////////////////////////////////////////////////////////////
/// Processing of websockets call-backs, invoked from RWebWindowWSHandler
/// Method invoked from http server thread, therefore appropriate mutex must be used on all relevant data

bool RWebWindow::ProcessWS(THttpCallArg &arg)
{
   if (arg.GetWSId() == 0)
      return true;

   bool is_longpoll = arg.GetFileName() && ("root.longpoll"s == arg.GetFileName()),
        is_remote = arg.GetTopName() && ("remote"s == arg.GetTopName());

   // do not allow longpoll requests for loopback device
   if (is_longpoll && is_remote && RWebWindowsManager::IsLoopbackMode())
      return false;

   if (arg.IsMethod("WS_CONNECT")) {
      TUrl url;
      url.SetOptions(arg.GetQuery());
      std::string key, ntry;
      if(url.HasOption("key"))
         key = url.GetValueFromOptions("key");
      if(url.HasOption("ntry"))
         ntry = url.GetValueFromOptions("ntry");

      std::lock_guard<std::mutex> grd(fConnMutex);

      if (is_longpoll && !is_remote  && ntry == "1"s) {
         // special workaround for local displays like qt5/qt6
         // they are not disconnected regularly when page reload is invoked
         // therefore try to detect if new key is applied
         for (unsigned indx = 0; indx < fConn.size(); indx++) {
            if (!fConn[indx]->fNewKey.empty() && (key == HMAC(fConn[indx]->fNewKey, ""s, "attempt_1", 9))) {
               auto conn = std::move(fConn[indx]);
               fConn.erase(fConn.begin() + indx);
               conn->fKeyUsed = 0;
               conn->fKey = conn->fNewKey;
               conn->fNewKey.clear();
               conn->fConnId = ++fConnCnt; // change connection id to avoid confusion
               conn->fWasFirst = indx == 0;
               conn->ResetData();
               conn->ResetStamps(); // reset stamps, after timeout connection wll be removed
               fPendingConn.emplace_back(conn);
               break;
            }
         }
      }

      // refuse connection when number of connections exceed limit
      if (fConnLimit && (fConn.size() >= fConnLimit))
         return false;

      if (!fConnToken.empty()) {
         // refuse connection which does not provide proper token
         if (!url.HasOption("token") || (fConnToken != url.GetValueFromOptions("token"))) {
            R__LOG_DEBUG(0, WebGUILog()) << "Refuse connection without proper token";
            return false;
         }
      }

      if (!IsRequireAuthKey())
         return true;

      if(key.empty()) {
         R__LOG_DEBUG(0, WebGUILog()) << "key parameter not provided in url";
         return false;
      }

      for (auto &conn : fPendingConn)
         if (_CanTrustIn(conn, key, ntry, is_remote, true /* test_first_time */))
             return true;

      return false;
   }

   if (arg.IsMethod("WS_READY")) {

      if (FindConnection(arg.GetWSId())) {
         R__LOG_ERROR(WebGUILog()) << "WSHandle with given websocket id " << arg.GetWSId() << " already exists";
         return false;
      }

      std::shared_ptr<WebConn> conn;
      std::string key, ntry;

      TUrl url;
      url.SetOptions(arg.GetQuery());
      if (url.HasOption("key"))
         key = url.GetValueFromOptions("key");
      if (url.HasOption("ntry"))
         ntry = url.GetValueFromOptions("ntry");

      std::lock_guard<std::mutex> grd(fConnMutex);

      // check if in pending connections exactly this combination was checked
      for (size_t n = 0; n < fPendingConn.size(); ++n)
         if (_CanTrustIn(fPendingConn[n], key, ntry, is_remote, false /* test_first_time */)) {
            conn = std::move(fPendingConn[n]);
            fPendingConn.erase(fPendingConn.begin() + n);
            break;
         }

      if (conn) {
         conn->fWSId = arg.GetWSId();
         conn->fActive = true;
         conn->fRecvSeq = 0;
         conn->fSendSeq = 1;
         // preserve key for longpoll or when with session key used for HMAC hash of messages
         // conn->fKey.clear();
         conn->ResetStamps();
         if (conn->fWasFirst)
            fConn.emplace(fConn.begin(), conn);
         else
            fConn.emplace_back(conn);
         return true;
      } else if (!IsRequireAuthKey() && (!fConnLimit || (fConn.size() < fConnLimit))) {
         fConn.emplace_back(std::make_shared<WebConn>(++fConnCnt, arg.GetWSId()));
         return true;
      }

      // reject connection, should not really happen
      return false;
   }

   // special security check for the longpoll requests
   if(is_longpoll) {
      auto conn = FindConnection(arg.GetWSId());
      if (!conn)
         return false;

      TUrl url;
      url.SetOptions(arg.GetQuery());

      std::string key, ntry;
      if(url.HasOption("key"))
         key = url.GetValueFromOptions("key");
      if(url.HasOption("ntry"))
         ntry = url.GetValueFromOptions("ntry");

      if (!_CanTrustIn(conn, key, ntry, is_remote, true /* test_first_time */))
         return false;
   }

   if (arg.IsMethod("WS_CLOSE")) {
      // connection is closed, one can remove handle, associated window will be closed

      auto conn = RemoveConnection(arg.GetWSId(), true);

      if (conn) {
         bool do_clear_on_close = false;
         if (!conn->fNewKey.empty()) {
            // case when same handle want to be reused by client with new key
            std::lock_guard<std::mutex> grd(fConnMutex);
            conn->fKeyUsed = 0;
            conn->fKey = conn->fNewKey;
            conn->fNewKey.clear();
            conn->fConnId = ++fConnCnt; // change connection id to avoid confusion
            conn->ResetData();
            conn->ResetStamps(); // reset stamps, after timeout connection wll be removed
            fPendingConn.emplace_back(conn);
         } else {
            std::lock_guard<std::mutex> grd(fConnMutex);
            do_clear_on_close = (fPendingConn.size() == 0) && (fConn.size() == 0);
         }

         if (do_clear_on_close)
            fClearOnClose.reset();
      }

      return true;
   }

   if (!arg.IsMethod("WS_DATA")) {
      R__LOG_ERROR(WebGUILog()) << "only WS_DATA request expected!";
      return false;
   }

   auto conn = FindConnection(arg.GetWSId());

   if (!conn) {
      R__LOG_ERROR(WebGUILog()) << "Get websocket data without valid connection - ignore!!!";
      return false;
   }

   if (arg.GetPostDataLength() <= 0)
      return true;

   // here start testing of HMAC in the begin of the message

   const char *buf0 = (const char *) arg.GetPostData();
   Long_t data_len = arg.GetPostDataLength();

   const char *buf = strchr(buf0, ':');
   if (!buf) {
      R__LOG_ERROR(WebGUILog()) << "missing separator for HMAC checksum";
      return false;
   }

   Int_t code_len =  buf - buf0;
   data_len -= code_len + 1;
   buf++; // starting of normal message

   if (data_len < 0) {
      R__LOG_ERROR(WebGUILog()) << "no any data after HMAC checksum";
      return false;
   }

   bool is_none = strncmp(buf0, "none:", 5) == 0, is_match = false;

   if (!is_none) {
      std::string hmac = HMAC(conn->fKey, fMgr->fSessionKey, buf, data_len);

      is_match = (code_len == (Int_t) hmac.length()) && (strncmp(buf0, hmac.c_str(), code_len) == 0);
   } else if (!fMgr->fUseSessionKey) {
      // no packet signing without session key
      is_match = true;
   }

   // IMPORTANT: final place where integrity of input message is checked!
   if (!is_match) {
      // mismatch of HMAC checksum
      if (is_remote && IsRequireAuthKey())
         return false;
      if (!is_none) {
         R__LOG_ERROR(WebGUILog()) << "wrong HMAC checksum provided";
         return false;
      }
   }

   // here processing of received data should be performed
   // this is task for the implemented windows

   char *str_end = nullptr;

   unsigned long oper_seq = std::strtoul(buf, &str_end, 10);
   if (!str_end || *str_end != ':') {
      R__LOG_ERROR(WebGUILog()) << "missing operation sequence";
      return false;
   }

   if (is_remote && (oper_seq <= conn->fRecvSeq)) {
      R__LOG_ERROR(WebGUILog()) << "supply same package again - MiM attacker?";
      return false;
   }

   conn->fRecvSeq = oper_seq;

   unsigned long ackn_oper = std::strtoul(str_end + 1, &str_end, 10);
   if (!str_end || *str_end != ':') {
      R__LOG_ERROR(WebGUILog()) << "missing number of acknowledged operations";
      return false;
   }

   unsigned long can_send = std::strtoul(str_end + 1, &str_end, 10);
   if (!str_end || *str_end != ':') {
      R__LOG_ERROR(WebGUILog()) << "missing can_send counter";
      return false;
   }

   unsigned long nchannel = std::strtoul(str_end + 1, &str_end, 10);
   if (!str_end || *str_end != ':') {
      R__LOG_ERROR(WebGUILog()) << "missing channel number";
      return false;
   }

   Long_t processed_len = (str_end + 1 - buf);

   if (processed_len > data_len) {
      R__LOG_ERROR(WebGUILog()) << "corrupted buffer";
      return false;
   }

   std::string cdata(str_end + 1, data_len - processed_len);

   timestamp_t stamp = std::chrono::system_clock::now();

   {
      std::lock_guard<std::mutex> grd(conn->fMutex);

      conn->fSendCredits += ackn_oper;
      conn->fRecvCount++;
      conn->fClientCredits = (int)can_send;
      conn->fRecvStamp = stamp;
   }

   if (fProtocolCnt >= 0)
      if (!fProtocolConnId || (conn->fConnId == fProtocolConnId)) {
         fProtocolConnId = conn->fConnId; // remember connection

         // record send event only for normal channel or very first message via ch0
         if ((nchannel != 0) || (cdata.find("READY=") == 0)) {
            if (fProtocol.length() > 2)
               fProtocol.insert(fProtocol.length() - 1, ",");
            fProtocol.insert(fProtocol.length() - 1, "\"send\"");

            std::ofstream pfs(fProtocolFileName);
            pfs.write(fProtocol.c_str(), fProtocol.length());
            pfs.close();
         }
      }

   if (nchannel == 0) {
      // special system channel
      if ((cdata.compare(0, 6, "READY=") == 0) && !conn->fReady) {

         std::string key = cdata.substr(6);
         bool new_key = false;
         if (key.find("generate_key;") == 0) {
            new_key = true;
            key = key.substr(13);
         }

         if (key.empty() && IsNativeOnlyConn()) {
            RemoveConnection(conn->fWSId);
            return false;
         }

         if (!key.empty() && !conn->fKey.empty() && (conn->fKey != key)) {
            R__LOG_ERROR(WebGUILog()) << "Key mismatch after established connection " << key << " != " << conn->fKey;
            RemoveConnection(conn->fWSId);
            return false;
         }

         if (!fPanelName.empty()) {
            // initialization not yet finished, appropriate panel should be started
            Send(conn->fConnId, "SHOWPANEL:"s + fPanelName);
            conn->fReady = 5;
         } else {
            ProvideQueueEntry(conn->fConnId, kind_Connect, ""s);
            conn->fReady = 10;
         }
         if (new_key && !fMaster) {
            conn->fNewKey = GenerateKey();
            if(!conn->fNewKey.empty())
               SubmitData(conn->fConnId, true, "NEW_KEY="s + conn->fNewKey, 0);
         }
      } else if (cdata.compare(0, 8, "CLOSECH=") == 0) {
         int channel = std::stoi(cdata.substr(8));
         auto iter = conn->fEmbed.find(channel);
         if (iter != conn->fEmbed.end()) {
            iter->second->ProvideQueueEntry(conn->fConnId, kind_Disconnect, ""s);
            conn->fEmbed.erase(iter);
         }
      } else if (cdata.compare(0, 7, "RESIZE=") == 0) {
         auto p = cdata.find(",");
         if (p != std::string::npos) {
            auto width = std::stoi(cdata.substr(7, p - 7));
            auto height = std::stoi(cdata.substr(p + 1));
            if ((width > 0) && (height > 0) && conn->fDisplayHandle)
               conn->fDisplayHandle->Resize(width, height);
         }
      } else if (cdata == "GENERATE_KEY") {
         if (fMaster) {
            R__LOG_ERROR(WebGUILog()) << "Not able to generate new key with master connections";
         } else {
            conn->fNewKey = GenerateKey();
            if(!conn->fNewKey.empty())
               SubmitData(conn->fConnId, true, "NEW_KEY="s + conn->fNewKey, -1);
         }
      }
   } else if (fPanelName.length() && (conn->fReady < 10)) {
      if (cdata == "PANEL_READY") {
         R__LOG_DEBUG(0, WebGUILog()) << "Get panel ready " << fPanelName;
         ProvideQueueEntry(conn->fConnId, kind_Connect, ""s);
         conn->fReady = 10;
      } else {
         RemoveConnection(conn->fWSId, true);
      }
   } else if (nchannel == 1) {
      ProvideQueueEntry(conn->fConnId, kind_Data, std::move(cdata));
   } else if (nchannel > 1) {
      // process embed window
      auto embed_window = conn->fEmbed[nchannel];
      if (embed_window)
         embed_window->ProvideQueueEntry(conn->fConnId, kind_Data, std::move(cdata));
   }

   CheckDataToSend();

   return true;
}

//////////////////////////////////////////////////////////////////////////////////////////
/// Complete websocket send operation
/// Clear "doing send" flag and check if next operation has to be started

void RWebWindow::CompleteWSSend(unsigned wsid)
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
/// Internal method to prepare text part of send data
/// Should be called under locked connection mutex

std::string RWebWindow::_MakeSendHeader(std::shared_ptr<WebConn> &conn, bool txt, const std::string &data, int chid)
{
   std::string buf;

   if (!conn->fWSId || !fWSHandler) {
      R__LOG_ERROR(WebGUILog()) << "try to send text data when connection not established";
      return buf;
   }

   if (conn->fSendCredits <= 0) {
      R__LOG_ERROR(WebGUILog()) << "No credits to send text data via connection";
      return buf;
   }

   if (conn->fDoingSend) {
      R__LOG_ERROR(WebGUILog()) << "Previous send operation not completed yet";
      return buf;
   }

   if (txt)
      buf.reserve(data.length() + 100);

   buf.append(std::to_string(conn->fSendSeq++));
   buf.append(":");
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
   } else if (data.length()==0) {
      buf.append("$$nullbinary$$");
   } else {
      buf.append("$$binary$$");
      if (!conn->fKey.empty() && !fMgr->fSessionKey.empty() && fMgr->fUseSessionKey)
         buf.append(HMAC(conn->fKey, fMgr->fSessionKey, data.data(), data.length()));
   }

   return buf;
}

//////////////////////////////////////////////////////////////////////////////////////////
/// Checks if one should send data for specified connection
/// Returns true when send operation was performed

bool RWebWindow::CheckDataToSend(std::shared_ptr<WebConn> &conn)
{
   std::string hdr, data, prefix;

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
         hdr = _MakeSendHeader(conn, true, "KEEPALIVE", 0);
      }

      if (hdr.empty()) return false;

      conn->fDoingSend = true;
   }

   // add HMAC checksum for string send to client
   if (!conn->fKey.empty() && !fMgr->fSessionKey.empty() && fMgr->fUseSessionKey) {
      prefix = HMAC(conn->fKey, fMgr->fSessionKey, hdr.c_str(), hdr.length());
   } else {
      prefix = "none";
   }

   prefix += ":";
   hdr.insert(0, prefix);

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
/// \param only_once if true, data sending performed once or until there is no data to send

void RWebWindow::CheckDataToSend(bool only_once)
{
   // make copy of all connections to be independent later, only active connections are checked
   auto arr = GetWindowConnections(0, true);

   do {
      bool isany = false;

      for (auto &conn : arr)
         if (CheckDataToSend(conn))
            isany = true;

      if (!isany) break;

   } while (!only_once);
}

///////////////////////////////////////////////////////////////////////////////////
/// Special method to process all internal activity when window runs in separate thread

void RWebWindow::Sync()
{
   InvokeCallbacks();

   CheckDataToSend();

   CheckPendingConnections();

   CheckInactiveConnections();
}

///////////////////////////////////////////////////////////////////////////////////
/// Returns window address which is used in URL

std::string RWebWindow::GetAddr() const
{
    return fWSHandler->GetName();
}

///////////////////////////////////////////////////////////////////////////////////
/// DEPRECATED. Use GetUrl method instead while more arguments are required to connect with the widget
/// Returns relative URL address for the specified window
/// Address can be required if one needs to access data from one window into another window
/// Used for instance when inserting panel into canvas

std::string RWebWindow::GetRelativeAddr(const std::shared_ptr<RWebWindow> &win) const
{
   if (fMgr != win->fMgr) {
      R__LOG_ERROR(WebGUILog()) << "Same web window manager should be used";
      return "";
   }

   std::string res("../");
   res.append(win->GetAddr());
   res.append("/");
   return res;
}

///////////////////////////////////////////////////////////////////////////////////
/// DEPRECATED. Use GetUrl method instead while more arguments are required to connect with the widget
/// Address can be required if one needs to access data from one window into another window
/// Used for instance when inserting panel into canvas

std::string RWebWindow::GetRelativeAddr(const RWebWindow &win) const
{
   if (fMgr != win.fMgr) {
      R__LOG_ERROR(WebGUILog()) << "Same web window manager should be used";
      return "";
   }

   std::string res("../");
   res.append(win.GetAddr());
   res.append("/");
   return res;
}

/////////////////////////////////////////////////////////////////////////
/// Set client version, used as prefix in scripts URL
/// When changed, web browser will reload all related JS files while full URL will be different
/// Default is empty value - no extra string in URL
/// Version should be string like "1.2" or "ver1.subv2" and not contain any special symbols

void RWebWindow::SetClientVersion(const std::string &vers)
{
   std::lock_guard<std::mutex> grd(fConnMutex);
   fClientVersion = vers;
}

/////////////////////////////////////////////////////////////////////////
/// Returns current client version

std::string RWebWindow::GetClientVersion() const
{
   std::lock_guard<std::mutex> grd(fConnMutex);
   return fClientVersion;
}

/////////////////////////////////////////////////////////////////////////
/// Set arbitrary JSON data, which is accessible via conn.getUserArgs() method in JavaScript
/// This JSON code injected into main HTML document into connectWebWindow({})
/// Must be set before RWebWindow::Show() method is called
/// \param args - arbitrary JSON data which can be provided to client side

void RWebWindow::SetUserArgs(const std::string &args)
{
   std::lock_guard<std::mutex> grd(fConnMutex);
   fUserArgs = args;
}

/////////////////////////////////////////////////////////////////////////
/// Returns configured user arguments for web window
/// See \ref SetUserArgs method for more details

std::string RWebWindow::GetUserArgs() const
{
   std::lock_guard<std::mutex> grd(fConnMutex);
   return fUserArgs;
}

///////////////////////////////////////////////////////////////////////////////////
/// Returns current number of active clients connections
/// \param with_pending if true, also pending (not yet established) connection accounted

int RWebWindow::NumConnections(bool with_pending) const
{
   bool is_master = !!fMaster;

   std::lock_guard<std::mutex> grd(fConnMutex);

   if (is_master)
      return fMasterConns.size();

   auto sz = fConn.size();
   if (with_pending)
      sz += fPendingConn.size();
   return sz;
}

///////////////////////////////////////////////////////////////////////////////////
/// Configures recording of communication data in protocol file
/// Provided filename will be used to store JSON array with names of written files - text or binary
/// If data was send from client, "send" entry will be placed. JSON file will look like:
///
///      ["send", "msg0.txt", "send", "msg1.txt", "msg2.txt"]
///
/// If empty file name is provided, data recording will be disabled
/// Recorded data can be used in JSROOT directly to test client code without running C++ server

void RWebWindow::RecordData(const std::string &fname, const std::string &fprefix)
{
   fProtocolFileName = fname;
   fProtocolCnt = fProtocolFileName.empty() ? -1 : 0;
   fProtocolConnId = fProtocolFileName.empty() ? 0 : GetConnectionId(0);
   fProtocolPrefix = fprefix;
   fProtocol = "[]"; // empty array
}

///////////////////////////////////////////////////////////////////////////////////
/// Returns connection id for specified connection sequence number
/// Only active connections are returned - where clients confirms connection
/// Total number of connections can be retrieved with NumConnections() method
/// \param num connection sequence number

unsigned RWebWindow::GetConnectionId(int num) const
{
   bool is_master = !!fMaster;

   std::lock_guard<std::mutex> grd(fConnMutex);

   if (is_master)
      return (num >= 0) && (num < (int)fMasterConns.size()) ? fMasterConns[num].connid : 0;

   return ((num >= 0) && (num < (int)fConn.size()) && fConn[num]->fActive) ? fConn[num]->fConnId : 0;
}

///////////////////////////////////////////////////////////////////////////////////
/// returns vector with all existing connections ids
/// One also can exclude specified connection from return result,
/// which can be useful to be able reply too all but this connections

std::vector<unsigned> RWebWindow::GetConnections(unsigned excludeid) const
{
   std::vector<unsigned> res;

   bool is_master = !!fMaster;

   std::lock_guard<std::mutex> grd(fConnMutex);

   if (is_master) {
      for (auto & entry : fMasterConns)
         if (entry.connid != excludeid)
            res.emplace_back(entry.connid);
   } else {
      for (auto & entry : fConn)
         if (entry->fActive && (entry->fConnId != excludeid))
            res.emplace_back(entry->fConnId);
   }

   return res;
}

///////////////////////////////////////////////////////////////////////////////////
/// returns true if specified connection id exists
/// \param connid       connection id (0 - any)
/// \param only_active  when true only active connection will be checked, otherwise also pending (not yet established) connections are checked

bool RWebWindow::HasConnection(unsigned connid, bool only_active) const
{
   std::lock_guard<std::mutex> grd(fConnMutex);

   for (auto &conn : fConn) {
      if (connid && (conn->fConnId != connid))
         continue;
      if (conn->fActive || !only_active)
         return true;
   }

   if (!only_active)
      for (auto &conn : fPendingConn) {
         if (!connid || (conn->fConnId == connid))
            return true;
      }

   return false;
}

///////////////////////////////////////////////////////////////////////////////////
/// Closes all connection to clients
/// Normally leads to closing of all correspondent browser windows
/// Some browsers (like firefox) do not allow by default to close window

void RWebWindow::CloseConnections()
{
   SubmitData(0, true, "CLOSE", 0);
}

///////////////////////////////////////////////////////////////////////////////////
/// Close specified connection
/// \param connid  connection id, when 0 - all connections will be closed

void RWebWindow::CloseConnection(unsigned connid)
{
   if (connid)
      SubmitData(connid, true, "CLOSE", 0);
}

///////////////////////////////////////////////////////////////////////////////////
/// returns connection list (or all active connections)
/// \param connid  connection id, when 0 - all existing connections are returned
/// \param only_active  when true, only active (already established) connections are returned

RWebWindow::ConnectionsList_t RWebWindow::GetWindowConnections(unsigned connid, bool only_active) const
{
   ConnectionsList_t arr;

   {
      std::lock_guard<std::mutex> grd(fConnMutex);

      for (auto &conn : fConn) {
         if ((conn->fActive || !only_active) && (!connid || (conn->fConnId == connid)))
            arr.push_back(conn);
      }

      if (!only_active)
         for (auto &conn : fPendingConn)
            if (!connid || (conn->fConnId == connid))
               arr.push_back(conn);
   }

   return arr;
}

///////////////////////////////////////////////////////////////////////////////////
/// Returns true if sending via specified connection can be performed
/// \param connid  connection id, when 0 - all existing connections are checked
/// \param direct  when true, checks if direct sending (without queuing) is possible

bool RWebWindow::CanSend(unsigned connid, bool direct) const
{
   auto arr = GetWindowConnections(connid, direct); // for direct sending connection has to be active

   auto maxqlen = GetMaxQueueLength();

   for (auto &conn : arr) {

      std::lock_guard<std::mutex> grd(conn->fMutex);

      if (direct && (!conn->fQueue.empty() || (conn->fSendCredits == 0) || conn->fDoingSend))
         return false;

      if (conn->fQueue.size() >= maxqlen)
         return false;
   }

   return true;
}

///////////////////////////////////////////////////////////////////////////////////
/// Returns send queue length for specified connection
/// \param connid  connection id, 0 - maximal value for all connections is returned
/// If wrong connection id specified, -1 is return

int RWebWindow::GetSendQueueLength(unsigned connid) const
{
   int maxq = -1;

   for (auto &conn : GetWindowConnections(connid)) {
      std::lock_guard<std::mutex> grd(conn->fMutex);
      int len = conn->fQueue.size();
      if (len > maxq) maxq = len;
   }

   return maxq;
}

///////////////////////////////////////////////////////////////////////////////////
/// Internal method to send data
/// \param connid  connection id, when 0 - data will be send to all connections
/// \param txt  is text message that should be sent
/// \param data  data to be std-moved to SubmitData function
/// \param chid  channel id, 1 - normal communication, 0 - internal with highest priority

void RWebWindow::SubmitData(unsigned connid, bool txt, std::string &&data, int chid)
{
   if (fMaster) {
      auto lst = GetMasterConnections(connid);
      auto cnt = lst.size();
      for (auto & entry : lst)
         if (--cnt)
            fMaster->SubmitData(entry.connid, txt, std::string(data), entry.channel);
         else
            fMaster->SubmitData(entry.connid, txt, std::move(data), entry.channel);
      return;
   }

   auto arr = GetWindowConnections(connid);
   auto cnt = arr.size();
   auto maxqlen = GetMaxQueueLength();

   bool clear_queue = false;

   if (chid == -1) {
      chid = 0;
      clear_queue = true;
   }

   timestamp_t stamp = std::chrono::system_clock::now();

   for (auto &conn : arr) {

      if ((fProtocolCnt >= 0) && (chid > 0))
         if (!fProtocolConnId || (conn->fConnId == fProtocolConnId)) {
            fProtocolConnId = conn->fConnId; // remember connection
            std::string fname = fProtocolPrefix;
            fname.append("msg");
            fname.append(std::to_string(fProtocolCnt++));
            if (chid > 1) {
               fname.append("_ch");
               fname.append(std::to_string(chid));
            }
            fname.append(txt ? ".txt" : ".bin");

            std::ofstream ofs(fname);
            ofs.write(data.c_str(), data.length());
            ofs.close();

            if (fProtocol.length() > 2)
               fProtocol.insert(fProtocol.length() - 1, ",");
            fProtocol.insert(fProtocol.length() - 1, "\""s + fname + "\""s);

            std::ofstream pfs(fProtocolFileName);
            pfs.write(fProtocol.c_str(), fProtocol.length());
            pfs.close();
         }

      conn->fSendStamp = stamp;

      std::lock_guard<std::mutex> grd(conn->fMutex);

      if (clear_queue) {
         while (!conn->fQueue.empty())
            conn->fQueue.pop();
      }

      if (conn->fQueue.size() < maxqlen) {
         if (--cnt)
            conn->fQueue.emplace(chid, txt, std::string(data)); // make copy
         else
            conn->fQueue.emplace(chid, txt, std::move(data));  // move content
      } else {
         R__LOG_ERROR(WebGUILog()) << "Maximum queue length achieved";
      }
   }

   CheckDataToSend();
}

///////////////////////////////////////////////////////////////////////////////////
/// Sends data to specified connection
/// \param connid  connection id, when 0 - data will be send to all connections
/// \param data  data to be copied to SubmitData function

void RWebWindow::Send(unsigned connid, const std::string &data)
{
   SubmitData(connid, true, std::string(data), 1);
}

///////////////////////////////////////////////////////////////////////////////////
/// Send binary data to specified connection
/// \param connid  connection id, when 0 - data will be send to all connections
/// \param data  data to be std-moved to SubmitData function

void RWebWindow::SendBinary(unsigned connid, std::string &&data)
{
   SubmitData(connid, false, std::move(data), 1);
}

///////////////////////////////////////////////////////////////////////////////////
/// Send binary data to specified connection
/// \param connid  connection id, when 0 - data will be send to all connections
/// \param data  pointer to binary data
/// \param len number of bytes in data

void RWebWindow::SendBinary(unsigned connid, const void *data, std::size_t len)
{
   std::string buf;
   buf.resize(len);
   std::copy((const char *)data, (const char *)data + len, buf.begin());
   SubmitData(connid, false, std::move(buf), 1);
}

///////////////////////////////////////////////////////////////////////////////////
/// Assign thread id which has to be used for callbacks
/// WARNING!!!  only for expert use
/// Automatically done at the moment when any callback function is invoked
/// Can be invoked once again if window Run method will be invoked from other thread
/// Normally should be invoked before Show() method is called

void RWebWindow::AssignThreadId()
{
   fUseServerThreads = false;
   fUseProcessEvents = false;
   fProcessMT = false;
   fCallbacksThrdIdSet = true;
   fCallbacksThrdId = std::this_thread::get_id();
   if (!RWebWindowsManager::IsMainThrd()) {
      fProcessMT = true;
   } else if (fMgr->IsUseHttpThread()) {
      // special thread is used by the manager, but main thread used for the canvas - not supported
      R__LOG_ERROR(WebGUILog()) << "create web window from main thread when THttpServer created with special thread - not supported";
   }
}

/////////////////////////////////////////////////////////////////////////////////
/// Let use THttpServer threads to process requests
/// WARNING!!! only for expert use
/// Should be only used when application provides proper locking and
/// does not block. Such mode provides minimal possible latency
/// Must be called before callbacks are assigned

void RWebWindow::UseServerThreads()
{
   fUseServerThreads = true;
   fUseProcessEvents = false;
   fCallbacksThrdIdSet = false;
   fProcessMT = true;
}

/////////////////////////////////////////////////////////////////////////////////
/// Start special thread which will be used by the window to handle all callbacks
/// One has to be sure, that access to global ROOT structures are minimized and
/// protected with ROOT::EnableThreadSafety(); call

void RWebWindow::StartThread()
{
   if (fHasWindowThrd) {
      R__LOG_WARNING(WebGUILog()) << "thread already started for the window";
      return;
   }

   fHasWindowThrd = true;

   std::thread thrd([this] {
      AssignThreadId();
      while(fHasWindowThrd)
         Run(0.1);
      fCallbacksThrdIdSet = false;
   });

   fWindowThrd = std::move(thrd);
}

/////////////////////////////////////////////////////////////////////////////////
/// Stop special thread

void RWebWindow::StopThread()
{
   if (!fHasWindowThrd)
      return;

   fHasWindowThrd = false;
   fWindowThrd.join();
}


/////////////////////////////////////////////////////////////////////////////////
/// Set call-back function for data, received from the clients via websocket
///
/// Function should have signature like void func(unsigned connid, const std::string &data)
/// First argument identifies connection (unique for each window), second argument is received data
///
/// At the moment when callback is assigned, RWebWindow working thread is detected.
/// If called not from main application thread, RWebWindow::Run() function must be regularly called from that thread.
///
/// Most simple way to assign call-back - use of c++11 lambdas like:
/// ~~~ {.cpp}
/// auto win = RWebWindow::Create();
/// win->SetDefaultPage("file:./page.htm");
/// win->SetDataCallBack(
///          [](unsigned connid, const std::string &data) {
///                  printf("Conn:%u data:%s\n", connid, data.c_str());
///           }
///       );
/// win->Show();
/// ~~~

void RWebWindow::SetDataCallBack(WebWindowDataCallback_t func)
{
   if (!fUseServerThreads && !fUseProcessEvents)
      AssignThreadId();
   fDataCallback = func;
}

/////////////////////////////////////////////////////////////////////////////////
/// Set call-back function for new connection

void RWebWindow::SetConnectCallBack(WebWindowConnectCallback_t func)
{
   if (!fUseServerThreads && !fUseProcessEvents)
      AssignThreadId();
   fConnCallback = func;
}

/////////////////////////////////////////////////////////////////////////////////
/// Set call-back function for disconnecting

void RWebWindow::SetDisconnectCallBack(WebWindowConnectCallback_t func)
{
   if (!fUseServerThreads && !fUseProcessEvents)
      AssignThreadId();
   fDisconnCallback = func;
}

/////////////////////////////////////////////////////////////////////////////////
/// Set handle which is cleared when last active connection is closed
/// Typically can be used to destroy web-based widget at such moment

void RWebWindow::SetClearOnClose(const std::shared_ptr<void> &handle)
{
   fClearOnClose = handle;
}

/////////////////////////////////////////////////////////////////////////////////
/// Set call-backs function for connect, data and disconnect events

void RWebWindow::SetCallBacks(WebWindowConnectCallback_t conn, WebWindowDataCallback_t data, WebWindowConnectCallback_t disconn)
{
   if (!fUseServerThreads && !fUseProcessEvents)
      AssignThreadId();
   fConnCallback = conn;
   fDataCallback = data;
   fDisconnCallback = disconn;
}

/////////////////////////////////////////////////////////////////////////////////
/// Waits until provided check function or lambdas returns non-zero value
/// Check function has following signature: int func(double spent_tm)
/// Waiting will be continued, if function returns zero.
/// Parameter spent_tm is time in seconds, which already spent inside the function
/// First non-zero value breaks loop and result is returned.
/// Runs application mainloop and short sleeps in-between

int RWebWindow::WaitFor(WebWindowWaitFunc_t check)
{
   return fMgr->WaitFor(*this, check);
}

/////////////////////////////////////////////////////////////////////////////////
/// Waits until provided check function or lambdas returns non-zero value
/// Check function has following signature: int func(double spent_tm)
/// Waiting will be continued, if function returns zero.
/// Parameter spent_tm in lambda is time in seconds, which already spent inside the function
/// First non-zero value breaks waiting loop and result is returned (or 0 if time is expired).
/// Runs application mainloop and short sleeps in-between
/// WebGui.OperationTmout rootrc parameter defines waiting time in seconds

int RWebWindow::WaitForTimed(WebWindowWaitFunc_t check)
{
   return fMgr->WaitFor(*this, check, true, GetOperationTmout());
}

/////////////////////////////////////////////////////////////////////////////////
/// Waits until provided check function or lambdas returns non-zero value
/// Check function has following signature: int func(double spent_tm)
/// Waiting will be continued, if function returns zero.
/// Parameter spent_tm in lambda is time in seconds, which already spent inside the function
/// First non-zero value breaks waiting loop and result is returned (or 0 if time is expired).
/// Runs application mainloop and short sleeps in-between
/// duration (in seconds) defines waiting time

int RWebWindow::WaitForTimed(WebWindowWaitFunc_t check, double duration)
{
   return fMgr->WaitFor(*this, check, true, duration);
}


/////////////////////////////////////////////////////////////////////////////////
/// Run window functionality for specified time
/// If no action can be performed - just sleep specified time

void RWebWindow::Run(double tm)
{
   if (!fCallbacksThrdIdSet || (fCallbacksThrdId != std::this_thread::get_id())) {
      R__LOG_WARNING(WebGUILog()) << "Change thread id where RWebWindow is executed";
      fCallbacksThrdIdSet = true;
      fCallbacksThrdId = std::this_thread::get_id();
   }

   if (tm <= 0) {
      Sync();
   } else {
      WaitForTimed([](double) { return 0; }, tm);
   }
}


/////////////////////////////////////////////////////////////////////////////////
/// Add embed window

unsigned RWebWindow::AddEmbedWindow(std::shared_ptr<RWebWindow> window, unsigned connid, int channel)
{
   if (channel < 2)
      return 0;

   auto arr = GetWindowConnections(connid, true);
   if (arr.size() == 0)
      return 0;

   // check if channel already occupied
   if (arr[0]->fEmbed.find(channel) != arr[0]->fEmbed.end())
      return 0;

   arr[0]->fEmbed[channel] = window;

   return arr[0]->fConnId;
}

/////////////////////////////////////////////////////////////////////////////////
/// Remove RWebWindow associated with the channel

void RWebWindow::RemoveEmbedWindow(unsigned connid, int channel)
{
   auto arr = GetWindowConnections(connid);

   for (auto &conn : arr) {
      auto iter = conn->fEmbed.find(channel);
      if (iter != conn->fEmbed.end())
         conn->fEmbed.erase(iter);
   }
}


/////////////////////////////////////////////////////////////////////////////////
/// Create new RWebWindow
/// Using default RWebWindowsManager

std::shared_ptr<RWebWindow> RWebWindow::Create()
{
   return RWebWindowsManager::Instance()->CreateWindow();
}

/////////////////////////////////////////////////////////////////////////////////
/// Terminate ROOT session
/// Tries to correctly close THttpServer, associated with RWebWindowsManager
/// After that exit from process

void RWebWindow::TerminateROOT()
{

   // workaround to release all connection-specific handles as soon as possible
   // required to work with QWebEngine
   // once problem solved, can be removed here
   ConnectionsList_t arr1, arr2;

   {
      std::lock_guard<std::mutex> grd(fConnMutex);
      std::swap(arr1, fConn);
      std::swap(arr2, fPendingConn);
   }

   fMgr->Terminate();
}

/////////////////////////////////////////////////////////////////////////////////
/// Static method to show web window
/// Has to be used instead of RWebWindow::Show() when window potentially can be embed into other windows
/// Soon RWebWindow::Show() method will be done protected

unsigned RWebWindow::ShowWindow(std::shared_ptr<RWebWindow> window, const RWebDisplayArgs &args)
{
   if (!window)
      return 0;

   if (args.GetBrowserKind() == RWebDisplayArgs::kEmbedded) {
      auto master = args.fMaster;
      while (master && master->fMaster)
         master = master->fMaster;

      if (master && window->fMaster && window->fMaster != master) {
         R__LOG_ERROR(WebGUILog()) << "Cannot use different master for same RWebWindow";
         return 0;
      }

      unsigned connid = master ? master->AddEmbedWindow(window, args.fMasterConnection, args.fMasterChannel) : 0;

      if (connid > 0) {

         window->RemoveMasterConnection(connid);

         window->AddMasterConnection(master, connid, args.fMasterChannel);

         // inform client that connection is established and window initialized
         master->SubmitData(connid, true, "EMBED_DONE"s, args.fMasterChannel);

         // provide call back for window itself that connection is ready
         window->ProvideQueueEntry(connid, kind_Connect, ""s);
      }

      return connid;
   }

   return window->Show(args);
}

std::function<bool(const std::shared_ptr<RWebWindow> &, unsigned, const std::string &)> RWebWindow::gStartDialogFunc = nullptr;

/////////////////////////////////////////////////////////////////////////////////////
/// Configure func which has to be used for starting dialog


void RWebWindow::SetStartDialogFunc(std::function<bool(const std::shared_ptr<RWebWindow> &, unsigned, const std::string &)> func)
{
   gStartDialogFunc = func;
}

/////////////////////////////////////////////////////////////////////////////////////
/// Check if this could be the message send by client to start new file dialog
/// If returns true, one can call RWebWindow::EmbedFileDialog() to really create file dialog
/// instance inside existing widget

bool RWebWindow::IsFileDialogMessage(const std::string &msg)
{
   return msg.compare(0, 11, "FILEDIALOG:") == 0;
}

/////////////////////////////////////////////////////////////////////////////////////
/// Create dialog instance to use as embedded dialog inside provided widget
/// Loads libROOTBrowserv7 and tries to call RFileDialog::Embedded() method
/// Embedded dialog started on the client side where FileDialogController.SaveAs() method called
/// Such method immediately send message with "FILEDIALOG:" prefix
/// On the server side widget should detect such message and call RFileDialog::Embedded()
/// providing received string as second argument.
/// Returned instance of shared_ptr<RFileDialog> may be used to assign callback when file is selected

bool RWebWindow::EmbedFileDialog(const std::shared_ptr<RWebWindow> &window, unsigned connid, const std::string &args)
{
   if (!gStartDialogFunc)
      gSystem->Load("libROOTBrowserv7");

   if (!gStartDialogFunc)
      return false;

   return gStartDialogFunc(window, connid, args);
}

/////////////////////////////////////////////////////////////////////////////////////
/// Calculate HMAC checksum for provided key and message
/// Key combined from connection key and session key

std::string RWebWindow::HMAC(const std::string &key, const std::string &sessionKey, const char *msg, int msglen)
{
   using namespace ROOT::Internal::SHA256;

   auto get_digest = [](sha256_t &hash, bool as_hex = false) -> std::string {
      std::string digest;
      digest.resize(32);

      sha256_final(&hash, reinterpret_cast<unsigned char *>(digest.data()));

      if (!as_hex) return digest;

      static const char* digits = "0123456789abcdef";
      std::string hex;
      for (int n = 0; n < 32; n++) {
         unsigned char code = (unsigned char) digest[n];
         hex += digits[code / 16];
         hex += digits[code % 16];
      }
      return hex;
   };

   // calculate hash of sessionKey + key;
   sha256_t hash1;
   sha256_init(&hash1);
   sha256_update(&hash1, (const unsigned char *) sessionKey.data(), sessionKey.length());
   sha256_update(&hash1, (const unsigned char *) key.data(), key.length());
   std::string kbis = get_digest(hash1);

   kbis.resize(64, 0); // resize to blocksize 64 bytes required by the sha256

   std::string ki = kbis, ko = kbis;
   const int opad = 0x5c;
   const int ipad = 0x36;
   for (size_t i = 0; i < kbis.length(); ++i) {
      ko[i] = kbis[i] ^ opad;
      ki[i] = kbis[i] ^ ipad;
   }

   // calculate hash for ko + msg;
   sha256_t hash2;
   sha256_init(&hash2);
   sha256_update(&hash2, (const unsigned char *) ki.data(), ki.length());
   sha256_update(&hash2, (const unsigned char *) msg, msglen);
   std::string m2digest = get_digest(hash2);

   // calculate hash for ki + m2_digest;
   sha256_t hash3;
   sha256_init(&hash3);
   sha256_update(&hash3, (const unsigned char *) ko.data(), ko.length());
   sha256_update(&hash3, (const unsigned char *) m2digest.data(), m2digest.length());

   return get_digest(hash3, true);
}

/////////////////////////////////////////////////////////////////////////////////////
/// Set JSROOT settings as json string
/// Will be applied for any web window at the connection time
/// Can be used to chang `settings` object of JSROOT like:
/// ~~~ {.cpp}
/// ROOT::RWebWindow::SetJSROOTSettings("{ ToolBar: false, CanEnlarge: false }");
/// ~~~

void RWebWindow::SetJSROOTSettings(const std::string &json)
{
   gJSROOTsettings = json;
}
