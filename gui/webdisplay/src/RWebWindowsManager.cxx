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

#include <ROOT/RWebWindowsManager.hxx>

#include <ROOT/RLogger.hxx>
#include <ROOT/RWebDisplayArgs.hxx>
#include <ROOT/RWebDisplayHandle.hxx>

#include "RWebWindowWSHandler.hxx"

#include "THttpServer.h"

#include "TSystem.h"
#include "TString.h"
#include "TApplication.h"
#include "TTimer.h"
#include "TRandom3.h"
#include "TError.h"
#include "TROOT.h"
#include "TEnv.h"
#include "TExec.h"
#include "TSocket.h"
#include "TThread.h"
#include "TObjArray.h"

#include <thread>
#include <chrono>
#include <iostream>

using namespace ROOT;

///////////////////////////////////////////////////////////////
/// Parse boolean gEnv variable which should be "yes" or "no"
/// \return 1 for true or 0 for false
/// Returns \param dflt if result is not defined
/// \param name name of the env variable

int RWebWindowWSHandler::GetBoolEnv(const std::string &name, int dflt)
{
   const char *undef = "<undefined>";
   const char *value = gEnv->GetValue(name.c_str(), undef);
   if (!value) return dflt;
   std::string svalue = value;
   if (svalue == undef) return dflt;

   if (svalue == "yes") return 1;
   if (svalue == "no") return 0;

   R__LOG_ERROR(WebGUILog()) << name << " has to be yes or no";
   return dflt;
}


/** \class ROOT::RWebWindowsManager
\ingroup webdisplay

Central instance to create and show web-based windows like Canvas or FitPanel.

Manager responsible to creating THttpServer instance, which is used for RWebWindow's
communication with clients.

Method RWebWindows::Show() used to show window in specified location.
*/

//////////////////////////////////////////////////////////////////////////////////////////
/// Returns default window manager
/// Used to display all standard ROOT elements like TCanvas or TFitPanel

std::shared_ptr<RWebWindowsManager> &RWebWindowsManager::Instance()
{
   static std::shared_ptr<RWebWindowsManager> sInstance = std::make_shared<RWebWindowsManager>();
   return sInstance;
}

//////////////////////////////////////////////////////////////////
/// This thread id used to identify main application thread, where ROOT event processing runs
/// To inject code in that thread, one should use TTimer (like THttpServer does)
/// In other threads special run methods have to be invoked like RWebWindow::Run()
///
/// TODO: probably detection of main thread should be delivered by central ROOT instances like gApplication or gROOT
/// Main thread can only make sense if special processing runs there and one can inject own functionality there

static std::thread::id gWebWinMainThrd = std::this_thread::get_id();
static bool gWebWinMainThrdSet = true;
static bool gWebWinLoopbackMode = true;
static bool gWebWinUseSessionKey = true;

//////////////////////////////////////////////////////////////////////////////////////////
/// Returns true when called from main process
/// Main process recognized at the moment when library is loaded
/// It supposed to be a thread where gApplication->Run() will be called
/// If application runs in separate thread, one have to use AssignMainThrd() method
/// to let RWebWindowsManager correctly recognize such situation

bool RWebWindowsManager::IsMainThrd()
{
   return gWebWinMainThrdSet && (std::this_thread::get_id() == gWebWinMainThrd);
}

//////////////////////////////////////////////////////////////////////////////////////////
/// Re-assigns main thread id
/// Normally main thread id recognized at the moment when library is loaded
/// It supposed to be a thread where gApplication->Run() will be called
/// If application runs in separate thread, one have to call this method
/// to let RWebWindowsManager correctly recognize such situation

void RWebWindowsManager::AssignMainThrd()
{
   gWebWinMainThrdSet = true;
   gWebWinMainThrd = std::this_thread::get_id();
}


//////////////////////////////////////////////////////////////////////////////////////////
/// Set loopback mode for THttpServer used for web widgets
/// By default is on. Only local communication via localhost address is possible
/// Disable it only if really necessary - it may open unauthorized access to your application from external nodes!!

void RWebWindowsManager::SetLoopbackMode(bool on)
{
   gWebWinLoopbackMode = on;
   bool print_warning = RWebWindowWSHandler::GetBoolEnv("WebGui.Warning", 1) == 1;
   if (!on) {
      if (print_warning) {
         printf("\nWARNING!\n");
         printf("Disabling loopback mode may leads to security problem.\n");
         printf("See https://root.cern/about/security/ for more information.\n\n");
      }
      if (!gWebWinUseSessionKey) {
         if (print_warning) {
            printf("Enforce session key to safely work on public network.\n");
            printf("One may call RWebWindowsManager::SetUseSessionKey(false); to disable it.\n");
         }
         gWebWinUseSessionKey = true;
      }
   }
}

//////////////////////////////////////////////////////////////////////////////////////////
/// Returns true if loopback mode used by THttpServer for web widgets

bool RWebWindowsManager::IsLoopbackMode()
{
   return gWebWinLoopbackMode;
}

//////////////////////////////////////////////////////////////////////////////////////////
/// Enable or disable usage of session key (default on)
/// If enabled, secrete session key used to calculate hash sum of each packet send to or from server
/// This protects ROOT http server from anauthorized usage

void RWebWindowsManager::SetUseSessionKey(bool on)
{
   gWebWinUseSessionKey = on;
}

//////////////////////////////////////////////////////////////////////////////////////////
/// Enable or disable usage of connection key (default on)
/// If enabled, each connection (and reconnection) to widget requires unique key
/// Connection key used together with session key to calculate hash sum of each packet send to or from server
/// This protects ROOT http server from anauthorized usage

void RWebWindowsManager::SetUseConnectionKey(bool on)
{
   gEnv->SetValue("WebGui.OnetimeKey", on ? "yes" : "no");
}

//////////////////////////////////////////////////////////////////////////////////////////
/// Enable or disable single connection mode (default on)
/// If enabled, one connection only with any web widget is possible
/// Any attempt to establish more connections will fail
/// if this mode is disabled some widgets like geom viewer or web canvas will be able to
/// to serve several clients - only when they are connected with required authentication keys

void RWebWindowsManager::SetSingleConnMode(bool on)
{
   gEnv->SetValue("WebGui.SingleConnMode", on ? "yes" : "no");
}

//////////////////////////////////////////////////////////////////////////////////////////
/// Configure server location which can be used for loading of custom scripts or files
/// When THttpServer instance of RWebWindowsManager will be created,
/// THttpServer::AddLocation() method with correspondent arguments will be invoked.

void RWebWindowsManager::AddServerLocation(const std::string &server_prefix, const std::string &files_path)
{
   if (server_prefix.empty() || files_path.empty())
      return;
   auto loc = GetServerLocations();
   std::string prefix = server_prefix;
   if (prefix.back() != '/')
      prefix.append("/");
   loc[prefix] = files_path;

   // now convert back to plain string
   TString cfg;
   for (auto &entry : loc) {
      if (cfg.Length() > 0)
         cfg.Append(";");
      cfg.Append(entry.first.c_str());
      cfg.Append(":");
      cfg.Append(entry.second.c_str());
   }

   gEnv->SetValue("WebGui.ServerLocations", cfg);

   auto serv = Instance()->GetServer();
   if (serv)
      serv->AddLocation(prefix.c_str(), files_path.c_str());
}

//////////////////////////////////////////////////////////////////////////////////////////
/// Returns server locations as <std::string, std::string>
/// Key is location name (with slash at the end) and value is file path

std::map<std::string, std::string> RWebWindowsManager::GetServerLocations()
{
   std::map<std::string, std::string> res;

   TString cfg = gEnv->GetValue("WebGui.ServerLocations","");
   auto arr = cfg.Tokenize(";");
   if (arr) {
      TIter next(arr);
      while(auto obj = next()) {
         TString arg = obj->GetName();

         auto p = arg.First(":");
         if (p == kNPOS) continue;

         TString prefix = arg(0, p);
         if (!prefix.EndsWith("/"))
            prefix.Append("/");
         TString path = arg(p+1, arg.Length() - p);

         res[prefix.Data()] = path.Data();
      }
      delete arr;
   }
   return res;
}

//////////////////////////////////////////////////////////////////////////////////////////
/// Clear all server locations
/// Does not change configuration of already running HTTP server

void RWebWindowsManager::ClearServerLocations()
{
   gEnv->SetValue("WebGui.ServerLocations", "");
}

//////////////////////////////////////////////////////////////////////////////////////////
/// Static method to generate cryptographic key
/// Parameter keylen defines length of cryptographic key in bytes
/// Output string will be hex formatted and includes "-" separator after every 4 bytes
/// Example for 16 bytes: "fca45856-41bee066-ff74cc96-9154d405"

std::string RWebWindowsManager::GenerateKey(int keylen)
{
   std::vector<unsigned char> buf(keylen, 0);
   auto res = gSystem->GetCryptoRandom(buf.data(), keylen);

   R__ASSERT(res == keylen && "Error in gSystem->GetCryptoRandom");

   std::string key;
   for (int n = 0; n < keylen; n++) {
      if ((n > 0) && (n % 4 == 0))
         key.append("-");
      auto t = TString::Itoa(buf[n], 16);
      if (t.Length() == 1)
         key.append("0");
      key.append(t.Data());
   }
   return key;
}

//////////////////////////////////////////////////////////////////////////////////////////
/// window manager constructor
/// Required here for correct usage of unique_ptr<THttpServer>

RWebWindowsManager::RWebWindowsManager()
{
   fSessionKey = GenerateKey(32);
   fUseSessionKey = gWebWinUseSessionKey;

   fExternalProcessEvents = RWebWindowWSHandler::GetBoolEnv("WebGui.ExternalProcessEvents") == 1;
   if (fExternalProcessEvents)
      RWebWindowsManager::AssignMainThrd();
}

//////////////////////////////////////////////////////////////////////////////////////////
/// window manager destructor
/// Required here for correct usage of unique_ptr<THttpServer>

RWebWindowsManager::~RWebWindowsManager()
{
   if (gApplication && fServer && !fServer->IsTerminated()) {
      gApplication->Disconnect("Terminate(Int_t)", fServer.get(), "SetTerminate()");
      fServer->SetTerminate();
   }
}

//////////////////////////////////////////////////////////////////////////////////////////
/// If ROOT_LISTENER_SOCKET variable is configured,
/// message will be sent to that unix socket

bool RWebWindowsManager::InformListener(const std::string &msg)
{
#ifdef R__WIN32
   (void) msg;
   return false;

#else

   const char *fname = gSystem->Getenv("ROOT_LISTENER_SOCKET");
   if (!fname || !*fname)
      return false;

   TSocket s(fname);
   if (!s.IsValid()) {
      R__LOG_ERROR(WebGUILog()) << "Problem with open listener socket " << fname << ", check ROOT_LISTENER_SOCKET environment variable";
      return false;
   }

   int res = s.SendRaw(msg.c_str(), msg.length());

   s.Close();

   if (res > 0) {
      // workaround to let handle socket by system outside ROOT process
      gSystem->ProcessEvents();
      gSystem->Sleep(10);
   }

   return res > 0;
#endif
}


//////////////////////////////////////////////////////////////////////////////////////////
/// Creates http server, if required - with real http engine (civetweb)
/// One could configure concrete HTTP port, which should be used for the server,
/// provide following entry in rootrc file:
///
///      WebGui.HttpPort: 8088
///
/// or specify range of http ports, which can be used:
///
///      WebGui.HttpPortMin: 8800
///      WebGui.HttpPortMax: 9800
///
/// By default range [8800..9800] is used
///
/// One also can bind HTTP server socket to loopback address,
/// In that case only connection from localhost will be available:
///
///      WebGui.HttpLoopback: yes
///
/// Or one could specify hostname which should be used for binding of server socket
///
///      WebGui.HttpBind: hostname | ipaddress
///
/// To use secured protocol, following parameter should be specified
///
///      WebGui.UseHttps: yes
///      WebGui.ServerCert: sertificate_filename.pem
///
/// Alternatively, one can specify unix socket to handle requests:
///
///      WebGui.UnixSocket: /path/to/unix/socket
///      WebGui.UnixSocketMode: 0700
///
/// Typically one used unix sockets together with server mode like `root --web=server:/tmp/root.socket` and
/// then redirect it via ssh tunnel (e.g. using `rootssh`) to client node
///
/// All incoming requests processed in THttpServer in timer handler with 10 ms timeout.
/// One may decrease value to improve latency or increase value to minimize CPU load
///
///      WebGui.HttpTimer: 10
///
/// To processing incoming http requests and websockets, THttpServer allocate 10 threads
/// One have to increase this number if more simultaneous connections are expected:
///
///      WebGui.HttpThrds: 10
///
/// One also can configure usage of special thread of processing of http server requests
///
///      WebGui.HttpThrd: no
///
/// Extra threads can be used to send data to different clients via websocket (default no)
///
///      WebGui.SenderThrds: no
///
/// If required, one could change websocket timeouts (default is 10000 ms)
///
///      WebGui.HttpWSTmout: 10000
///
/// By default, THttpServer created in restricted mode which only allows websocket handlers
/// and processes only very few other related http requests. For security reasons such mode
/// should be always enabled. Only if it is really necessary to process all other kinds
/// of HTTP requests, one could specify no for following parameter (default yes):
///
///      WebGui.WSOnly: yes
///
/// In some applications one may need to force longpoll websocket emulations from the beginning,
/// for instance when clients connected via proxys. Although JSROOT should automatically fallback
/// to longpoll engine, one can configure this directly (default no)
///
///      WebGui.WSLongpoll: no
///
/// Following parameter controls browser max-age caching parameter for files (default 3600)
/// When 0 is specified, browser cache will be disabled
///
///      WebGui.HttpMaxAge: 3600
///
/// Also one can provide extra URL options for, see TCivetweb::Create for list of supported options
///
///      WebGui.HttpExtraArgs: winsymlinks=no
///
/// One also can configure usage of FastCGI server for web windows:
///
///      WebGui.FastCgiPort: 4000
///      WebGui.FastCgiThreads: 10
///
/// To be able start web browser for such windows, one can provide real URL of the
/// web server which will connect with that FastCGI instance:
///
///      WebGui.FastCgiServer: https://your_apache_server.com/root_cgi_path
///
/// For some custom applications one requires to load JavaScript modules or other files.
/// For such applications one may require to load files from other locations which can be configured
/// with AddServerLocation() method or directly via:
///
///      WebGui.ServerLocations: location1:/file/path/to/location1;location2:/file/path/to/location2

bool RWebWindowsManager::CreateServer(bool with_http)
{
   if (gROOT->GetWebDisplay() == "off")
      return false;

   // explicitly protect server creation
   std::lock_guard<std::recursive_mutex> grd(fMutex);

   if (!fServer) {

      fServer = std::make_unique<THttpServer>("basic_sniffer");

      if (fExternalProcessEvents) {
         fUseHttpThrd = false;
      } else {
         auto serv_thrd = RWebWindowWSHandler::GetBoolEnv("WebGui.HttpThrd");
         if (serv_thrd != -1)
            fUseHttpThrd = serv_thrd != 0;
      }

      auto send_thrds = RWebWindowWSHandler::GetBoolEnv("WebGui.SenderThrds");
      if (send_thrds != -1)
         fUseSenderThreads = send_thrds != 0;

      if (IsUseHttpThread())
         fServer->CreateServerThread();

      if (gApplication)
         gApplication->Connect("Terminate(Int_t)", "THttpServer", fServer.get(), "SetTerminate()");

      fServer->SetWSOnly(RWebWindowWSHandler::GetBoolEnv("WebGui.WSOnly", 1) != 0);

      // this is location where all ROOT UI5 sources are collected
      // normally it is $ROOTSYS/ui5 or <prefix>/ui5 location
      TString ui5dir = gSystem->Getenv("ROOTUI5SYS");
      if (ui5dir.Length() == 0)
         ui5dir = gEnv->GetValue("WebGui.RootUi5Path","");

      if (ui5dir.Length() == 0)
         ui5dir.Form("%s/ui5", TROOT::GetDataDir().Data());

      if (gSystem->ExpandPathName(ui5dir)) {
         R__LOG_ERROR(WebGUILog()) << "Path to ROOT ui5 sources " << ui5dir << " not found, set ROOTUI5SYS correctly";
         ui5dir = ".";
      }

      fServer->AddLocation("rootui5sys/", ui5dir.Data());

      auto loc = GetServerLocations();
      for (auto &entry : loc)
         fServer->AddLocation(entry.first.c_str(), entry.second.c_str());
   }

   if (!with_http || fServer->IsAnyEngine())
      return true;

   int http_port = gEnv->GetValue("WebGui.HttpPort", 0);
   int http_min = gEnv->GetValue("WebGui.HttpPortMin", 8800);
   int http_max = gEnv->GetValue("WebGui.HttpPortMax", 9800);
   int http_timer = gEnv->GetValue("WebGui.HttpTimer", 10);
   int http_thrds = gEnv->GetValue("WebGui.HttpThreads", 10);
   int http_wstmout = gEnv->GetValue("WebGui.HttpWSTmout", 10000);
   int http_maxage = gEnv->GetValue("WebGui.HttpMaxAge", -1);
   const char *extra_args = gEnv->GetValue("WebGui.HttpExtraArgs", "");
   int fcgi_port = gEnv->GetValue("WebGui.FastCgiPort", 0);
   int fcgi_thrds = gEnv->GetValue("WebGui.FastCgiThreads", 10);
   const char *fcgi_serv = gEnv->GetValue("WebGui.FastCgiServer", "");
   fLaunchTmout = gEnv->GetValue("WebGui.LaunchTmout", 30.);
   bool assign_loopback = gWebWinLoopbackMode;
   const char *http_bind = gEnv->GetValue("WebGui.HttpBind", "");
   bool use_secure = RWebWindowWSHandler::GetBoolEnv("WebGui.UseHttps", 0) == 1;
   const char *ssl_cert = gEnv->GetValue("WebGui.ServerCert", "rootserver.pem");

   const char *unix_socket = gSystem->Getenv("ROOT_WEBGUI_SOCKET");
   if (!unix_socket || !*unix_socket)
      unix_socket = gEnv->GetValue("WebGui.UnixSocket", "");
   const char *unix_socket_mode = gEnv->GetValue("WebGui.UnixSocketMode", "0700");
   bool use_unix_socket = unix_socket && *unix_socket;

   if (use_unix_socket)
      fcgi_port = http_port = -1;

   if (assign_loopback)
      fcgi_port = -1;

   int ntry = 100;

   if ((http_port < 0) && (fcgi_port <= 0) && !use_unix_socket) {
      R__LOG_ERROR(WebGUILog()) << "Not allowed to create HTTP server, check WebGui.HttpPort variable";
      return false;
   }

   if ((http_timer > 0) && !IsUseHttpThread())
      fServer->SetTimer(http_timer);

   TRandom3 rnd;

   if (http_port < 0) {
      ntry = 0;
   } else {
      if (http_port == 0)
         rnd.SetSeed(0);
      if (http_max - http_min < ntry)
         ntry = http_max - http_min;
   }

   if (fcgi_port > 0)
      ntry++;

   if (use_unix_socket)
      ntry++;

   while (ntry-- >= 0) {
      if ((http_port == 0) && (fcgi_port <= 0) && !use_unix_socket) {
         if ((http_min <= 0) || (http_max <= http_min)) {
            R__LOG_ERROR(WebGUILog()) << "Wrong HTTP range configuration, check WebGui.HttpPortMin/Max variables";
            return false;
         }

         http_port = (int)(http_min + (http_max - http_min) * rnd.Rndm(1));
      }

      TString engine, url;
      if (fcgi_port > 0) {
         engine.Form("fastcgi:%d?thrds=%d", fcgi_port, fcgi_thrds);
         if (!fServer->CreateEngine(engine))
            return false;
         if (fcgi_serv && (strlen(fcgi_serv) > 0))
            fAddr = fcgi_serv;
         if (http_port < 0)
            return true;
         fcgi_port = 0;
      } else {
         if (use_unix_socket) {
            engine.Form("socket:%s?socket_mode=%s&", unix_socket, unix_socket_mode);
         } else {
            url = use_secure ? "https://" : "http://";
            engine.Form("%s:%d?", (use_secure ? "https" : "http"), http_port);
            if (assign_loopback) {
               engine.Append("loopback&");
               url.Append("localhost");
            } else if (http_bind && (strlen(http_bind) > 0)) {
               engine.Append(TString::Format("bind=%s&", http_bind));
               url.Append(http_bind);
            } else {
               url.Append("localhost");
            }
         }

         engine.Append(TString::Format("webgui&top=remote&thrds=%d&websocket_timeout=%d", http_thrds, http_wstmout));

         if (http_maxage >= 0)
            engine.Append(TString::Format("&max_age=%d", http_maxage));

         if (use_secure && !strchr(ssl_cert,'&')) {
            engine.Append("&ssl_cert=");
            engine.Append(ssl_cert);
         }

         if (!use_unix_socket && !assign_loopback && extra_args && strlen(extra_args) > 0) {
            engine.Append("&");
            engine.Append(extra_args);
         }

         if (fServer->CreateEngine(engine)) {
            if (use_unix_socket) {
               fAddr = "socket://"; // fictional socket URL
               fAddr.append(unix_socket);
               // InformListener(std::string("socket:") + unix_socket + "\n");
            } else if (http_port > 0) {
               fAddr = url.Data();
               fAddr.append(":");
               fAddr.append(std::to_string(http_port));
               // InformListener(std::string("http:") + std::to_string(http_port) + "\n");
            }
            return true;
         }
         use_unix_socket = false;
         http_port = 0;
      }
   }

   return false;
}

//////////////////////////////////////////////////////////////////////////////////////////
/// Creates new window
/// To show window, RWebWindow::Show() have to be called

std::shared_ptr<RWebWindow> RWebWindowsManager::CreateWindow()
{
   // we book manager mutex for a longer operation, locked again in server creation
   std::lock_guard<std::recursive_mutex> grd(fMutex);

   if (!CreateServer()) {
      R__LOG_ERROR(WebGUILog()) << "Cannot create server when creating window";
      return nullptr;
   }

   std::shared_ptr<RWebWindow> win = std::make_shared<RWebWindow>();

   if (!win) {
      R__LOG_ERROR(WebGUILog()) << "Fail to create RWebWindow instance";
      return nullptr;
   }

   double dflt_tmout = gEnv->GetValue("WebGui.OperationTmout", 50.);

   auto wshandler = win->CreateWSHandler(Instance(), ++fIdCnt, dflt_tmout);

   if (RWebWindowWSHandler::GetBoolEnv("WebGui.RecordData") > 0) {
      std::string fname, prefix;
      if (fIdCnt > 1) {
         prefix = std::string("f") + std::to_string(fIdCnt) + "_";
         fname = std::string("protcol") + std::to_string(fIdCnt) + ".json";
      } else {
         fname = "protocol.json";
      }
      win->RecordData(fname, prefix);
   }

   int queuelen = gEnv->GetValue("WebGui.QueueLength", 10);
   if (queuelen > 0)
      win->SetMaxQueueLength(queuelen);

   if (fExternalProcessEvents) {
      // special mode when window communication performed in THttpServer::ProcessRequests
      // used only with python which create special thread - but is has to be ignored!!!
      // therefore use main thread id to detect callbacks which are invoked only from that main thread
      win->fUseProcessEvents = true;
      win->fCallbacksThrdIdSet = gWebWinMainThrdSet;
      win->fCallbacksThrdId = gWebWinMainThrd;
   } else if (IsUseHttpThread())
      win->UseServerThreads();

   const char *token = gEnv->GetValue("WebGui.ConnToken", "");
   if (token && *token)
      win->SetConnToken(token);

   fServer->RegisterWS(wshandler);

   return win;
}

//////////////////////////////////////////////////////////////////////////////////////////
/// Release all references to specified window
/// Called from RWebWindow destructor

void RWebWindowsManager::Unregister(RWebWindow &win)
{
   if (win.fWSHandler)
      fServer->UnregisterWS(win.fWSHandler);

   if (fDeleteCallback)
      fDeleteCallback(win);
}

//////////////////////////////////////////////////////////////////////////
/// Provide URL address to access specified window from inside or from remote

std::string RWebWindowsManager::GetUrl(RWebWindow &win, bool remote, std::string *produced_key)
{
   if (!fServer) {
      R__LOG_ERROR(WebGUILog()) << "Server instance not exists when requesting window URL";
      return "";
   }

   std::string addr = "/";
   addr.append(win.fWSHandler->GetName());
   addr.append("/");

   bool qmark = false;

   std::string key;

   if (win.IsRequireAuthKey() || produced_key) {
      key = win.GenerateKey();
      R__ASSERT(!key.empty());
      addr.append("?key=");
      addr.append(key);
      qmark = true;
      std::unique_ptr<ROOT::RWebDisplayHandle> dummy;
      win.AddDisplayHandle(false, key, dummy);
   }

   auto token = win.GetConnToken();
   if (!token.empty()) {
      addr.append(qmark ? "&" : "?");
      addr.append("token=");
      addr.append(token);
   }

   if (remote) {
      if (!CreateServer(true) || fAddr.empty()) {
         R__LOG_ERROR(WebGUILog()) << "Fail to start real HTTP server when requesting URL";
         if (!key.empty())
            win.RemoveKey(key);
         return "";
      }

      addr = fAddr + addr;

      if (!key.empty() && !fSessionKey.empty() && fUseSessionKey && win.IsRequireAuthKey())
         addr += "#"s + fSessionKey;
   }

   if (produced_key)
      *produced_key = key;

   return addr;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
/// Show web window in specified location.
///
/// \param[inout] win web window by reference
/// \param user_args specifies where and how display web window
///
/// As display args one can use string like "firefox" or "chrome" - these are two main supported web browsers.
/// See RWebDisplayArgs::SetBrowserKind() for all available options. Default value for the browser can be configured
/// when starting root with --web argument like: "root --web=chrome". When root started in web server mode "root --web=server",
/// no web browser will be started - just the URL will be printed, which can be opened in any running web browser.
/// Also configurable via ROOT_WEBDISPLAY environment variable taking the same options.
///
/// If allowed, same window can be displayed several times (like for RCanvas or TCanvas)
///
/// Following parameters can be configured in rootrc file:
///
///      WebGui.Display: kind of display, identical to --web option and ROOT_WEBDISPLAY environment variable documented above
///      WebGui.OnetimeKey: if configured requires unique key every time window is connected (default yes)
///      WebGui.SingleConnMode: if configured the only connection and the only user of any widget is possible (default yes)
///      WebGui.Chrome: full path to Google Chrome executable
///      WebGui.ChromeBatch: command to start chrome in batch, used for image production, like "$prog --headless --disable-gpu $geometry $url"
///      WebGui.ChromeHeadless: command to start chrome in headless mode, like "fork: --headless --disable-gpu $geometry $url"
///      WebGui.ChromeInteractive: command to start chrome in interactive mode, like "$prog $geometry --app=\'$url\' &"
///      WebGui.Firefox: full path to Mozilla Firefox executable
///      WebGui.FirefoxHeadless: command to start Firefox in headless mode, like "fork:--headless --private-window --no-remote $profile $url"
///      WebGui.FirefoxInteractive: command to start Firefox in interactive mode, like "$prog --private-window \'$url\' &"
///      WebGui.FirefoxProfile: name of Firefox profile to use
///      WebGui.FirefoxProfilePath: file path to Firefox profile
///      WebGui.FirefoxRandomProfile: usage of random Firefox profile "no" - disabled, "yes" - enabled (default)
///      WebGui.LaunchTmout: time required to start process in seconds (default 30 s)
///      WebGui.CefTimer: periodic time to run CEF event loop (default 10 ms)
///      WebGui.CefUseViews: "yes" - enable / "no" - disable usage of CEF views frameworks (default is platform/version dependent)
///      WebGui.OperationTmout: time required to perform WebWindow operation like execute command or update drawings
///      WebGui.RecordData: if specified enables data recording for each web window; "yes" or "no" (default)
///      WebGui.JsonComp: compression factor for JSON conversion, if not specified - each widget uses own default values
///      WebGui.ForceHttp: "no" (default), "yes" - always create real http server to run web window
///      WebGui.Console: -1 - output only console.error(), 0 - add console.warn(), 1  - add console.log() output
///      WebGui.Debug: "no" (default), "yes" - enable more debug output on JSROOT side
///      WebGui.ConnCredits: 10 - number of packets which can be send by server or client without acknowledge from receiving side
///      WebGui.QueueLength: 10 - maximal number of entires in window send queue
///      WebGui.openui5src: alternative location for openui5 like https://openui5.hana.ondemand.com/1.135.0/
///      WebGui.openui5libs: list of pre-loaded ui5 libs like sap.m, sap.ui.layout, sap.ui.unified
///      WebGui.openui5theme: openui5 theme like sap_fiori_3 (default) or sap_horizon
///      WebGui.DarkMode: "no" (default), "yes" - switch to JSROOT dark mode and will use sap_fiori_3_dark theme
///
/// THttpServer-related parameters documented in \ref CreateServer method
///
/// In case of using web browsers based on snap sandboxing, if you see a runtime error about unauthorized access to the system
/// `/tmp/` folder, try callign `export TMPDIR=/home/user/` (adapt path to a real folder) before running ROOT. This workaround should
/// no longer be needed for recognized snap-installed firefox or chrome browsers if ROOT version >= 6.38

unsigned RWebWindowsManager::ShowWindow(RWebWindow &win, const RWebDisplayArgs &user_args)
{
   // silently ignore regular Show() calls in batch mode
   if (!user_args.IsHeadless() && gROOT->IsWebDisplayBatch())
      return 0;

   // for embedded window no any browser need to be started
   // also when off is specified, no browser should be started
   if ((user_args.GetBrowserKind() == RWebDisplayArgs::kEmbedded) || (user_args.GetBrowserKind() == RWebDisplayArgs::kOff))
      return 0;

   // catch window showing, used by the RBrowser to embed some of ROOT widgets
   if (fShowCallback)
      if (fShowCallback(win, user_args)) {
         // add dummy handle to pending connections, widget (like TWebCanvas) may wait until connection established
         auto handle = std::make_unique<RWebDisplayHandle>("");
         win.AddDisplayHandle(false, "", handle);
         return 0;
      }

   if (!fServer) {
      R__LOG_ERROR(WebGUILog()) << "Server instance not exists to show window";
      return 0;
   }

   RWebDisplayArgs args(user_args);

   if (args.IsHeadless() && !args.IsSupportHeadless()) {
      R__LOG_ERROR(WebGUILog()) << "Cannot use batch mode with " << args.GetBrowserName();
      return 0;
   }

   bool normal_http = RWebDisplayHandle::NeedHttpServer(args);
   if (!normal_http && (RWebWindowWSHandler::GetBoolEnv("WebGui.ForceHttp") > 0))
      normal_http = true;

   std::string key;

   std::string url = GetUrl(win, normal_http, &key);
   // empty url indicates failure, which already printed by GetUrl method
   if (url.empty())
      return 0;

   // we book manager mutex for a longer operation,
   std::lock_guard<std::recursive_mutex> grd(fMutex);

   args.SetUrl(url);

   if (args.GetWidth() <= 0)
      args.SetWidth(win.GetWidth());
   if (args.GetHeight() <= 0)
      args.SetHeight(win.GetHeight());
   if (args.GetX() < 0)
      args.SetX(win.GetX());
   if (args.GetY() < 0)
      args.SetY(win.GetY());

   if (args.IsHeadless())
      args.AppendUrlOpt("headless"); // used to create holder request

   if (!args.IsHeadless() && normal_http) {
      auto winurl = args.GetUrl();
      winurl.erase(0, fAddr.length());
      InformListener(std::string("win:") + winurl + "\n");
   }

   auto server = GetServer();

   if (win.IsUseCurrentDir() && server)
      server->AddLocation("currentdir/", ".");

   if (!args.IsHeadless() && ((args.GetBrowserKind() == RWebDisplayArgs::kServer) || gROOT->IsWebDisplayBatch()) /*&& (RWebWindowWSHandler::GetBoolEnv("WebGui.OnetimeKey") != 1)*/) {
      std::cout << "New web window: " << args.GetUrl() << std::endl;
      return 0;
   }

   if (fAddr.compare(0,9,"socket://") == 0)
      return 0;

#if !defined(R__MACOSX) && !defined(R__WIN32)
   if (args.IsInteractiveBrowser()) {
      const char *varname = "WebGui.CheckRemoteDisplay";
      if (RWebWindowWSHandler::GetBoolEnv(varname, 1) == 1) {
         const char *displ = gSystem->Getenv("DISPLAY");
         if (displ && *displ && (*displ != ':')) {
            gEnv->SetValue(varname, "no");
            std::cout << "\n"
               "ROOT web-based widget started in the session where DISPLAY set to " << displ << "\n" <<
               "Means web browser will be displayed on remote X11 server which is usually very inefficient\n"
               "One can start ROOT session in server mode like \"root -b --web=server:8877\" and forward http port to display node\n"
               "Or one can use rootssh script to configure port forwarding and display web widgets automatically\n"
               "Find more info on https://root.cern/for_developers/root7/#rbrowser\n"
               "This message can be disabled by setting \"" << varname << ": no\" in .rootrc file\n";
         }
      }
   }
#endif

   if (!normal_http)
      args.SetHttpServer(server);

   auto handle = RWebDisplayHandle::Display(args);

   if (!handle) {
      R__LOG_ERROR(WebGUILog()) << "Cannot display window in " << args.GetBrowserName();
      if (!key.empty())
         win.RemoveKey(key);
      return 0;
   }

   return win.AddDisplayHandle(args.IsHeadless(), key, handle);
}

//////////////////////////////////////////////////////////////////////////
/// Waits until provided check function or lambdas returns non-zero value
/// Regularly calls WebWindow::Sync() method to let run event loop
/// If call from the main thread, runs system events processing
/// Check function has following signature: int func(double spent_tm)
/// Parameter spent_tm is time in seconds, which already spent inside function
/// Waiting will be continued, if function returns zero.
/// First non-zero value breaks waiting loop and result is returned (or 0 if time is expired).
/// If parameter timed is true, timelimit (in seconds) defines how long to wait

int RWebWindowsManager::WaitFor(RWebWindow &win, WebWindowWaitFunc_t check, bool timed, double timelimit)
{
   int res = 0, cnt = 0;
   double spent = 0.;

   auto start = std::chrono::high_resolution_clock::now();

   win.Sync(); // in any case call sync once to ensure

   auto is_main_thread = IsMainThrd();

   while ((res = check(spent)) == 0) {

      if (is_main_thread)
         gSystem->ProcessEvents();

      win.Sync();

      // only when first 1000 events processed, invoke sleep
      if (++cnt > 1000)
         std::this_thread::sleep_for(std::chrono::milliseconds(cnt > 5000 ? 10 : 1));

      std::chrono::duration<double, std::milli> elapsed = std::chrono::high_resolution_clock::now() - start;

      spent = elapsed.count() * 1e-3; // use ms precision

      if (timed && (spent > timelimit))
         return -3;
   }

   return res;
}

//////////////////////////////////////////////////////////////////////////
/// Terminate http server and ROOT application

void RWebWindowsManager::Terminate()
{
   if (fServer)
      fServer->SetTerminate();

   // set flag which sometimes checked in TSystem::ProcessEvents
   gROOT->SetInterrupt(kTRUE);

   if (gApplication)
      TTimer::SingleShot(100, "TApplication",  gApplication, "Terminate()");
}
