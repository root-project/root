/// \file RWebWindowsManager.cxx
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

#include "ROOT/RWebWindowsManager.hxx"

#include <ROOT/TLogger.hxx>

#include "THttpServer.h"
#include "RWebWindowWSHandler.hxx"

#include "RConfigure.h"
#include "TSystem.h"
#include "TRandom.h"
#include "TString.h"
#include "TApplication.h"
#include "TTimer.h"
#include "TObjArray.h"
#include "TROOT.h"
#include "TEnv.h"

#include <thread>
#include <chrono>

#if !defined(_MSC_VER)
#include <unistd.h>
#include <stdlib.h>
#include <signal.h>
#include <spawn.h>
#else
#include <process.h>
#endif


/** \class ROOT::Experimental::RWebWindowsManager
\ingroup webdisplay

Central instance to create and show web-based windows like Canvas or FitPanel.

Manager responsible to creating THttpServer instance, which is used for RWebWindow's
communication with clients.

Method RWebWindowsManager::Show() used to show window in specified location.
*/

//////////////////////////////////////////////////////////////////////////////////////////
/// Returns default window manager
/// Used to display all standard ROOT elements like TCanvas or TFitPanel

std::shared_ptr<ROOT::Experimental::RWebWindowsManager> &ROOT::Experimental::RWebWindowsManager::Instance()
{
   static std::shared_ptr<RWebWindowsManager> sInstance = std::make_shared<ROOT::Experimental::RWebWindowsManager>();
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

//////////////////////////////////////////////////////////////////////////////////////////
/// Returns true when called from main process
/// Main process recognized at the moment when library is loaded

bool ROOT::Experimental::RWebWindowsManager::IsMainThrd()
{
   return std::this_thread::get_id() == gWebWinMainThrd;
}

//////////////////////////////////////////////////////////////////////////////////////////
/// window manager constructor
/// Required here for correct usage of unique_ptr<THttpServer>

ROOT::Experimental::RWebWindowsManager::RWebWindowsManager() = default;

//////////////////////////////////////////////////////////////////////////////////////////
/// window manager destructor
/// Required here for correct usage of unique_ptr<THttpServer>

ROOT::Experimental::RWebWindowsManager::~RWebWindowsManager()
{
   if (gApplication && fServer && !fServer->IsTerminated())
      gApplication->Disconnect("Terminate(Int_t)", "THttpServer", fServer.get(), "SetTerminate()");
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

bool ROOT::Experimental::RWebWindowsManager::CreateServer(bool with_http)
{
   // explicitly protect server creation
   std::lock_guard<std::recursive_mutex> grd(fMutex);

   if (!fServer) {

      fServer = std::make_unique<THttpServer>("basic_sniffer");

      const char *serv_thrd = gEnv->GetValue("WebGui.HttpThrd", "");
      if (serv_thrd && strstr(serv_thrd, "yes"))
         fUseHttpThrd = true;
      else if (serv_thrd && strstr(serv_thrd, "no"))
         fUseHttpThrd = false;

      const char *send_thrds = gEnv->GetValue("WebGui.SenderThrds", "");
      if (send_thrds && *send_thrds) {
         if (strstr(send_thrds, "yes"))
            fUseSenderThreads = true;
         else if (strstr(send_thrds, "no"))
            fUseSenderThreads = false;
         else
            R__ERROR_HERE("WebDisplay") << "WebGui.SenderThrds has to be yes or no";
      }

      if (IsUseHttpThread())
         fServer->CreateServerThread();

      if (gApplication)
         gApplication->Connect("Terminate(Int_t)", "THttpServer", fServer.get(), "SetTerminate()");
   }

   if (!with_http || !fAddr.empty())
      return true;

   int http_port = gEnv->GetValue("WebGui.HttpPort", 0);
   int http_min = gEnv->GetValue("WebGui.HttpPortMin", 8800);
   int http_max = gEnv->GetValue("WebGui.HttpPortMax", 9800);
   int http_wstmout = gEnv->GetValue("WebGui.HttpWSTmout", 10000);
   fLaunchTmout = gEnv->GetValue("WebGui.LaunchTmout", 30.);
   const char *http_loopback = gEnv->GetValue("WebGui.HttpLoopback", "no");
   const char *http_bind = gEnv->GetValue("WebGui.HttpBind", "");
   const char *http_ssl = gEnv->GetValue("WebGui.UseHttps", "no");
   const char *ssl_cert = gEnv->GetValue("WebGui.ServerCert", "rootserver.pem");

   bool assign_loopback = http_loopback && strstr(http_loopback, "yes");
   bool use_secure = http_ssl && strstr(http_ssl, "yes");
   int ntry = 100;

   if (http_port < 0) {
      R__ERROR_HERE("WebDisplay") << "Not allowed to create real HTTP server, check WebGui.HttpPort variable";
      return false;
   }

   if (!http_port)
      gRandom->SetSeed(0);

   if (http_max - http_min < ntry)
      ntry = http_max - http_min;

   while (ntry-- >= 0) {
      if (!http_port) {
         if ((http_min <= 0) || (http_max <= http_min)) {
            R__ERROR_HERE("WebDisplay") << "Wrong HTTP range configuration, check WebGui.HttpPortMin/Max variables";
            return false;
         }

         http_port = (int)(http_min + (http_max - http_min) * gRandom->Rndm(1));
      }

      TString engine, url(use_secure ? "https://" : "http://");
      engine.Form("%s:%d?websocket_timeout=%d", (use_secure ? "https" : "http"), http_port, http_wstmout);
      if (assign_loopback) {
         engine.Append("&loopback");
         url.Append("localhost");
      } else if (http_bind && (strlen(http_bind) > 0)) {
         engine.Append("&bind=");
         engine.Append(http_bind);
         url.Append(http_bind);
      } else {
         url.Append("localhost");
      }

      if (use_secure) {
         engine.Append("&ssl_cert=");
         engine.Append(ssl_cert);
      }

      if (fServer->CreateEngine(engine)) {
         fAddr = url.Data();
         fAddr.append(":");
         fAddr.append(std::to_string(http_port));
         return true;
      }

      http_port = 0;
   }

   return false;
}

//////////////////////////////////////////////////////////////////////////////////////////
/// Creates new window
/// To show window, RWebWindow::Show() have to be called

std::shared_ptr<ROOT::Experimental::RWebWindow> ROOT::Experimental::RWebWindowsManager::CreateWindow()
{

   // we book manager mutex for a longer operation, locked again in server creation
   std::lock_guard<std::recursive_mutex> grd(fMutex);

   if (!CreateServer()) {
      R__ERROR_HERE("WebDisplay") << "Cannot create server when creating window";
      return nullptr;
   }

   std::shared_ptr<ROOT::Experimental::RWebWindow> win = std::make_shared<ROOT::Experimental::RWebWindow>();

   if (!win) {
      R__ERROR_HERE("WebDisplay") << "Fail to create RWebWindow instance";
      return nullptr;
   }

   double dflt_tmout = gEnv->GetValue("WebGui.OperationTmout", 50.);

   auto wshandler = win->CreateWSHandler(Instance(), ++fIdCnt, dflt_tmout);

   fServer->RegisterWS(wshandler);

   return win;
}

//////////////////////////////////////////////////////////////////////////////////////////
/// Release all references to specified window
/// Called from RWebWindow destructor

void ROOT::Experimental::RWebWindowsManager::Unregister(ROOT::Experimental::RWebWindow &win)
{
   if (win.fWSHandler)
      fServer->UnregisterWS(win.fWSHandler);
}

//////////////////////////////////////////////////////////////////////////
/// Provide URL address to access specified window from inside or from remote

std::string ROOT::Experimental::RWebWindowsManager::GetUrl(const ROOT::Experimental::RWebWindow &win, bool batch_mode, bool remote)
{
   if (!fServer) {
      R__ERROR_HERE("WebDisplay") << "Server instance not exists when requesting window URL";
      return "";
   }

   std::string addr = "/";

   addr.append(win.fWSHandler->GetName());

   if (batch_mode)
      addr.append("/?batch_mode");
   else
      addr.append("/");

   if (remote) {
      if (!CreateServer(true)) {
         R__ERROR_HERE("WebDisplay") << "Fail to start real HTTP server when requesting URL";
         return "";
      }

      addr = fAddr + addr;
   }

   return addr;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
/// checks if provided executable exists

void ROOT::Experimental::RWebWindowsManager::TestProg(TString &prog, const std::string &nexttry)
{
   if ((prog.Length()==0) && !nexttry.empty())
      if (!gSystem->AccessPathName(nexttry.c_str(), kExecutePermission))
          prog = nexttry.c_str();
}

///////////////////////////////////////////////////////////////////////////////////////////////////
/// Show window in specified location
/// Parameter "where" specifies that kind of window display should be used. Possible values:
///
///  chrome  - use Google Chrome web browser, supports headless mode from v60, default
///  firefox - use Mozilla Firefox browser, supports headless mode from v57
///   native - (or empty string) either chrome or firefox, only these browsers support batch (headless) mode
///  browser - default system web-browser, no batch mode
///      cef - Chromium Embeded Framework, local display, local communication
///      qt5 - Qt5 WebEngine, local display, local communication
///    local - either cef or qt5
///   <prog> - any program name which will be started instead of default browser, like /usr/bin/opera
///            one could use following parameters:
///                  $url - URL address of the widget
///                $width - widget width
///               $height - widget height
///
///  If allowed, same window can be displayed several times (like for TCanvas)
///  Following parameters can be configured in rootrc file:
///
///   WebGui.Chrome:  full path to Google Chrome executable
///   WebGui.ChromeBatch: command to start chrome in batch
///   WebGui.ChromeInteractive: command to start chrome in interactive mode
///   WebGui.Firefox: full path to Mozialla Firefox executable
///   WebGui.FirefoxBatch: command to start Firefox in batch mode
///   WebGui.FirefoxInteractive: command to start Firefox in interactive mode
///   WebGui.FirefoxProfile: name of Firefox profile to use
///   WebGui.FirefoxProfilePath: file path to Firefox profile
///   WebGui.FirefoxRandomProfile: usage of random Firefox profile -1 never, 0 - only for batch mode (dflt), 1 - always
///   WebGui.LaunchTmout: time required to start process in seconds (default 30 s)
///   WebGui.OperationTmout: time required to perform WebWindow operation like execute command or update drawings

unsigned ROOT::Experimental::RWebWindowsManager::Show(ROOT::Experimental::RWebWindow &win, bool batch_mode, const std::string &_where)
{

   // silently ignore regular Show() calls in batch mode
   if (!batch_mode && gROOT->IsWebDisplayBatch())
      return 0;

   // we book manager mutex for a longer operation,
   std::lock_guard<std::recursive_mutex> grd(fMutex);

   if (!fServer) {
      R__ERROR_HERE("WebDisplay") << "Server instance not exists to show window";
      return 0;
   }

   std::string key;
   std::string rmdir;
   int ntry = 100000;

   do {
      key = std::to_string(gRandom->Integer(0x100000));
   } while ((--ntry > 0) && win.HasKey(key));
   if (ntry == 0) {
      R__ERROR_HERE("WebDisplay") << "Fail to create unique key for the window";
      return 0;
   }

   std::string addr = GetUrl(win, batch_mode, false);
   if (addr.find("?") != std::string::npos)
      addr.append("&key=");
   else
      addr.append("?key=");
   addr.append(key);

   std::string where = _where;
   if (where.empty())
      where = gROOT->GetWebDisplay().Data();

   enum { kCustom, kNative, kLocal, kChrome, kFirefox, kCEF, kQt5 } kind = kCustom;

   if (where == "local")
      kind = kLocal;
   else if (where.empty() || (where == "native"))
      kind = kNative;
   else if (where == "firefox")
      kind = kFirefox;
   else if ((where == "chrome") || (where == "chromium"))
      kind = kChrome;
   else if (where == "cef")
      kind = kCEF;
   else if (where == "qt5")
      kind = kQt5;
   else
      kind = kCustom; // all others kinds, normally name of alternative web browser

#ifdef R__HAS_CEFWEB

   const char *cef_path = gSystem->Getenv("CEF_PATH");
   const char *rootsys = gSystem->Getenv("ROOTSYS");
   if (cef_path && !gSystem->AccessPathName(cef_path) && rootsys && ((kind == kLocal) || (kind == kCEF))) {

      Func_t symbol_cef = gSystem->DynFindSymbol("*", "webgui_start_browser_in_cef3");

      if (!symbol_cef) {
         gSystem->Load("libROOTCefDisplay");
         // TODO: make minimal C++ interface here
         symbol_cef = gSystem->DynFindSymbol("*", "webgui_start_browser_in_cef3");
      }

      if (symbol_cef) {

         if (batch_mode) {
            const char *displ = gSystem->Getenv("DISPLAY");
            if (!displ || !*displ) {
                R__ERROR_HERE("WebDisplay") << "To use CEF in batch mode DISPLAY variable should be set."
                                               " See gui/cefdisplay/Readme.md for more info";
                return 0;
             }
         }

         typedef void (*FunctionCef3)(const char *, void *, bool, const char *, const char *, unsigned, unsigned);
         R__DEBUG_HERE("WebDisplay") << "Show window " << addr << " in CEF";
         FunctionCef3 func = (FunctionCef3)symbol_cef;
         func(addr.c_str(), fServer.get(), batch_mode, rootsys, cef_path, win.GetWidth(), win.GetHeight());
         return win.AddProcId(batch_mode, key, "cef");
      }

      if (kind == kCEF) {
         R__ERROR_HERE("WebDisplay") << "CEF libraries not found";
         return 0;
      }
   }
#endif

#ifdef R__HAS_QT5WEB

   if ((kind == kLocal) || (kind == kQt5)) {
      Func_t symbol_qt5 = gSystem->DynFindSymbol("*", "webgui_start_browser_in_qt5");

      if (!symbol_qt5) {
         gSystem->Load("libROOTQt5WebDisplay");
         symbol_qt5 = gSystem->DynFindSymbol("*", "webgui_start_browser_in_qt5");
      }
      if (symbol_qt5) {
         if (batch_mode) {
            R__ERROR_HERE("WebDisplay") << "Qt5 does not support batch mode";
            return 0;
         }
         typedef void (*FunctionQt5)(const char *, void *, bool, unsigned, unsigned);
         R__DEBUG_HERE("WebDisplay") << "Show window " << addr << " in Qt5 WebEngine";
         FunctionQt5 func = (FunctionQt5)symbol_qt5;
         func(addr.c_str(), fServer.get(), batch_mode, win.GetWidth(), win.GetHeight());
         return win.AddProcId(batch_mode, key, "qt5");
      }
      if (kind == kQt5) {
         R__ERROR_HERE("WebDisplay") << "Qt5 libraries not found";
         return 0;
      }
   }
#endif

   if ((kind == kLocal) || (kind == kQt5) || (kind == kCEF)) {
      R__ERROR_HERE("WebDisplay") << "Neither Qt5 nor CEF libraries were found to provide local display";
      return 0;
   }

#ifdef _MSC_VER
   std::string ProgramFiles = gSystem->Getenv("ProgramFiles");
   size_t pos = ProgramFiles.find(" (x86)");
   if (pos != std::string::npos)
      ProgramFiles.erase(pos, 6);
   std::string ProgramFilesx86 = gSystem->Getenv("ProgramFiles(x86)");
#endif

   TString exec;

   std::string swidth = std::to_string(win.GetWidth() ? win.GetWidth() : 800);
   std::string sheight = std::to_string(win.GetHeight() ? win.GetHeight() : 600);
   TString prog;

   if ((kind == kNative) || (kind == kChrome)) {
      // see https://peter.sh/experiments/chromium-command-line-switches/

      TestProg(prog, gEnv->GetValue("WebGui.Chrome", ""));

#ifdef _MSC_VER
      std::string fullpath;
      if (!ProgramFiles.empty()){
         fullpath = ProgramFiles + "\\Google\\Chrome\\Application\\chrome.exe";
         TestProg(prog, fullpath);
      }
      if (!ProgramFilesx86.empty()) {
         fullpath = ProgramFilesx86 + "\\Google\\Chrome\\Application\\chrome.exe";
         TestProg(prog, fullpath);
      }
#endif
#ifdef R__MACOSX
      prog.ReplaceAll("%20"," ");
      TestProg(prog, "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome");
#endif
#ifdef R__LINUX
      TestProg(prog, "/usr/bin/chromium");
      TestProg(prog, "/usr/bin/chromium-browser");
      TestProg(prog, "/usr/bin/chrome-browser");
#endif
      if (prog.Length() > 0)
         kind = kChrome;
#ifdef _MSC_VER
      if (batch_mode)
         exec = gEnv->GetValue("WebGui.ChromeBatch", "fork: --headless --disable-gpu $url");
      else
         exec = gEnv->GetValue("WebGui.ChromeInteractive", "$prog --window-size=$width,$height --app=$url");
#else
      if (batch_mode)
         exec = gEnv->GetValue("WebGui.ChromeBatch", "fork:--headless $url");
      else
         exec = gEnv->GetValue("WebGui.ChromeInteractive", "$prog --window-size=$width,$height --app=\'$url\' &");
#endif
   }

   if ((kind == kFirefox) || ((kind == kNative) && (kind != kChrome))) {
      // to use firefox in batch mode at the same time as other firefox is running,
      // one should use extra profile. This profile should be created first:
      //    firefox -no-remote -CreateProfile root_batch
      // And then in the start command one should add:
      //    $prog -headless -no-remote -P root_batch -window-size=$width,$height $url
      // By default, no profile is specified, but this requires that no firefox is running

      TestProg(prog, gEnv->GetValue("WebGui.Firefox", ""));

#ifdef _MSC_VER
      std::string fullpath;
      if (!ProgramFiles.empty()) {
         fullpath = ProgramFiles + "\\Mozilla Firefox\\firefox.exe";
         TestProg(prog, fullpath);
      }
      if (!ProgramFilesx86.empty()) {
         fullpath = ProgramFilesx86 + "\\Mozilla Firefox\\firefox.exe";
         TestProg(prog, fullpath);
      }
#endif
#ifdef R__MACOSX
      prog.ReplaceAll("%20"," ");
      TestProg(prog, "/Applications/Firefox.app/Contents/MacOS/firefox");
#endif
#ifdef R__LINUX
      TestProg(prog, "/usr/bin/firefox");
#endif

      if (prog.Length() > 0)
         kind = kFirefox;

#ifdef _MSC_VER
      if (batch_mode)
         // there is a problem when specifying the window size with wmic on windows:
         // It gives: Invalid format. Hint: <paramlist> = <param> [, <paramlist>].
         exec = gEnv->GetValue("WebGui.FirefoxBatch", "fork: -headless -no-remote $profile $url");
      else
         exec = gEnv->GetValue("WebGui.FirefoxInteractive", "$prog -width=$width -height=$height $profile $url");
#else
      if (batch_mode)
         exec = gEnv->GetValue("WebGui.FirefoxBatch", "fork:-headless -no-remote $profile $url");
      else
         exec = gEnv->GetValue("WebGui.FirefoxInteractive", "$prog -width $width -height $height $profile \'$url\' &");
#endif

      if ((kind == kFirefox) && (exec.Index("$profile") != kNPOS)) {
         TString profile_arg;

         const char *ff_profile = gEnv->GetValue("WebGui.FirefoxProfile","");
         const char *ff_profilepath = gEnv->GetValue("WebGui.FirefoxProfilePath","");
         Int_t ff_randomprofile = gEnv->GetValue("WebGui.FirefoxRandomProfile", 0);
         if (ff_profile && *ff_profile) {
            profile_arg.Form("-P %s", ff_profile);
         } else if (ff_profilepath && *ff_profilepath) {
            profile_arg.Form("-profile %s", ff_profilepath);
         } else if ((ff_randomprofile > 0) || (batch_mode && (ff_randomprofile>=0))) {

            gRandom->SetSeed(0);

            TString rnd_profile = TString::Format("root_ff_profile_%d", gRandom->Integer(0x100000));
            TString profile_dir = TString::Format("%s/%s", gSystem->TempDirectory(), rnd_profile.Data());

            profile_arg.Form("-profile %s", profile_dir.Data());
            if (!batch_mode) profile_arg.Prepend("-no-remote ");

            gSystem->Exec(Form("%s %s -no-remote -CreateProfile \"%s %s\"", prog.Data(), (batch_mode ? "-headless" : ""), rnd_profile.Data(), profile_dir.Data()));

            rmdir = std::string("$rmdir$") + profile_dir.Data();
         }

         exec.ReplaceAll("$profile", profile_arg.Data());
      }
   }

   if ((kind != kFirefox) && (kind != kChrome)) {
      if (where == "native") {
         R__ERROR_HERE("WebDisplay") << "Neither firefox nor chrome are detected for native display";
         return 0;
      }

      if (batch_mode) {
         R__ERROR_HERE("WebDisplay") << "To use batch mode 'chrome' or 'firefox' should be configured as output";
         return 0;
      }

      if (kind != kNative) {
         if (where.find("$") != std::string::npos) {
            exec = where.c_str();
         } else {
            exec = "$prog $url &";
            prog = where.c_str();
         }
      } else if (gSystem->InheritsFrom("TMacOSXSystem")) {
         exec = "open \'$url\'";
      } else if (gSystem->InheritsFrom("TWinNTSystem")) {
         exec = "start $url";
      } else {
         exec = "xdg-open \'$url\' &";
      }
   }

   if (!CreateServer(true)) {
      R__ERROR_HERE("WebDisplay") << "Fail to start real HTTP server";
      return 0;
   }

   addr = fAddr + addr;

   exec.ReplaceAll("$url", addr.c_str());
   exec.ReplaceAll("$width", swidth.c_str());
   exec.ReplaceAll("$height", sheight.c_str());

   if (exec.Index("fork:") == 0) {
      exec.Remove(0, 5);
#if !defined(_MSC_VER)

      std::unique_ptr<TObjArray> args(exec.Tokenize(" "));
      if (!args || (args->GetLast()<=0)) {
         R__ERROR_HERE("WebDisplay") << "Fork instruction is empty";
         return 0;
      }

      std::vector<char *> argv;
      argv.push_back((char *) prog.Data());
      for (Int_t n = 0; n <= args->GetLast(); ++n)
         argv.push_back((char *)args->At(n)->GetName());
      argv.push_back(nullptr);

      R__DEBUG_HERE("WebDisplay") << "Show web window in browser with posix_spawn:\n" << prog << " " << exec;

      pid_t pid;
      int status = posix_spawn(&pid, argv[0], nullptr, nullptr, argv.data(), nullptr);
      if (status != 0) {
         R__ERROR_HERE("WebDisplay") << "Fail to launch " << argv[0];
         return 0;
      }

      return win.AddProcId(batch_mode, key, std::string("pid:") + std::to_string((int)pid) + rmdir);

#else
      std::string tmp;
      char c;
      int pid;
      if (prog.Length()) {
         exec.Prepend(Form("wmic process call create \"%s", prog.Data()));
      } else {
         R__ERROR_HERE("WebDisplay") << "No Web browser found in Program Files!";
         return false;
      }
      exec.Append("\" | find \"ProcessId\" ");
      TString process_id(gSystem->GetFromPipe(exec.Data()));
      std::stringstream ss(process_id.Data());
      ss >> tmp >> c >> pid;
      return win.AddProcId(batch_mode, key, std::string("pid:") + std::to_string((int)pid) + rmdir);
#endif
   }

#ifdef R__MACOSX
   prog.ReplaceAll(" ", "\\ ");
#endif

#ifdef _MSC_VER
   std::unique_ptr<TObjArray> args(exec.Tokenize(" "));
   std::vector<char *> argv;
   if (prog.EndsWith("chrome.exe"))
      argv.push_back("chrome.exe");
   else if (prog.EndsWith("firefox.exe"))
      argv.push_back("firefox.exe");
   for (Int_t n = 1; n <= args->GetLast(); ++n)
      argv.push_back((char *)args->At(n)->GetName());
   argv.push_back(nullptr);
#endif

   exec.ReplaceAll("$prog", prog.Data());

   unsigned connid = win.AddProcId(batch_mode, key, where + rmdir); // for now just application name

   R__DEBUG_HERE("WebDisplay") << "Showing web window in browser with:\n" << exec;

#ifdef _MSC_VER
   _spawnv(_P_NOWAIT, prog.Data(), argv.data());
#else
   gSystem->Exec(exec);
#endif

   return connid;
}

///////////////////////////////////////////////////////////////////////////////////////////
/// When window connection is closed, correspondent browser application may need to be halted
/// Process id produced by the Show() method

void ROOT::Experimental::RWebWindowsManager::HaltClient(const std::string &procid)
{
   std::string arg = procid;
   std::string tmpdir;

   auto pos = arg.find("$rmdir$");
   if (pos != std::string::npos) {
      tmpdir = arg.substr(pos+7);
      arg.resize(pos);
   }

   // kill program first
   if (arg.find("pid:") == 0) {

      int pid = std::stoi(arg.substr(4));

#if !defined(_MSC_VER)
      if (pid>0) kill(pid, SIGKILL);
#else
      if (pid > 0) gSystem->Exec(TString::Format("taskkill /F /PID %d", pid));
#endif
   }

   // delete temporary directory at the end
   if (!tmpdir.empty()) {
#if !defined(_MSC_VER)
      gSystem->Exec(TString::Format("rm -rf %s", tmpdir.c_str()));
#else
      gSystem->Exec(TString::Format("rmdir /S /Q %s", tmpdir.c_str()));
#endif
   }
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

int ROOT::Experimental::RWebWindowsManager::WaitFor(RWebWindow &win, WebWindowWaitFunc_t check, bool timed, double timelimit)
{
   int res = 0;
   int cnt = 0;
   double spent = 0;

   auto start = std::chrono::high_resolution_clock::now();

   win.Sync(); // in any case call sync once to ensure

   while ((res = check(spent)) == 0) {

      if (IsMainThrd())
         gSystem->ProcessEvents();

      win.Sync();

      std::this_thread::sleep_for(std::chrono::milliseconds(1));

      std::chrono::duration<double, std::milli> elapsed = std::chrono::high_resolution_clock::now() - start;

      spent = elapsed.count() * 1e-3; // use ms precision

      if (timed && (spent > timelimit))
         return -3;

      cnt++;
   }

   return res;
}

//////////////////////////////////////////////////////////////////////////
/// Terminate http server and ROOT application

void ROOT::Experimental::RWebWindowsManager::Terminate()
{
   if (fServer)
      fServer->SetTerminate();

   // use timer to avoid situation when calling object is deleted by terminate
   if (gApplication)
      TTimer::SingleShot(100, "TApplication", gApplication, "Terminate()");
}
