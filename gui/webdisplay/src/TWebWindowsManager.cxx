/// \file TWebWindowsManager.cxx
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

#include "ROOT/TWebWindowsManager.hxx"

#include <ROOT/TLogger.hxx>

#include "THttpServer.h"
#include "TWebWindowWSHandler.hxx"

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
#include <signal.h>
#include <spawn.h>
#else
#include <process.h>
#endif


namespace ROOT {
namespace Experimental {

class TWebWindowManagerGuard {

   TWebWindowsManager &fMgr;

   public:
      TWebWindowManagerGuard(TWebWindowsManager &mgr) : fMgr(mgr)
      {
         while (true) {
            {
               std::lock_guard<std::mutex> grd(fMgr.fMutex);
               if (fMgr.fMutexBooked == 0) {
                  fMgr.fMutexBooked++;
                  fMgr.fBookedThrd = std::this_thread::get_id();
                  break;
               }

               if (fMgr.fBookedThrd == std::this_thread::get_id()) {
                  fMgr.fMutexBooked++;
                  break;
               }
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(10));
         }
      }

      ~TWebWindowManagerGuard()
      {
         std::lock_guard<std::mutex> grd(fMgr.fMutex);
         if (!fMgr.fMutexBooked) {
            R__ERROR_HERE("WebDisplay") << "fMutexBooked counter is empty - fatal error";
         } else {
            fMgr.fMutexBooked--;
         }
      }
};

}
}


/** \class ROOT::Experimental::TWebWindowManager
\ingroup webdisplay

Central instance to create and show web-based windows like Canvas or FitPanel.

Manager responsible to creating THttpServer instance, which is used for TWebWindow's
communication with clients.

Method TWebWindowsManager::Show() used to show window in specified location.
*/

//////////////////////////////////////////////////////////////////////////////////////////
/// Returns default window manager
/// Used to display all standard ROOT elements like TCanvas or TFitPanel

std::shared_ptr<ROOT::Experimental::TWebWindowsManager> &ROOT::Experimental::TWebWindowsManager::Instance()
{
   static std::shared_ptr<TWebWindowsManager> sInstance = std::make_shared<ROOT::Experimental::TWebWindowsManager>();
   return sInstance;
}

//////////////////////////////////////////////////////////////////
/// This thread id used to identify main application thread, where ROOT event processing runs
/// To inject code in that thread, one should use TTimer (like THttpServer does)
/// In other threads special run methods have to be invoked like TWebWindow::Run()
///
/// TODO: probably detection of main thread should be delivered by central ROOT instances like gApplication or gROOT
/// Main thread can only make sense if special processing runs there and one can inject own functionality there

static std::thread::id gWebWinMainThrd = std::this_thread::get_id();

//////////////////////////////////////////////////////////////////////////////////////////
/// Returns true when called from main process
/// Main process recognized at the moment when library is loaded

bool ROOT::Experimental::TWebWindowsManager::IsMainThrd()
{
   return std::this_thread::get_id() == gWebWinMainThrd;
}

//////////////////////////////////////////////////////////////////////////////////////////
/// window manager constructor
/// Required here for correct usage of unique_ptr<THttpServer>

ROOT::Experimental::TWebWindowsManager::TWebWindowsManager() = default;

//////////////////////////////////////////////////////////////////////////////////////////
/// window manager destructor
/// Required here for correct usage of unique_ptr<THttpServer>

ROOT::Experimental::TWebWindowsManager::~TWebWindowsManager()
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
///      WebGui.HttpWStmout: 10000

bool ROOT::Experimental::TWebWindowsManager::CreateHttpServer(bool with_http)
{
   if (!fServer) {
      // explicitly protect server creation
      TWebWindowManagerGuard grd(*this);

      fServer = std::make_unique<THttpServer>("basic_sniffer");

      const char *serv_thrd = gEnv->GetValue("WebGui.HttpThrd", "");
      if (serv_thrd && strstr(serv_thrd, "yes"))
         fUseHttpThrd = true;
      else if (serv_thrd && strstr(serv_thrd, "no"))
         fUseHttpThrd = false;

      if (IsUseHttpThread())
         fServer->CreateServerThread();

      if (gApplication)
         gApplication->Connect("Terminate(Int_t)", "THttpServer", fServer.get(), "SetTerminate()");
   }

   if (!with_http || !fAddr.empty())
      return true;

   // explicitly protect HTTP engine creation

   TWebWindowManagerGuard grd(*this);

   int http_port = gEnv->GetValue("WebGui.HttpPort", 0);
   int http_min = gEnv->GetValue("WebGui.HttpPortMin", 8800);
   int http_max = gEnv->GetValue("WebGui.HttpPortMax", 9800);
   int http_wstmout = gEnv->GetValue("WebGui.HttpWStmout", 10000);
   const char *http_loopback = gEnv->GetValue("WebGui.HttpLoopback", "no");
   const char *http_bind = gEnv->GetValue("WebGui.HttpBind", "");
   const char *http_ssl = gEnv->GetValue("WebGui.UseHttps", "no");
   const char *ssl_cert = gEnv->GetValue("WebGui.ServerCert", "rootserver.pem");

   const char *send_thrds = gEnv->GetValue("WebGui.SenderThrds", "");
   if (send_thrds && strstr(send_thrds, "yes"))
      fUseSenderThreads = true;
   else if (send_thrds && strstr(send_thrds, "no"))
      fUseSenderThreads = false;

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
/// To show window, TWebWindow::Show() have to be called

std::shared_ptr<ROOT::Experimental::TWebWindow> ROOT::Experimental::TWebWindowsManager::CreateWindow(bool batch_mode)
{

   // we book manager mutex for a longer operation, later
   TWebWindowManagerGuard grd(*this);

   if (!CreateHttpServer()) {
      R__ERROR_HERE("WebDisplay") << "Cannot create http server when creating window";
      return nullptr;
   }

   std::shared_ptr<ROOT::Experimental::TWebWindow> win = std::make_shared<ROOT::Experimental::TWebWindow>();

   if (!win) {
      R__ERROR_HERE("WebDisplay") << "Fail to create TWebWindow instance";
      return nullptr;
   }

   win->SetBatchMode(batch_mode || gROOT->IsWebDisplayBatch());

   win->SetId(++fIdCnt); // set unique ID

   // fDisplays.push_back(win);

   win->fMgr = Instance();

   win->CreateWSHandler();

   fServer->RegisterWS(win->fWSHandler);

   return win;
}

//////////////////////////////////////////////////////////////////////////////////////////
/// Release all references to specified window
/// Called from TWebWindow destructor

void ROOT::Experimental::TWebWindowsManager::Unregister(ROOT::Experimental::TWebWindow &win)
{
   // TODO: close all active connections of the window

   if (win.fWSHandler)
      fServer->UnregisterWS(win.fWSHandler);

   //   for (auto displ = fDisplays.begin(); displ != fDisplays.end(); displ++) {
   //      if (displ->get() == win) {
   //         fDisplays.erase(displ);
   //         break;
   //      }
   //   }
}

//////////////////////////////////////////////////////////////////////////
/// Provide URL address to access specified window from inside or from remote

std::string ROOT::Experimental::TWebWindowsManager::GetUrl(ROOT::Experimental::TWebWindow &win, bool remote)
{
   if (!fServer) {
      R__ERROR_HERE("WebDisplay") << "Server instance not exists when requesting window URL";
      return "";
   }

   std::string addr = "/";

   addr.append(win.fWSHandler->GetName());

   if (win.IsBatchMode())
      addr.append("/?batch_mode");
   else
      addr.append("/");

   if (remote) {
      if (!CreateHttpServer(true)) {
         R__ERROR_HERE("WebDisplay") << "Fail to start real HTTP server when requesting URL";
         return "";
      }

      addr = fAddr + addr;
   }

   return addr;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
/// checks if provided executable exists

void ROOT::Experimental::TWebWindowsManager::TestProg(TString &prog, const std::string &nexttry)
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

bool ROOT::Experimental::TWebWindowsManager::Show(ROOT::Experimental::TWebWindow &win, const std::string &_where)
{
   if (!fServer) {
      R__ERROR_HERE("WebDisplay") << "Server instance not exists to show window";
      return false;
   }

   std::string key;
   int ntry = 1000;

   do {
      key = std::to_string(gRandom->Integer(0x100000));
   } while ((--ntry > 0) && win.HasKey(key));
   if (ntry == 0) {
      R__ERROR_HERE("WebDisplay") << "Fail to create unique key for the window";
      return false;
   }

   std::string addr = GetUrl(win, false);
   if (addr.find("?") != std::string::npos)
      addr.append("&key=");
   else
      addr.append("?key=");
   addr.append(key);

   std::string where = _where;
   if (where.empty())
      where = gROOT->GetWebDisplay().Data();

   bool is_native = where.empty() || (where == "native"),
        is_local = where == "local", // either cef or qt5
        is_chrome = (where == "chrome") || (where == "chromium"),
        is_firefox = (where == "firefox");

#ifdef R__HAS_CEFWEB

   bool is_cef = (where == "cef");

   const char *cef_path = gSystem->Getenv("CEF_PATH");
   const char *rootsys = gSystem->Getenv("ROOTSYS");
   if (cef_path && !gSystem->AccessPathName(cef_path) && rootsys && (is_local || is_cef)) {

      Func_t symbol_cef = gSystem->DynFindSymbol("*", "webgui_start_browser_in_cef3");

      if (!symbol_cef) {
         gSystem->Load("libROOTCefDisplay");
         // TODO: make minimal C++ interface here
         symbol_cef = gSystem->DynFindSymbol("*", "webgui_start_browser_in_cef3");
      }

      if (symbol_cef) {

         if (win.IsBatchMode()) {
            const char *displ = gSystem->Getenv("DISPLAY");
            if (!displ || !*displ) {
                R__ERROR_HERE("WebDisplay") << "To use CEF in batch mode DISPLAY variable should be set."
                                               " See gui/cefdisplay/Readme.md for more info";
                return false;
             }
         }

         typedef void (*FunctionCef3)(const char *, void *, bool, const char *, const char *, unsigned, unsigned);
         R__DEBUG_HERE("WebDisplay") << "Show window " << addr << " in CEF";
         FunctionCef3 func = (FunctionCef3)symbol_cef;
         func(addr.c_str(), fServer.get(), win.IsBatchMode(), rootsys, cef_path, win.GetWidth(), win.GetHeight());
         win.AddKey(key, "cef");
         return true;
      }

      if (is_cef) {
         R__ERROR_HERE("WebDisplay") << "CEF libraries not found";
         return false;
      }
   }
#endif

#ifdef R__HAS_QT5WEB

   bool is_qt5 = (where == "qt5");

   if (is_local || is_qt5) {
      Func_t symbol_qt5 = gSystem->DynFindSymbol("*", "webgui_start_browser_in_qt5");

      if (!symbol_qt5) {
         gSystem->Load("libROOTQt5WebDisplay");
         symbol_qt5 = gSystem->DynFindSymbol("*", "webgui_start_browser_in_qt5");
      }
      if (symbol_qt5) {
         if (win.IsBatchMode()) {
            R__ERROR_HERE("WebDisplay") << "Qt5 does not support batch mode";
            return false;
         }
         typedef void (*FunctionQt5)(const char *, void *, bool, unsigned, unsigned);
         R__DEBUG_HERE("WebDisplay") << "Show window " << addr << " in Qt5 WebEngine";
         FunctionQt5 func = (FunctionQt5)symbol_qt5;
         func(addr.c_str(), fServer.get(), win.IsBatchMode(), win.GetWidth(), win.GetHeight());
         win.AddKey(key, "qt5");
         return true;
      }
      if (is_qt5) {
         R__ERROR_HERE("WebDisplay") << "Qt5 libraries not found";
         return false;
      }
   }
#endif

   if (is_local) {
      R__ERROR_HERE("WebDisplay") << "Neither Qt5 nor CEF libraries were found to provide local display";
      return false;
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

   if (is_native || is_chrome) {
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
      is_chrome = prog.Length()>0;
#ifdef _MSC_VER
      if (win.IsBatchMode())
         exec = gEnv->GetValue("WebGui.ChromeBatch", "fork: --headless --disable-gpu --remote-debugging-port=$port $url");
      else
         exec = gEnv->GetValue("WebGui.ChromeInteractive", "$prog --window-size=$width,$height --app=$url");
#else
      if (win.IsBatchMode())
         exec = gEnv->GetValue("WebGui.ChromeBatch", "fork:--headless --disable-gpu --disable-webgl --remote-debugging-socket-fd=0 $url");
      else
         exec = gEnv->GetValue("WebGui.ChromeInteractive", "$prog --window-size=$width,$height --app=\'$url\' &");
#endif
   }

   if (is_firefox || (is_native && !is_chrome)) {
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

      is_firefox = prog.Length() > 0;

#ifdef _MSC_VER
      if (win.IsBatchMode())
         // there is a problem when specifying the window size with wmic on windows:
         // It gives: Invalid format. Hint: <paramlist> = <param> [, <paramlist>].
         exec = gEnv->GetValue("WebGui.FirefoxBatch", "fork: -headless -no-remote $url");
      else
         exec = gEnv->GetValue("WebGui.FirefoxInteractive", "$prog -width=$width -height=$height $url");
#else
      if (win.IsBatchMode())
         exec = gEnv->GetValue("WebGui.FirefoxBatch", "fork:-headless -no-remote $url");
      else
         exec = gEnv->GetValue("WebGui.FirefoxInteractive", "$prog -width $width -height $height \'$url\' &");
#endif
   }

   if (!is_firefox && !is_chrome) {
      if (where == "native") {
         R__ERROR_HERE("WebDisplay") << "Neither firefox nor chrome are detected for native display";
         return false;
      }

      if (win.IsBatchMode()) {
         R__ERROR_HERE("WebDisplay") << "To use batch mode 'chrome' or 'firefox' should be configured as output";
         return false;
      }

      if (!is_native) {
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

   if (!CreateHttpServer(true)) {
      R__ERROR_HERE("WebDisplay") << "Fail to start real HTTP server";
      return false;
   }

   addr = fAddr + addr;

   int port = gEnv->GetValue("WebGui.HeadlessPort", 9222);
   exec.ReplaceAll("$port", std::to_string(port).c_str());
   exec.ReplaceAll("$url", addr.c_str());
   exec.ReplaceAll("$width", swidth.c_str());
   exec.ReplaceAll("$height", sheight.c_str());

   if (exec.Index("fork:") == 0) {
      exec.Remove(0, 5);
#if !defined(_MSC_VER)
      std::unique_ptr<TObjArray> args(exec.Tokenize(" "));
      if (!args || (args->GetLast()<=0))
         return false;

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
         return false;
      }
      win.AddKey(key, std::string("pid:") + std::to_string((int)pid));
      return true;
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
      win.AddKey(key, std::string("pid:") + std::to_string((int)pid));
      return true;
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

   win.AddKey(key, where); // for now just application name

   R__DEBUG_HERE("WebDisplay") << "Show web window in browser with:\n" << exec;

#ifdef _MSC_VER
   _spawnv(_P_NOWAIT, prog.Data(), argv.data());
#else
   gSystem->Exec(exec);
#endif

   return true;
}

///////////////////////////////////////////////////////////////////////////////////////////
/// When window connection is closed, correspondent browser application may need to be halted
/// Process id produced by the Show() method

void ROOT::Experimental::TWebWindowsManager::HaltClient(const std::string &procid)
{
   if (procid.find("pid:") != 0) return;

   int pid = std::stoi(procid.substr(4));

#if !defined(_MSC_VER)
   if (pid>0) kill(pid, SIGKILL);
#else
   if (pid > 0) gSystem->Exec(TString::Format("taskkill /F /PID %d", pid));
#endif
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
/// If value is negative, WebGui.WaitForTmout value will be used

int ROOT::Experimental::TWebWindowsManager::WaitFor(TWebWindow &win, WebWindowWaitFunc_t check, bool timed, double timelimit)
{
   int res(0), cnt(0);
   double spent = 0;

   if (timed && (timelimit < 0))
      timelimit = gEnv->GetValue("WebGui.WaitForTmout", 100.);

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
         return 0;

      cnt++;
   }

   // R__DEBUG_HERE("WebDisplay") << "Waiting result " << res << " spent time " << spent << " ntry " << cnt;

   return res;
}

//////////////////////////////////////////////////////////////////////////
/// Terminate http server and ROOT application

void ROOT::Experimental::TWebWindowsManager::Terminate()
{
   if (fServer)
      fServer->SetTerminate();

   // use timer to avoid situation when calling object is deleted by terminate
   if (gApplication)
      TTimer::SingleShot(100, "TApplication", gApplication, "Terminate()");
}
