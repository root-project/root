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
#include "THttpWSHandler.h"

#include "TSystem.h"
#include "TRandom.h"
#include "TString.h"
#include "TStopwatch.h"
#include "TApplication.h"
#include "TTimer.h"
#include "TObjArray.h"
#include "RConfigure.h"
#include "TROOT.h"
#include "TEnv.h"

#include <unistd.h>

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

//////////////////////////////////////////////////////////////////////////////////////////
/// window manager constructor
/// Required here for correct usage of unique_ptr<THttpServer>

ROOT::Experimental::TWebWindowsManager::TWebWindowsManager() = default;

//////////////////////////////////////////////////////////////////////////////////////////
/// window manager destructor
/// Required here for correct usage of unique_ptr<THttpServer>

ROOT::Experimental::TWebWindowsManager::~TWebWindowsManager()
{
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

bool ROOT::Experimental::TWebWindowsManager::CreateHttpServer(bool with_http)
{
   if (!fServer)
      fServer = std::make_unique<THttpServer>("basic_sniffer");

   if (!with_http || !fAddr.empty())
      return true;

   int http_port = gEnv->GetValue("WebGui.HttpPort", 0);
   int http_min = gEnv->GetValue("WebGui.HttpPortMin", 8800);
   int http_max = gEnv->GetValue("WebGui.HttpPortMax", 9800);
   int http_wstmout = gEnv->GetValue("WebGui.HttpWStmout", 10000);
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
/// To show window, TWebWindow::Show() have to be called

std::shared_ptr<ROOT::Experimental::TWebWindow> ROOT::Experimental::TWebWindowsManager::CreateWindow(bool batch_mode)
{
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

   fServer->Register("/web7gui", (THttpWSHandler *)win->fWSHandler.get());

   return win;
}

//////////////////////////////////////////////////////////////////////////////////////////
/// Release all references to specified window
/// Called from TWebWindow destructor

void ROOT::Experimental::TWebWindowsManager::Unregister(ROOT::Experimental::TWebWindow &win)
{
   // TODO: close all active connections of the window

   if (win.fWSHandler)
      fServer->Unregister((THttpWSHandler *)win.fWSHandler.get());

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

   std::string addr = "/web7gui/";

   addr.append(((THttpWSHandler *)win.fWSHandler.get())->GetName());

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
/// Show window in specified location
/// Parameter "where" specifies that kind of window display should be used. Possible values:
///
///      cef - Chromium Embeded Framework, local display, local communication
///      qt5 - Qt5 WebEngine, local display, local communication
///  browser - default system web-browser, communication via configured http port
///  chrome  - use Google Chrome web browser (requires at least v60), supports headless mode,
///            preferable display kind if cef or qt5 are not available
/// chromium - open-source flavor of Chrome, available on most Linux distributions
///   native - either any available local display or default browser
///   <prog> - any program name which will be started instead of default browser, like firefox or /usr/bin/opera
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

   std::string addr = GetUrl(win, false) + "?key=" + key;

   std::string where = _where;
   if (where.empty())
      where = gROOT->GetWebDisplay().Data();

   bool is_native = where.empty() || (where == "native"), is_qt5 = (where == "qt5"), is_cef = (where == "cef"),
        is_chrome = (where == "chrome") || (where == "chromium"), is_firefox = (where == "firefox");

#ifdef R__HAS_CEFWEB
   if (is_native)
      is_cef = true;
#endif

   if (win.IsBatchMode()) {
      if (!is_cef && !is_chrome && !is_firefox) {
         R__ERROR_HERE("WebDisplay")
            << "To use batch mode 'cef' or 'chromium' or 'firefox' should be configured as output";
         return false;
      }
      if (is_cef) {
         const char *displ = gSystem->Getenv("DISPLAY");
         if (!displ || (*displ == 0)) {
            R__ERROR_HERE("WebDisplay") << "To use CEF in batch mode DISPLAY variable should be set."
                                           " See gui/cefdisplay/Readme.md for more info";
            return false;
         }
      }
   }

#ifdef R__HAS_CEFWEB

   const char *cef_path = gSystem->Getenv("CEF_PATH");
   const char *rootsys = gSystem->Getenv("ROOTSYS");
   if (cef_path && !gSystem->AccessPathName(cef_path) && rootsys && (is_native || is_cef)) {

      Func_t symbol_cef = gSystem->DynFindSymbol("*", "webgui_start_browser_in_cef3");

      if (!symbol_cef) {
         gSystem->Load("libROOTCefDisplay");
         // TODO: make minimal C++ interface here
         symbol_cef = gSystem->DynFindSymbol("*", "webgui_start_browser_in_cef3");
      }

      if (symbol_cef) {
         typedef void (*FunctionCef3)(const char *, void *, bool, const char *, const char *, unsigned, unsigned);
         R__DEBUG_HERE("WebDisplay") << "Show window " << addr << " in CEF";
         FunctionCef3 func = (FunctionCef3)symbol_cef;
         func(addr.c_str(), fServer.get(), win.IsBatchMode(), rootsys, cef_path, win.GetWidth(), win.GetHeight());
         win.AddKey(key, "cef");
         return true;
      }
   }

#endif

#ifdef R__HAS_QT5WEB

   if (is_native || is_qt5) {
      Func_t symbol_qt5 = gSystem->DynFindSymbol("*", "webgui_start_browser_in_qt5");

      if (!symbol_qt5) {
         gSystem->Load("libROOTQt5WebDisplay");
         symbol_qt5 = gSystem->DynFindSymbol("*", "webgui_start_browser_in_qt5");
      }
      if (symbol_qt5) {
         typedef void (*FunctionQt5)(const char *, void *, bool, unsigned, unsigned);
         R__DEBUG_HERE("WebDisplay") << "Show window " << addr << " in Qt5 WebEngine";
         FunctionQt5 func = (FunctionQt5)symbol_qt5;
         func(addr.c_str(), fServer.get(), win.IsBatchMode(), win.GetWidth(), win.GetHeight());
         win.AddKey(key, "qt5");
         return true;
      }
   }
#endif

   if (!CreateHttpServer(true)) {
      R__ERROR_HERE("WebDisplay") << "Fail to start real HTTP server";
      return false;
   }

   addr = fAddr + addr;

   TString exec;

   int dbg_port = gEnv->GetValue("WebGui.DebugPort", 0);
   int dbg_min = gEnv->GetValue("WebGui.DebugMin", 40800);
   int dbg_max = gEnv->GetValue("WebGui.DebugMax", 41800);

   if (!dbg_port) {
      dbg_port = dbg_min;
      if (dbg_max > dbg_min)
         dbg_port += (int)((dbg_max - dbg_max) * gRandom->Rndm(1));
   }

   if (!dbg_port)
      dbg_port = 40879;

   std::string swidth = std::to_string(win.GetWidth() ? win.GetWidth() : 800);
   std::string sheight = std::to_string(win.GetHeight() ? win.GetHeight() : 600);
   std::string prog = where;

   if (is_chrome) {
      // see https://peter.sh/experiments/chromium-command-line-switches/

      prog = gEnv->GetValue("WebGui.Chrome", where.c_str());

      if (win.IsBatchMode())
         exec = gEnv->GetValue("WebGui.ChromeBatch", "fork: $prog --headless --disable-gpu --disable-webgl --remote-debugging-port=$dbgport $url");
      else
         exec = gEnv->GetValue("WebGui.ChromeInteractive", "$prog --window-size=$width,$height --app=\'$url\' &");
   } else if (is_firefox) {

      prog = gEnv->GetValue("WebGui.Firefox", where.c_str());

      if (win.IsBatchMode())
         exec = gEnv->GetValue("WebGui.FirefoxBatch", "timeout 30 $prog -headless -window-size=$width,$height \'$url\' &");
      else
         exec = gEnv->GetValue("WebGui.FirefoxInteractive", "$prog -window-size=$width,$height \'$url\' &");

   } else if (!is_native && !is_cef && !is_qt5 && (where != "browser")) {
      if (where.find("$") != std::string::npos) {
         exec = where.c_str();
      } else {
         exec = "$prog $url &";
      }
   } else if (gSystem->InheritsFrom("TMacOSXSystem")) {
      exec = "open \'$url\'";
   } else if (gSystem->InheritsFrom("TWinNTSystem")) {
      exec = "start $url";
   } else {
      exec = "xdg-open \'$url\' &";
   }

   exec.ReplaceAll("$prog", prog.c_str());
   exec.ReplaceAll("$url", addr.c_str());
   exec.ReplaceAll("$width", swidth.c_str());
   exec.ReplaceAll("$height", sheight.c_str());
   exec.ReplaceAll("$dbgport", std::to_string(dbg_port).c_str());

   R__DEBUG_HERE("WebDisplay") << "Show web window in browser with cmd:\n" << exec;

   if (exec.Index("fork:") == 0) {
      exec.Remove(0, 5);

      TObjArray *args = exec.Tokenize(" ");
      if (!args) return false;

      std::vector<char *> argv;
      for (Int_t n=0;n<=args->GetLast();++n) {
         argv.push_back((char *) args->At(n)->GetName());
         printf("Arg %d = %s\n", n, argv[n]);
      }
      argv.push_back(nullptr);

      char dname[100];
      strcpy(dname, "/usr/bin/chromium");

      argv[0] = dname;

      int pid = fork();              //fork child

      if (pid == 0) {
         printf("After fork num args %d\n", (int) argv.size());
         for(int n=0;n<argv.size();++n)
            if (argv[n]) printf("   fork arg %d = %s\n", n, argv[n]);

         execvp(argv[0], argv.data());
         printf("Never come here\n");
         // never come here, dummy exit
         exit(0);
      } else {
         printf("Continue with parent pid: %d \n", pid);
         win.AddKey(key, std::string("pid:") + std::to_string(pid)); // process id
      }

      delete args;

   } else {
      win.AddKey(key, where); // for now just application name
      gSystem->Exec(exec);
   }

   return true;
}

//////////////////////////////////////////////////////////////////////////
/// Waits until provided check function or lambdas returns non-zero value
/// Runs application mainloop and short sleeps in-between
/// timelimit (in seconds) defines how long to wait (0 - forever)
/// Function has following signature: int func(double spent_tm)
/// Parameter spent_tm is time in seconds, which already spent inside function
/// Waiting will be continued, if function returns zero.
/// First non-zero value breaks waiting loop and result is returned (or 0 if time is expired).

int ROOT::Experimental::TWebWindowsManager::WaitFor(WebWindowWaitFunc_t check, double timelimit)
{
   TStopwatch tm;
   tm.Start();
   double spent(0);
   int res(0), cnt(0);

   while ((res = check(spent)) == 0) {
      gSystem->ProcessEvents();
      gSystem->Sleep(10);

      spent = tm.RealTime();
      tm.Continue();
      if ((timelimit > 0) && (spent > timelimit))
         return 0;
      cnt++;
   }

   R__DEBUG_HERE("WebDisplay") << "Waiting result " << res << " spent time " << spent << " ntry " << cnt;

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
