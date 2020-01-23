/// \file RWebDisplayHandle.cxx
/// \ingroup WebGui ROOT7
/// \author Sergey Linev <s.linev@gsi.de>
/// \date 2018-10-17
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/RWebDisplayHandle.hxx>

#include <ROOT/RMakeUnique.hxx>
#include <ROOT/RLogger.hxx>

#include "RConfigure.h"
#include "TSystem.h"
#include "TRandom.h"
#include "TString.h"
#include "TObjArray.h"
#include "TEnv.h"

#include <regex>

#ifdef _MSC_VER
#include <process.h>
#else
#include <unistd.h>
#include <stdlib.h>
#include <signal.h>
#include <spawn.h>
#endif

using namespace std::string_literals;

//////////////////////////////////////////////////////////////////////////////////////////////////
/// Static holder of registered creators of web displays

std::map<std::string, std::unique_ptr<ROOT::Experimental::RWebDisplayHandle::Creator>> &ROOT::Experimental::RWebDisplayHandle::GetMap()
{
   static std::map<std::string, std::unique_ptr<ROOT::Experimental::RWebDisplayHandle::Creator>> sMap;
   return sMap;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/// Search for specific browser creator
/// If not found, try to add one
/// \param name - creator name like ChromeCreator
/// \param libname - shared library name where creator could be provided

std::unique_ptr<ROOT::Experimental::RWebDisplayHandle::Creator> &ROOT::Experimental::RWebDisplayHandle::FindCreator(const std::string &name, const std::string &libname)
{
   auto &m = GetMap();
   auto search = m.find(name);
   if (search == m.end()) {

      if (libname == "ChromeCreator") {
         m.emplace(name, std::make_unique<ChromeCreator>());
      } else if (libname == "FirefoxCreator") {
         m.emplace(name, std::make_unique<FirefoxCreator>());
      } else if (libname == "BrowserCreator") {
         m.emplace(name, std::make_unique<BrowserCreator>(false));
      } else if (!libname.empty()) {
         gSystem->Load(libname.c_str());
      }

      search = m.find(name); // try again
   }

   if (search != m.end())
      return search->second;

   static std::unique_ptr<ROOT::Experimental::RWebDisplayHandle::Creator> dummy;
   return dummy;
}

namespace ROOT {
namespace Experimental {

//////////////////////////////////////////////////////////////////////////////////////////////////
/// Specialized handle to hold information about running browser process
/// Used to correctly cleanup all processes and temporary directories

class RWebBrowserHandle : public RWebDisplayHandle {

#ifdef _MSC_VER
   typedef int browser_process_id;
#else
   typedef pid_t browser_process_id;
#endif
   std::string fTmpDir;
   bool fHasPid{false};
   browser_process_id fPid;

public:
   RWebBrowserHandle(const std::string &url, const std::string &tmpdir) : RWebDisplayHandle(url), fTmpDir(tmpdir) {}

   RWebBrowserHandle(const std::string &url, const std::string &tmpdir, browser_process_id pid)
      : RWebDisplayHandle(url), fTmpDir(tmpdir), fHasPid(true), fPid(pid)
   {
   }

   virtual ~RWebBrowserHandle()
   {
#ifdef _MSC_VER
      if (fHasPid)
         gSystem->Exec(("taskkill /F /PID "s + std::to_string(fPid) + " >NUL 2>NUL").c_str());
      std::string rmdir = "rmdir /S /Q ";
#else
      if (fHasPid)
         kill(fPid, SIGKILL);
      std::string rmdir = "rm -rf ";
#endif
      if (!fTmpDir.empty())
         gSystem->Exec((rmdir + fTmpDir).c_str());
   }
};

} // namespace Experimental
} // namespace ROOT

//////////////////////////////////////////////////////////////////////////////////////////////////
/// Class to handle starting of web-browsers like Chrome or Firefox

ROOT::Experimental::RWebDisplayHandle::BrowserCreator::BrowserCreator(bool custom, const std::string &exec)
{
   if (custom) return;

   if (!exec.empty()) {
      if (exec.find("$url") == std::string::npos) {
         fProg = exec;
#ifdef _MSC_VER
         fExec = exec + " $url";
#else
         fExec = exec + " $url &";
#endif
      } else {
         fExec = exec;
         auto pos = exec.find(" ");
         if (pos != std::string::npos)
            fProg = exec.substr(0, pos);
      }
   } else if (gSystem->InheritsFrom("TMacOSXSystem")) {
      fExec = "open \'$url\'";
   } else if (gSystem->InheritsFrom("TWinNTSystem")) {
      fExec = "start $url";
   } else {
      fExec = "xdg-open \'$url\' &";
   }
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/// Check if browser executable exists and can be used

void ROOT::Experimental::RWebDisplayHandle::BrowserCreator::TestProg(const std::string &nexttry, bool check_std_paths)
{
   if (nexttry.empty() || !fProg.empty())
      return;

   if (!gSystem->AccessPathName(nexttry.c_str(), kExecutePermission)) {
#ifdef R__MACOSX
      fProg = std::regex_replace(nexttry, std::regex("%20"), " ");
#else
      fProg = nexttry;
#endif
      return;
   }

   if (!check_std_paths)
      return;

#ifdef _MSC_VER
   std::string ProgramFiles = gSystem->Getenv("ProgramFiles");
   auto pos = ProgramFiles.find(" (x86)");
   if (pos != std::string::npos)
      ProgramFiles.erase(pos, 6);
   std::string ProgramFilesx86 = gSystem->Getenv("ProgramFiles(x86)");

   if (!ProgramFiles.empty())
      TestProg(ProgramFiles + nexttry, false);
   if (!ProgramFilesx86.empty())
      TestProg(ProgramFilesx86 + nexttry, false);
#endif
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/// Display given URL in web browser

std::unique_ptr<ROOT::Experimental::RWebDisplayHandle>
ROOT::Experimental::RWebDisplayHandle::BrowserCreator::Display(const RWebDisplayArgs &args)
{
   std::string url = args.GetFullUrl();
   if (url.empty())
      return nullptr;

   std::string exec;
   if (args.IsHeadless())
      exec = fBatchExec;
   else if (args.IsStandalone())
      exec = fExec;
   else
#ifdef _MSC_VER
      exec = "$prog $url";
#else
      exec = "$prog $url &";
#endif

   if (exec.empty())
      return nullptr;

   std::string swidth = std::to_string(args.GetWidth() > 0 ? args.GetWidth() : 800),
               sheight = std::to_string(args.GetHeight() > 0 ? args.GetHeight() : 600),
               sposx = std::to_string(args.GetX() >= 0 ? args.GetX() : 0),
               sposy = std::to_string(args.GetY() >= 0 ? args.GetY() : 0);

   ProcessGeometry(exec, args);

   std::string rmdir = MakeProfile(exec, args.IsHeadless());

   exec = std::regex_replace(exec, std::regex("\\$url"), url);
   exec = std::regex_replace(exec, std::regex("\\$width"), swidth);
   exec = std::regex_replace(exec, std::regex("\\$height"), sheight);
   exec = std::regex_replace(exec, std::regex("\\$posx"), sposx);
   exec = std::regex_replace(exec, std::regex("\\$posy"), sposy);

   if (exec.compare(0,5,"fork:") == 0) {
      if (fProg.empty()) {
         R__ERROR_HERE("WebDisplay") << "Fork instruction without executable";
         return nullptr;
      }

      exec.erase(0, 5);

#ifndef _MSC_VER

      std::unique_ptr<TObjArray> fargs(TString(exec.c_str()).Tokenize(" "));
      if (!fargs || (fargs->GetLast()<=0)) {
         R__ERROR_HERE("WebDisplay") << "Fork instruction is empty";
         return nullptr;
      }

      std::vector<char *> argv;
      argv.push_back((char *) fProg.c_str());
      for (Int_t n = 0; n <= fargs->GetLast(); ++n)
         argv.push_back((char *)fargs->At(n)->GetName());
      argv.push_back(nullptr);

      posix_spawn_file_actions_t action;
      posix_spawn_file_actions_init(&action);
      if (!args.GetRedirectOutput().empty())
         if (posix_spawn_file_actions_addopen (&action, 1 /*STDOUT_FILENO*/, args.GetRedirectOutput().c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644) != 0) {
            R__ERROR_HERE("WebDisplay") << "Fail to redirect output for spawn process";
            return nullptr;
         }


      R__DEBUG_HERE("WebDisplay") << "Show web window in browser with posix_spawn:\n" << fProg << " " << exec;

      pid_t pid;
      int status = posix_spawn(&pid, argv[0], &action, nullptr, argv.data(), nullptr);
      if (status != 0) {
         R__ERROR_HERE("WebDisplay") << "Fail to launch " << argv[0];
         return nullptr;
      }

      // add processid and rm dir

      return std::make_unique<RWebBrowserHandle>(url, rmdir, pid);

#else

      if (fProg.empty()) {
         R__ERROR_HERE("WebDisplay") << "No Web browser found";
         return nullptr;
      }

      if (!args.GetRedirectOutput().empty()) {
         // use simple redirection, not found solution with wmic

         exec = "\""s + gSystem->UnixPathName(fProg.c_str()) + "\" "s + exec + " > "s + args.GetRedirectOutput();

         gSystem->Exec(exec.c_str());

         return std::make_unique<RWebBrowserHandle>(url, rmdir);
      }

      // use UnixPathName to simplify handling of backslashes
      exec = "wmic process call create '"s + gSystem->UnixPathName(fProg.c_str()) + exec + "' | find \"ProcessId\" "s;
      std::string process_id = gSystem->GetFromPipe(exec.c_str());
      std::stringstream ss(process_id);
      std::string tmp;
      char c;
      int pid = 0;
      ss >> tmp >> c >> pid;

      if (pid <= 0) {
         R__ERROR_HERE("WebDisplay") << "Fail to launch " << fProg;
         return nullptr;
      }

      // add processid and rm dir
      return std::make_unique<RWebBrowserHandle>(url, rmdir, pid);
#endif
   }

#ifdef _MSC_VER
   std::vector<char *> argv;
   std::string firstarg = fProg;
   auto slashpos = firstarg.find_last_of("/\\");
   if (slashpos != std::string::npos)
      firstarg.erase(0, slashpos + 1);
   argv.push_back((char *)firstarg.c_str());

   std::unique_ptr<TObjArray> fargs(TString(exec.c_str()).Tokenize(" "));
   for (Int_t n = 1; n <= fargs->GetLast(); ++n)
      argv.push_back((char *)fargs->At(n)->GetName());
   argv.push_back(nullptr);

   R__DEBUG_HERE("WebDisplay") << "Showing web window in " << fProg << " with:\n" << exec;

   _spawnv(_P_NOWAIT, fProg.c_str(), argv.data());

#else

#ifdef R__MACOSX
   std::string prog = std::regex_replace(fProg, std::regex(" "), "\\ ");
#else
   std::string prog = fProg;
#endif

   exec = std::regex_replace(exec, std::regex("\\$prog"), prog);

   R__DEBUG_HERE("WebDisplay") << "Showing web window in browser with:\n" << exec;

   gSystem->Exec(exec.c_str());
#endif

   // add rmdir if required
   return std::make_unique<RWebBrowserHandle>(url, rmdir);
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/// Constructor

ROOT::Experimental::RWebDisplayHandle::ChromeCreator::ChromeCreator() : BrowserCreator(true)
{
   TestProg(gEnv->GetValue("WebGui.Chrome", ""));

#ifdef _MSC_VER
   TestProg("\\Google\\Chrome\\Application\\chrome.exe", true);
#endif
#ifdef R__MACOSX
   TestProg("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome");
#endif
#ifdef R__LINUX
   TestProg("/usr/bin/chromium");
   TestProg("/usr/bin/chromium-browser");
   TestProg("/usr/bin/chrome-browser");
#endif

#ifdef _MSC_VER
   fBatchExec = gEnv->GetValue("WebGui.ChromeBatch", "fork: --headless --disable-gpu $geometry $url");
   fExec = gEnv->GetValue("WebGui.ChromeInteractive", "$prog $geometry --no-first-run --app=$url");
#else
   fBatchExec = gEnv->GetValue("WebGui.ChromeBatch", "fork:--headless --incognito $geometry $url");
   fExec = gEnv->GetValue("WebGui.ChromeInteractive", "$prog $geometry --no-first-run --incognito --app=\'$url\' &");
#endif
}


//////////////////////////////////////////////////////////////////////////////////////////////////
/// Replace $geometry placeholder with geometry settings
/// Also RWebDisplayArgs::GetExtraArgs() are appended

void ROOT::Experimental::RWebDisplayHandle::ChromeCreator::ProcessGeometry(std::string &exec, const RWebDisplayArgs &args)
{
   std::string geometry;
   if ((args.GetWidth() > 0) && (args.GetHeight() > 0))
      geometry = "--window-size="s + std::to_string(args.GetWidth())
                                   + (args.IsHeadless() ? "x"s : ","s)
                                   + std::to_string(args.GetHeight());

   if (((args.GetX() >= 0) || (args.GetY() >= 0)) && !args.IsHeadless()) {
      if (!geometry.empty()) geometry.append(" ");
      geometry.append("--window-position="s + std::to_string(args.GetX() >= 0 ? args.GetX() : 0) + ","s +
                                           std::to_string(args.GetY() >= 0 ? args.GetY() : 0));
   }

   if (!args.GetExtraArgs().empty()) {
      if (!geometry.empty()) geometry.append(" ");
      geometry.append(args.GetExtraArgs());
   }

   exec = std::regex_replace(exec, std::regex("\\$geometry"), geometry);
}


//////////////////////////////////////////////////////////////////////////////////////////////////
/// Handle profile argument

std::string ROOT::Experimental::RWebDisplayHandle::ChromeCreator::MakeProfile(std::string &exec, bool)
{
   std::string rmdir, profile_arg;

   if (exec.find("$profile") == std::string::npos)
      return rmdir;

   const char *chrome_profile = gEnv->GetValue("WebGui.ChromeProfile", "");
   if (chrome_profile && *chrome_profile) {
      profile_arg = chrome_profile;
   } else {
      gRandom->SetSeed(0);
      rmdir = profile_arg = std::string(gSystem->TempDirectory()) + "/root_chrome_profile_"s + std::to_string(gRandom->Integer(0x100000));
   }

   exec = std::regex_replace(exec, std::regex("\\$profile"), profile_arg);

   return rmdir;
}


//////////////////////////////////////////////////////////////////////////////////////////////////
/// Constructor

ROOT::Experimental::RWebDisplayHandle::FirefoxCreator::FirefoxCreator() : BrowserCreator(true)
{
   TestProg(gEnv->GetValue("WebGui.Firefox", ""));

#ifdef _MSC_VER
   TestProg("\\Mozilla Firefox\\firefox.exe", true);
#endif
#ifdef R__MACOSX
   TestProg("/Applications/Firefox.app/Contents/MacOS/firefox");
#endif
#ifdef R__LINUX
   TestProg("/usr/bin/firefox");
#endif

#ifdef _MSC_VER
   // there is a problem when specifying the window size with wmic on windows:
   // It gives: Invalid format. Hint: <paramlist> = <param> [, <paramlist>].
   fBatchExec = gEnv->GetValue("WebGui.FirefoxBatch", "fork: -headless -no-remote $profile $url");
   fExec = gEnv->GetValue("WebGui.FirefoxInteractive", "$prog -no-remote $profile $url");
#else
   fBatchExec = gEnv->GetValue("WebGui.FirefoxBatch", "fork:--headless --private-window --no-remote $profile $url");
   fExec = gEnv->GetValue("WebGui.FirefoxInteractive", "$prog --private-window \'$url\' &");
#endif
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/// Create Firefox profile to run independent browser window

std::string ROOT::Experimental::RWebDisplayHandle::FirefoxCreator::MakeProfile(std::string &exec, bool batch_mode)
{
   std::string rmdir, profile_arg;

   if (exec.find("$profile") == std::string::npos)
      return rmdir;

   const char *ff_profile = gEnv->GetValue("WebGui.FirefoxProfile", "");
   const char *ff_profilepath = gEnv->GetValue("WebGui.FirefoxProfilePath", "");
   Int_t ff_randomprofile = gEnv->GetValue("WebGui.FirefoxRandomProfile", (Int_t) 0);
   if (ff_profile && *ff_profile) {
      profile_arg = "-P "s + ff_profile;
   } else if (ff_profilepath && *ff_profilepath) {
      profile_arg = "-profile "s + ff_profilepath;
   } else if ((ff_randomprofile > 0) || (batch_mode && (ff_randomprofile >= 0))) {

      gRandom->SetSeed(0);
      std::string rnd_profile = "root_ff_profile_"s + std::to_string(gRandom->Integer(0x100000));
      std::string profile_dir = std::string(gSystem->TempDirectory()) + "/"s + rnd_profile;

      profile_arg = "-profile "s + profile_dir;

      if (gSystem->mkdir(profile_dir.c_str()) == 0) {
         rmdir = profile_dir;
      } else {
         R__ERROR_HERE("WebDisplay") << "Cannot create Firefox profile directory " << profile_dir;
      }
   }

   exec = std::regex_replace(exec, std::regex("\\$profile"), profile_arg);

   return rmdir;
}


///////////////////////////////////////////////////////////////////////////////////////////////////
/// Create web display
/// \param args - defines where and how to display web window
/// Returns RWebDisplayHandle, which holds information of running browser application
/// Can be used fully independent from RWebWindow classes just to show any web page

std::unique_ptr<ROOT::Experimental::RWebDisplayHandle> ROOT::Experimental::RWebDisplayHandle::Display(const RWebDisplayArgs &args)
{
   std::unique_ptr<RWebDisplayHandle> handle;

   auto try_creator = [&](std::unique_ptr<Creator> &creator) {
      if (!creator || !creator->IsActive())
         return false;
      handle = creator->Display(args);
      return handle ? true : false;
   };

   if ((args.GetBrowserKind() == RWebDisplayArgs::kLocal) || (args.GetBrowserKind() == RWebDisplayArgs::kCEF)) {
      if (try_creator(FindCreator("cef", "libROOTCefDisplay")))
         return handle;
   }

   if ((args.GetBrowserKind() == RWebDisplayArgs::kLocal) || (args.GetBrowserKind() == RWebDisplayArgs::kQt5)) {
      if (try_creator(FindCreator("qt5", "libROOTQt5WebDisplay")))
         return handle;
   }

   if (args.IsLocalDisplay()) {
      R__ERROR_HERE("WebDisplay") << "Neither Qt5 nor CEF libraries were found to provide local display";
      return handle;
   }

   if ((args.GetBrowserKind() == RWebDisplayArgs::kNative) || (args.GetBrowserKind() == RWebDisplayArgs::kChrome)) {
      if (try_creator(FindCreator("chrome", "ChromeCreator")))
         return handle;
   }

   if ((args.GetBrowserKind() == RWebDisplayArgs::kNative) || (args.GetBrowserKind() == RWebDisplayArgs::kFirefox)) {
      if (try_creator(FindCreator("firefox", "FirefoxCreator")))
         return handle;
   }

   if ((args.GetBrowserKind() == RWebDisplayArgs::kChrome) || (args.GetBrowserKind() == RWebDisplayArgs::kFirefox)) {
      R__ERROR_HERE("WebDisplay") << "Neither Chrome nor Firefox browser cannot be started to provide display";
      return handle;
   }

   if ((args.GetBrowserKind() == RWebDisplayArgs::kCustom)) {
      std::unique_ptr<Creator> creator = std::make_unique<BrowserCreator>(false, args.GetCustomExec());
      try_creator(creator);
   } else {
      try_creator(FindCreator("browser", "BrowserCreator"));
   }

   return handle;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
/// Display provided url in configured web browser
/// \param url - specified URL address like https://root.cern
/// Browser can specified when starting `root --web=firefox`
/// Returns true when browser started
/// It is convenience method, equivalent to:
///  ~~~
///     RWebDisplayArgs args;
///     args.SetUrl(url);
///     args.SetStandalone(false);
///     auto handle = RWebDisplayHandle::Display(args);
/// ~~~

bool ROOT::Experimental::RWebDisplayHandle::DisplayUrl(const std::string &url)
{
   RWebDisplayArgs args;
   args.SetUrl(url);
   args.SetStandalone(false);

   auto handle = Display(args);

   return !!handle;
}
