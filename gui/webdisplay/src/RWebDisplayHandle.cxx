// Author: Sergey Linev <s.linev@gsi.de>
// Date: 2018-10-17
// Warning: This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/RWebDisplayHandle.hxx>

#include <ROOT/RLogger.hxx>

#include "RConfigure.h"
#include "TSystem.h"
#include "TRandom3.h"
#include "TString.h"
#include "TObjArray.h"
#include "THttpServer.h"
#include "TEnv.h"
#include "TError.h"
#include "TROOT.h"
#include "TBase64.h"
#include "TBufferJSON.h"
#include "RWebWindowWSHandler.hxx"

#include <fstream>
#include <iostream>
#include <filesystem>
#include <memory>
#include <regex>

#ifdef _MSC_VER
#include <process.h>
#else
#include <unistd.h>
#include <stdlib.h>
#include <signal.h>
#include <spawn.h>
#ifdef R__MACOSX
#include <sys/wait.h>
#include <crt_externs.h>
#elif defined(__FreeBSD__)
#include <sys/wait.h>
#include <dlfcn.h>
#else
#include <wait.h>
#endif
#endif

using namespace ROOT;
using namespace std::string_literals;

/** \class ROOT::RWebDisplayHandle
\ingroup webdisplay

Handle of created web-based display
Depending from type of web display, holds handle of started browser process or other display-specific information
to correctly stop and cleanup display.
*/


//////////////////////////////////////////////////////////////////////////////////////////////////
/// Static holder of registered creators of web displays

std::map<std::string, std::unique_ptr<RWebDisplayHandle::Creator>> &RWebDisplayHandle::GetMap()
{
   static std::map<std::string, std::unique_ptr<RWebDisplayHandle::Creator>> sMap;
   return sMap;
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/// Search for specific browser creator
/// If not found, try to add one
/// \param name - creator name like ChromeCreator
/// \param libname - shared library name where creator could be provided

std::unique_ptr<RWebDisplayHandle::Creator> &RWebDisplayHandle::FindCreator(const std::string &name, const std::string &libname)
{
   auto &m = GetMap();
   auto search = m.find(name);
   if (search == m.end()) {

      if (libname == "ChromeCreator") {
         m.emplace(name, std::make_unique<ChromeCreator>(name == "edge"));
      } else if (libname == "FirefoxCreator") {
         m.emplace(name, std::make_unique<FirefoxCreator>());
      } else if (libname == "SafariCreator") {
         m.emplace(name, std::make_unique<SafariCreator>());
      } else if (libname == "BrowserCreator") {
         m.emplace(name, std::make_unique<BrowserCreator>(false));
      } else if (!libname.empty()) {
         gSystem->Load(libname.c_str());
      }

      search = m.find(name); // try again
   }

   if (search != m.end())
      return search->second;

   static std::unique_ptr<RWebDisplayHandle::Creator> dummy;
   return dummy;
}

namespace ROOT {

//////////////////////////////////////////////////////////////////////////////////////////////////
/// Specialized handle to hold information about running browser process
/// Used to correctly cleanup all processes and temporary directories

class RWebBrowserHandle : public RWebDisplayHandle {

#ifdef _MSC_VER
   typedef int browser_process_id;
#else
   typedef pid_t browser_process_id;
#endif
   std::string fTmpDir;         ///< temporary directory to delete at the end
   std::string fTmpFile;        ///< temporary file to remove
   bool fHasPid{false};
   browser_process_id fPid;

public:
   RWebBrowserHandle(const std::string &url, const std::string &tmpdir, const std::string &tmpfile,
                     const std::string &dump)
      : RWebDisplayHandle(url), fTmpDir(tmpdir), fTmpFile(tmpfile)
   {
      SetContent(dump);
   }

   RWebBrowserHandle(const std::string &url, const std::string &tmpdir, const std::string &tmpfile,
                     browser_process_id pid)
      : RWebDisplayHandle(url), fTmpDir(tmpdir), fTmpFile(tmpfile), fHasPid(true), fPid(pid)
   {
   }

   ~RWebBrowserHandle() override
   {
#ifdef _MSC_VER
      if (fHasPid)
         gSystem->Exec(("taskkill /F /PID " + std::to_string(fPid) + " >NUL 2>NUL").c_str());
      std::string rmdir = "rmdir /S /Q ";
#else
      if (fHasPid)
         kill(fPid, SIGKILL);
      std::string rmdir = "rm -rf ";
#endif
      if (!fTmpDir.empty())
         gSystem->Exec((rmdir + fTmpDir).c_str());
      RemoveStartupFiles();
   }

   void RemoveStartupFiles() override
   {
#ifdef _MSC_VER
      std::string rmfile = "del /F ";
#else
      std::string rmfile = "rm -f ";
#endif
      if (!fTmpFile.empty()) {
         gSystem->Exec((rmfile + fTmpFile).c_str());
         fTmpFile.clear();
      }
   }
};

} // namespace ROOT

//////////////////////////////////////////////////////////////////////////////////////////////////
/// Class to handle starting of web-browsers like Chrome or Firefox

RWebDisplayHandle::BrowserCreator::BrowserCreator(bool custom, const std::string &exec)
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

void RWebDisplayHandle::BrowserCreator::TestProg(const std::string &nexttry, bool check_std_paths)
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
/// Create temporary file for web display
/// Normally gSystem->TempFileName() method used to create file in default temporary directory
/// For snap chromium use of default temp directory is not always possible therefore one switches to home directory
/// But one checks if default temp directory modified and already points to /home folder

FILE *RWebDisplayHandle::BrowserCreator::TemporaryFile(TString &name, int use_home_dir, const char *suffix)
{
   std::string dirname;
   if (use_home_dir > 0) {
      if (use_home_dir == 1) {
         const char *tmp_dir = gSystem->TempDirectory();
         if (tmp_dir && (strncmp(tmp_dir, "/home", 5) == 0))
            use_home_dir = 0;
         else if (!tmp_dir || (strncmp(tmp_dir, "/tmp", 4) == 0))
            use_home_dir = 2;
      }

      if (use_home_dir > 1)
         dirname = gSystem->GetHomeDirectory();
   }
   return gSystem->TempFileName(name, use_home_dir > 1 ? dirname.c_str() : nullptr, suffix);
}

static void DummyTimeOutHandler(int /* Sig */) {}


//////////////////////////////////////////////////////////////////////////////////////////////////
/// Display given URL in web browser

std::unique_ptr<RWebDisplayHandle>
RWebDisplayHandle::BrowserCreator::Display(const RWebDisplayArgs &args)
{
   std::string url = args.GetFullUrl();
   if (url.empty())
      return nullptr;

   if(args.GetBrowserKind() == RWebDisplayArgs::kServer) {
      std::cout << "New web window: " << url << std::endl;
      return std::make_unique<RWebBrowserHandle>(url, "", "", "");
   }

   std::string exec;
   if (args.IsBatchMode())
      exec = fBatchExec;
   else if (args.IsHeadless())
      exec = fHeadlessExec;
   else if (args.IsStandalone())
      exec = fExec;
   else
      exec = "$prog $url &";

   if (exec.empty())
      return nullptr;

   std::string swidth = std::to_string(args.GetWidth() > 0 ? args.GetWidth() : 800),
               sheight = std::to_string(args.GetHeight() > 0 ? args.GetHeight() : 600),
               sposx = std::to_string(args.GetX() >= 0 ? args.GetX() : 0),
               sposy = std::to_string(args.GetY() >= 0 ? args.GetY() : 0);

   ProcessGeometry(exec, args);

   std::string rmdir = MakeProfile(exec, args.IsBatchMode() || args.IsHeadless());

   std::string tmpfile;

   // these are secret parameters, hide them in temp file
   if (((url.find("token=") != std::string::npos) || (url.find("key=") != std::string::npos)) && !args.IsBatchMode() && !args.IsHeadless()) {
      TString filebase = "root_start_";

      auto f = TemporaryFile(filebase, IsSnapBrowser() ? 1 : 0, ".html");

      bool ferr = false;

      if (!f) {
         ferr = true;
      } else {
         std::string content = std::regex_replace(
            "<!DOCTYPE html>\n"
            "<html lang=\"en\">\n"
            "<head>\n"
            "   <meta charset=\"utf-8\">\n"
            "   <meta http-equiv=\"refresh\" content=\"0;url=$url\"/>\n"
            "   <title>Opening ROOT widget</title>\n"
            "</head>\n"
            "<body>\n"
            "<p>\n"
            "  This page should redirect you to a ROOT widget. If it doesn't,\n"
            "  <a href=\"$url\">click here to go to ROOT</a>.\n"
            "</p>\n"
            "</body>\n"
            "</html>\n", std::regex("\\$url"), url);

         if (fwrite(content.c_str(), 1, content.length(), f) != content.length())
            ferr = true;

         if (fclose(f) != 0)
            ferr = true;

         tmpfile = filebase.Data();

         url = "file://"s + tmpfile;
      }

      if (ferr) {
         if (!tmpfile.empty())
            gSystem->Unlink(tmpfile.c_str());
         R__LOG_ERROR(WebGUILog()) << "Fail to create temporary HTML file to startup widget";
         return nullptr;
      }
   }

   exec = std::regex_replace(exec, std::regex("\\$rootetcdir"), TROOT::GetEtcDir().Data());
   exec = std::regex_replace(exec, std::regex("\\$url"), url);
   exec = std::regex_replace(exec, std::regex("\\$width"), swidth);
   exec = std::regex_replace(exec, std::regex("\\$height"), sheight);
   exec = std::regex_replace(exec, std::regex("\\$posx"), sposx);
   exec = std::regex_replace(exec, std::regex("\\$posy"), sposy);

   if (exec.compare(0,5,"fork:") == 0) {
      if (fProg.empty()) {
         if (!tmpfile.empty())
            gSystem->Unlink(tmpfile.c_str());
         R__LOG_ERROR(WebGUILog()) << "Fork instruction without executable";
         return nullptr;
      }

      exec.erase(0, 5);

      // in case of redirection process will wait until output is produced
      std::string redirect = args.GetRedirectOutput();

#ifndef _MSC_VER

      std::unique_ptr<TObjArray> fargs(TString(exec.c_str()).Tokenize(" "));
      if (!fargs || (fargs->GetLast()<=0)) {
         if (!tmpfile.empty())
            gSystem->Unlink(tmpfile.c_str());
         R__LOG_ERROR(WebGUILog()) << "Fork instruction is empty";
         return nullptr;
      }

      std::vector<char *> argv;
      argv.push_back((char *) fProg.c_str());
      for (Int_t n = 0; n <= fargs->GetLast(); ++n)
         argv.push_back((char *)fargs->At(n)->GetName());
      argv.push_back(nullptr);

      R__LOG_DEBUG(0, WebGUILog()) << "Show web window in browser with posix_spawn:\n" << fProg << " " << exec;

      posix_spawn_file_actions_t action;
      posix_spawn_file_actions_init(&action);
      if (redirect.empty())
         posix_spawn_file_actions_addopen(&action, STDOUT_FILENO, "/dev/null", O_WRONLY|O_APPEND, 0);
      else
         posix_spawn_file_actions_addopen(&action, STDOUT_FILENO, redirect.c_str(), O_WRONLY|O_CREAT, 0600);
      posix_spawn_file_actions_addopen(&action, STDERR_FILENO, "/dev/null", O_WRONLY|O_APPEND, 0);

#ifdef R__MACOSX
      char **envp = *_NSGetEnviron();
#elif defined (__FreeBSD__)
      //this is needed because the FreeBSD linker does not like to resolve these special symbols
      //in shared libs with -Wl,--no-undefined
      char** envp = (char**)dlsym(RTLD_DEFAULT, "environ");
#else
      char **envp = environ;
#endif

      pid_t pid;
      int status = posix_spawn(&pid, argv[0], &action, nullptr, argv.data(), envp);

      posix_spawn_file_actions_destroy(&action);

      if (status != 0) {
         if (!tmpfile.empty())
            gSystem->Unlink(tmpfile.c_str());
         R__LOG_ERROR(WebGUILog()) << "Fail to launch " << argv[0];
         return nullptr;
      }

      if (!redirect.empty()) {
         Int_t batch_timeout = gEnv->GetValue("WebGui.BatchTimeout", 30);
         struct sigaction Act, Old;
         int elapsed_time = 0;

         if (batch_timeout) {
            memset(&Act, 0, sizeof(Act));
            Act.sa_handler = DummyTimeOutHandler;
            sigemptyset(&Act.sa_mask);
            sigaction(SIGALRM, &Act, &Old);
            int alarm_timeout = batch_timeout > 3 ? 3 : batch_timeout;
            alarm(alarm_timeout);
            elapsed_time = alarm_timeout;
         }

         int job_done = 0;
         std::string dump_content;

         while (!job_done) {

            // wait until output is produced
            int wait_status = 0;

            auto wait_res = waitpid(pid, &wait_status, WUNTRACED | WCONTINUED);

            // try read dump anyway
            dump_content = THttpServer::ReadFileContent(redirect.c_str());

            if (dump_content.find("<div>###batch###job###done###</div>") != std::string::npos)
               job_done = 1;

            if (wait_res == -1) {
               // failure when finish process
               int alarm_timeout = batch_timeout - elapsed_time;
               if ((errno == EINTR) && (alarm_timeout > 0) && !job_done) {
                  if (alarm_timeout > 2) alarm_timeout = 2;
                  elapsed_time += alarm_timeout;
                  alarm(alarm_timeout);
               } else {
                  // end of timeout - do not try to wait any longer
                  job_done = 1;
               }
            } else if (!WIFEXITED(wait_status) && !WIFSIGNALED(wait_status)) {
               // abnormal end of browser process
               job_done = 1;
            } else {
               // this is normal finish, no need for process kill
               job_done = 2;
            }
         }

         if (job_done != 2) {
            // kill browser process when no normal end was detected
            kill(pid, SIGKILL);
         }

         if (batch_timeout) {
            alarm(0); // disable alarm
            sigaction(SIGALRM, &Old, nullptr);
         }

         if (gEnv->GetValue("WebGui.PreserveBatchFiles", -1) > 0)
            ::Info("RWebDisplayHandle::Display", "Preserve dump file %s", redirect.c_str());
         else
            gSystem->Unlink(redirect.c_str());

         return std::make_unique<RWebBrowserHandle>(url, rmdir, tmpfile, dump_content);
      }

      // add processid and rm dir

      return std::make_unique<RWebBrowserHandle>(url, rmdir, tmpfile, pid);

#else

      if (fProg.empty()) {
         if (!tmpfile.empty())
            gSystem->Unlink(tmpfile.c_str());
         R__LOG_ERROR(WebGUILog()) << "No Web browser found";
         return nullptr;
      }

      // use UnixPathName to simplify handling of backslashes
      exec = "wmic process call create '"s + gSystem->UnixPathName(fProg.c_str()) + " " + exec + "' | find \"ProcessId\" "s;
      std::string process_id = gSystem->GetFromPipe(exec.c_str()).Data();
      std::stringstream ss(process_id);
      std::string tmp;
      char c;
      int pid = 0;
      ss >> tmp >> c >> pid;

      if (pid <= 0) {
         if (!tmpfile.empty())
            gSystem->Unlink(tmpfile.c_str());
         R__LOG_ERROR(WebGUILog()) << "Fail to launch " << fProg;
         return nullptr;
      }

      // add processid and rm dir
      return std::make_unique<RWebBrowserHandle>(url, rmdir, tmpfile, pid);
#endif
   }

#ifdef _MSC_VER

   if (exec.rfind("&") == exec.length() - 1) {

      // if last symbol is &, use _spawn to detach execution
      exec.resize(exec.length() - 1);

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

      R__LOG_DEBUG(0, WebGUILog()) << "Showing web window in " << fProg << " with:\n" << exec;

      _spawnv(_P_NOWAIT, gSystem->UnixPathName(fProg.c_str()), argv.data());

      return std::make_unique<RWebBrowserHandle>(url, rmdir, tmpfile, ""s);
   }

   std::string prog = "\""s + gSystem->UnixPathName(fProg.c_str()) + "\""s;

#else

#ifdef R__MACOSX
   std::string prog = std::regex_replace(fProg, std::regex(" "), "\\ ");
#else
   std::string prog = fProg;
#endif

#endif

   exec = std::regex_replace(exec, std::regex("\\$prog"), prog);

   std::string redirect = args.GetRedirectOutput(), dump_content;

   if (!redirect.empty()) {
      if (exec.find("$dumpfile") != std::string::npos) {
         exec = std::regex_replace(exec, std::regex("\\$dumpfile"), redirect);
      } else {
         auto p = exec.length();
         if (exec.rfind("&") == p-1) --p;
         exec.insert(p, " >"s + redirect + " "s);
      }
   }

   R__LOG_DEBUG(0, WebGUILog()) << "Showing web window in browser with:\n" << exec;

   gSystem->Exec(exec.c_str());

   // read content of redirected output
   if (!redirect.empty()) {
      dump_content = THttpServer::ReadFileContent(redirect.c_str());

      if (gEnv->GetValue("WebGui.PreserveBatchFiles", -1) > 0)
         ::Info("RWebDisplayHandle::Display", "Preserve dump file %s", redirect.c_str());
      else
         gSystem->Unlink(redirect.c_str());
   }

   return std::make_unique<RWebBrowserHandle>(url, rmdir, tmpfile, dump_content);
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/// Constructor

RWebDisplayHandle::SafariCreator::SafariCreator() : BrowserCreator(true)
{
   fExec = gEnv->GetValue("WebGui.SafariInteractive", "open -a Safari $url");
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/// Returns true if it can be used

bool RWebDisplayHandle::SafariCreator::IsActive() const
{
#ifdef R__MACOSX
   return true;
#else
   return false;
#endif
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/// Constructor

RWebDisplayHandle::ChromeCreator::ChromeCreator(bool _edge) : BrowserCreator(true)
{
   fEdge = _edge;

   fEnvPrefix = fEdge ? "WebGui.Edge" : "WebGui.Chrome";

   TestProg(gEnv->GetValue(fEnvPrefix.c_str(), ""));

   if (!fProg.empty() && !fEdge)
      fChromeVersion = gEnv->GetValue("WebGui.ChromeVersion", -1);

#ifdef _MSC_VER
   if (fEdge)
      TestProg("\\Microsoft\\Edge\\Application\\msedge.exe", true);
   else
      TestProg("\\Google\\Chrome\\Application\\chrome.exe", true);
#endif
#ifdef R__MACOSX
   TestProg("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome");
#endif
#ifdef R__LINUX
   TestProg("/snap/bin/chromium"); // test snap before to detect it properly
   TestProg("/usr/bin/chromium");
   TestProg("/usr/bin/chromium-browser");
   TestProg("/usr/bin/chrome-browser");
   TestProg("/usr/bin/google-chrome-stable");
   TestProg("/usr/bin/google-chrome");
#endif

// --no-sandbox is required to run chrome with super-user, but only in headless mode
// --headless=new was used when both old and new were available, but old was removed from chrome 132, see https://developer.chrome.com/blog/removing-headless-old-from-chrome

#ifdef _MSC_VER
   // here --headless=old was used to let normally end of Edge process when --dump-dom is used
   // while on Windows chrome and edge version not tested, just suppose that newest chrome is used
   fBatchExec = gEnv->GetValue((fEnvPrefix + "Batch").c_str(), "$prog --headless --no-sandbox $geometry --dump-dom $url");
   // in interactive headless mode fork used to let stop browser via process id
   fHeadlessExec = gEnv->GetValue((fEnvPrefix + "Headless").c_str(), "fork:--headless --no-sandbox --disable-gpu $geometry \"$url\"");
   fExec = gEnv->GetValue((fEnvPrefix + "Interactive").c_str(), "$prog $geometry --new-window --app=$url &"); // & in windows mean usage of spawn
#else
#ifdef R__MACOSX
   bool use_normal = true; // mac does not like new flag
#else
   bool use_normal = (fChromeVersion < 119) || (fChromeVersion > 131);
#endif
   if (use_normal) {
      // old or newest browser with standard headless mode
      fBatchExec = gEnv->GetValue((fEnvPrefix + "Batch").c_str(), "fork:--headless --no-sandbox --disable-extensions --disable-audio-output $geometry --dump-dom $url");
      fHeadlessExec = gEnv->GetValue((fEnvPrefix + "Headless").c_str(), "fork:--headless --no-sandbox --disable-extensions --disable-audio-output $geometry $url");
   } else {
      // newer version with headless=new mode
      fBatchExec = gEnv->GetValue((fEnvPrefix + "Batch").c_str(), "fork:--headless=new --no-sandbox --disable-extensions --disable-audio-output $geometry --dump-dom $url");
      fHeadlessExec = gEnv->GetValue((fEnvPrefix + "Headless").c_str(), "fork:--headless=new --no-sandbox --disable-extensions --disable-audio-output $geometry $url");
   }
   fExec = gEnv->GetValue((fEnvPrefix + "Interactive").c_str(), "$prog $geometry --new-window --app=\'$url\' >/dev/null 2>/dev/null &");
#endif
}


//////////////////////////////////////////////////////////////////////////////////////////////////
/// Replace $geometry placeholder with geometry settings
/// Also RWebDisplayArgs::GetExtraArgs() are appended

void RWebDisplayHandle::ChromeCreator::ProcessGeometry(std::string &exec, const RWebDisplayArgs &args)
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

std::string RWebDisplayHandle::ChromeCreator::MakeProfile(std::string &exec, bool)
{
   std::string rmdir, profile_arg;

   if (exec.find("$profile") == std::string::npos)
      return rmdir;

   const char *chrome_profile = gEnv->GetValue((fEnvPrefix + "Profile").c_str(), "");
   if (chrome_profile && *chrome_profile) {
      profile_arg = chrome_profile;
   } else {
      TRandom3 rnd;
      rnd.SetSeed(0);
      profile_arg = gSystem->TempDirectory();
      if ((profile_arg.compare(0, 4, "/tmp") == 0) && IsSnapBrowser())
         profile_arg = gSystem->GetHomeDirectory();

#ifdef _MSC_VER
      char slash = '\\';
#else
      char slash = '/';
#endif
      if (!profile_arg.empty() && (profile_arg[profile_arg.length()-1] != slash))
         profile_arg += slash;
      profile_arg += "root_chrome_profile_"s + std::to_string(rnd.Integer(0x100000));

      rmdir = profile_arg;
   }

   exec = std::regex_replace(exec, std::regex("\\$profile"), profile_arg);

   return rmdir;
}


//////////////////////////////////////////////////////////////////////////////////////////////////
/// Constructor

RWebDisplayHandle::FirefoxCreator::FirefoxCreator() : BrowserCreator(true)
{
   TestProg(gEnv->GetValue("WebGui.Firefox", ""));

#ifdef _MSC_VER
   TestProg("\\Mozilla Firefox\\firefox.exe", true);
#endif
#ifdef R__MACOSX
   TestProg("/Applications/Firefox.app/Contents/MacOS/firefox");
#endif
#ifdef R__LINUX
   TestProg("/snap/bin/firefox");
   TestProg("/usr/bin/firefox");
   TestProg("/usr/bin/firefox-bin");
#endif

#ifdef _MSC_VER
   // there is a problem when specifying the window size with wmic on windows:
   // It gives: Invalid format. Hint: <paramlist> = <param> [, <paramlist>].
   fBatchExec = gEnv->GetValue("WebGui.FirefoxBatch", "$prog -headless -no-remote $profile $url");
   fHeadlessExec = gEnv->GetValue("WebGui.FirefoxHeadless", "fork:-headless -no-remote $profile \"$url\"");
   fExec = gEnv->GetValue("WebGui.FirefoxInteractive", "$prog -no-remote $profile $geometry $url &");
#else
   fBatchExec = gEnv->GetValue("WebGui.FirefoxBatch", "fork:--headless -no-remote -new-instance $profile $url");
   fHeadlessExec = gEnv->GetValue("WebGui.FirefoxHeadless", "fork:--headless -no-remote $profile --private-window $url");
   fExec = gEnv->GetValue("WebGui.FirefoxInteractive", "$rootetcdir/runfirefox.sh __nodump__ $cleanup_profile $prog -no-remote $profile $geometry -url \'$url\' &");
#endif
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/// Process window geometry for Firefox

void RWebDisplayHandle::FirefoxCreator::ProcessGeometry(std::string &exec, const RWebDisplayArgs &args)
{
   std::string geometry;
   if ((args.GetWidth() > 0) && (args.GetHeight() > 0) && !args.IsHeadless())
      geometry = "-width="s + std::to_string(args.GetWidth()) + " -height=" + std::to_string(args.GetHeight());

   exec = std::regex_replace(exec, std::regex("\\$geometry"), geometry);
}

//////////////////////////////////////////////////////////////////////////////////////////////////
/// Create Firefox profile to run independent browser window

std::string RWebDisplayHandle::FirefoxCreator::MakeProfile(std::string &exec, bool batch_mode)
{
   std::string rmdir, profile_arg;

   if (exec.find("$profile") == std::string::npos)
      return rmdir;

   const char *ff_profile = gEnv->GetValue("WebGui.FirefoxProfile", "");
   const char *ff_profilepath = gEnv->GetValue("WebGui.FirefoxProfilePath", "");
   Int_t ff_randomprofile = RWebWindowWSHandler::GetBoolEnv("WebGui.FirefoxRandomProfile", 1);
   if (ff_profile && *ff_profile) {
      profile_arg = "-P "s + ff_profile;
   } else if (ff_profilepath && *ff_profilepath) {
      profile_arg = "-profile "s + ff_profilepath;
   } else if (ff_randomprofile > 0) {
      TRandom3 rnd;
      rnd.SetSeed(0);
      std::string profile_dir = gSystem->TempDirectory();
      if ((profile_dir.compare(0, 4, "/tmp") == 0) && IsSnapBrowser())
         profile_dir = gSystem->GetHomeDirectory();

#ifdef _MSC_VER
      char slash = '\\';
#else
      char slash = '/';
#endif
      if (!profile_dir.empty() && (profile_dir[profile_dir.length()-1] != slash))
         profile_dir += slash;
      profile_dir += "root_ff_profile_"s + std::to_string(rnd.Integer(0x100000));

      profile_arg = "-profile "s + profile_dir;

      if (gSystem->mkdir(profile_dir.c_str()) == 0) {
         rmdir = profile_dir;

         std::ofstream user_js(profile_dir + "/user.js", std::ios::trunc);
         // workaround for current Firefox, without such settings it fail to close window and terminate it from batch
         // also disable question about upload of data
         user_js << "user_pref(\"datareporting.policy.dataSubmissionPolicyAcceptedVersion\", 2);" << std::endl;
         user_js << "user_pref(\"datareporting.policy.dataSubmissionPolicyNotifiedTime\", \"1635760572813\");" << std::endl;

         // try to ensure that window closes with last tab
         user_js << "user_pref(\"browser.tabs.closeWindowWithLastTab\", true);" << std::endl;
         user_js << "user_pref(\"dom.allow_scripts_to_close_windows\", true);" << std::endl;
         user_js << "user_pref(\"browser.sessionstore.resume_from_crash\", false);" << std::endl;

         if (batch_mode) {
            // allow to dump messages to std output
            user_js << "user_pref(\"browser.dom.window.dump.enabled\", true);" << std::endl;
         } else {
            // to suppress annoying privacy tab
            user_js << "user_pref(\"datareporting.policy.firstRunURL\", \"\");" << std::endl;
            // to use custom userChrome.css files
            user_js << "user_pref(\"toolkit.legacyUserProfileCustomizations.stylesheets\", true);" << std::endl;
            // do not put tabs in title
            user_js << "user_pref(\"browser.tabs.inTitlebar\", 0);" << std::endl;

#ifdef R__LINUX
            // fix WebGL creation problem on some Linux platforms
            user_js << "user_pref(\"webgl.out-of-process\", false);" << std::endl;
#endif

            std::ofstream times_json(profile_dir + "/times.json", std::ios::trunc);
            times_json << "{" << std::endl;
            times_json << "   \"created\": 1699968480952," << std::endl;
            times_json << "   \"firstUse\": null" << std::endl;
            times_json << "}" << std::endl;
            if (gSystem->mkdir((profile_dir + "/chrome").c_str()) == 0) {
               std::ofstream style(profile_dir + "/chrome/userChrome.css", std::ios::trunc);
               // do not show tabs
               style << "#TabsToolbar { visibility: collapse; }" << std::endl;
               // do not show URL
               style << "#nav-bar, #urlbar-container, #searchbar { visibility: collapse !important; }" << std::endl;
            }
         }

      } else {
         R__LOG_ERROR(WebGUILog()) << "Cannot create Firefox profile directory " << profile_dir;
      }
   }

   exec = std::regex_replace(exec, std::regex("\\$profile"), profile_arg);

   if (exec.find("$cleanup_profile") != std::string::npos) {
      if (rmdir.empty()) rmdir = "__dummy__";
      exec = std::regex_replace(exec, std::regex("\\$cleanup_profile"), rmdir);
      rmdir.clear(); // no need to delete directory - it will be removed by script
   }

   return rmdir;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
/// Check if http server required for display
/// \param args - defines where and how to display web window

bool RWebDisplayHandle::NeedHttpServer(const RWebDisplayArgs &args)
{
   if ((args.GetBrowserKind() == RWebDisplayArgs::kOff) || (args.GetBrowserKind() == RWebDisplayArgs::kCEF) ||
       (args.GetBrowserKind() == RWebDisplayArgs::kQt6) || (args.GetBrowserKind() == RWebDisplayArgs::kLocal))
      return false;

   if (!args.IsHeadless() && (args.GetBrowserKind() == RWebDisplayArgs::kOn)) {

#ifdef WITH_QT6WEB
      auto &qt6 = FindCreator("qt6", "libROOTQt6WebDisplay");
      if (qt6 && qt6->IsActive())
         return false;
#endif
#ifdef WITH_CEFWEB
      auto &cef = FindCreator("cef", "libROOTCefDisplay");
      if (cef && cef->IsActive())
         return false;
#endif
   }

   return true;
}


///////////////////////////////////////////////////////////////////////////////////////////////////
/// Create web display
/// \param args - defines where and how to display web window
/// Returns RWebDisplayHandle, which holds information of running browser application
/// Can be used fully independent from RWebWindow classes just to show any web page

std::unique_ptr<RWebDisplayHandle> RWebDisplayHandle::Display(const RWebDisplayArgs &args)
{
   std::unique_ptr<RWebDisplayHandle> handle;

   if (args.GetBrowserKind() == RWebDisplayArgs::kOff)
      return handle;

   auto try_creator = [&](std::unique_ptr<Creator> &creator) {
      if (!creator || !creator->IsActive())
         return false;
      handle = creator->Display(args);
      return handle ? true : false;
   };

   bool handleAsLocal = (args.GetBrowserKind() == RWebDisplayArgs::kLocal) ||
                        (!args.IsHeadless() && (args.GetBrowserKind() == RWebDisplayArgs::kOn)),
        has_qt6web = false, has_cefweb = false;

#ifdef WITH_QT6WEB
   has_qt6web = true;
#endif

#ifdef WITH_CEFWEB
   has_cefweb = true;
#endif

   if ((handleAsLocal && has_qt6web) || (args.GetBrowserKind() == RWebDisplayArgs::kQt6)) {
      if (try_creator(FindCreator("qt6", "libROOTQt6WebDisplay")))
         return handle;
   }

   if ((handleAsLocal && has_cefweb) || (args.GetBrowserKind() == RWebDisplayArgs::kCEF)) {
      if (try_creator(FindCreator("cef", "libROOTCefDisplay")))
         return handle;
   }

   if (args.IsLocalDisplay()) {
      R__LOG_ERROR(WebGUILog()) << "Neither Qt5/6 nor CEF libraries were found to provide local display";
      return handle;
   }

   bool handleAsNative =
      (args.GetBrowserKind() == RWebDisplayArgs::kNative) || (args.GetBrowserKind() == RWebDisplayArgs::kOn);

   if (handleAsNative || (args.GetBrowserKind() == RWebDisplayArgs::kChrome)) {
      if (try_creator(FindCreator("chrome", "ChromeCreator")))
         return handle;
   }

   if (handleAsNative || (args.GetBrowserKind() == RWebDisplayArgs::kFirefox)) {
      if (try_creator(FindCreator("firefox", "FirefoxCreator")))
         return handle;
   }

#ifdef _MSC_VER
   // Edge browser cannot be run headless without registry change, therefore do not try it by default
   if ((handleAsNative && !args.IsHeadless() && !args.IsBatchMode()) || (args.GetBrowserKind() == RWebDisplayArgs::kEdge)) {
      if (try_creator(FindCreator("edge", "ChromeCreator")))
         return handle;
   }
#endif

   if ((args.GetBrowserKind() == RWebDisplayArgs::kNative) || (args.GetBrowserKind() == RWebDisplayArgs::kChrome) ||
       (args.GetBrowserKind() == RWebDisplayArgs::kFirefox) || (args.GetBrowserKind() == RWebDisplayArgs::kEdge)) {
      // R__LOG_ERROR(WebGUILog()) << "Neither Chrome nor Firefox browser cannot be started to provide display";
      return handle;
   }

   if (args.GetBrowserKind() == RWebDisplayArgs::kSafari) {
      if (try_creator(FindCreator("safari", "SafariCreator")))
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
/// ~~~
///     RWebDisplayArgs args;
///     args.SetUrl(url);
///     args.SetStandalone(false);
///     auto handle = RWebDisplayHandle::Display(args);
/// ~~~

bool RWebDisplayHandle::DisplayUrl(const std::string &url)
{
   RWebDisplayArgs args;
   args.SetUrl(url);
   args.SetStandalone(false);

   auto handle = Display(args);

   return !!handle;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
/// Checks if configured browser can be used for image production

bool RWebDisplayHandle::CheckIfCanProduceImages(RWebDisplayArgs &args)
{
   if ((args.GetBrowserKind() != RWebDisplayArgs::kFirefox) && (args.GetBrowserKind() != RWebDisplayArgs::kEdge) &&
       (args.GetBrowserKind() != RWebDisplayArgs::kChrome) && (args.GetBrowserKind() != RWebDisplayArgs::kCEF) &&
       (args.GetBrowserKind() != RWebDisplayArgs::kQt6)) {
      bool detected = false;

      auto &h1 = FindCreator("chrome", "ChromeCreator");
      if (h1 && h1->IsActive()) {
         args.SetBrowserKind(RWebDisplayArgs::kChrome);
         detected = true;
      }

      if (!detected) {
         auto &h2 = FindCreator("firefox", "FirefoxCreator");
         if (h2 && h2->IsActive()) {
            args.SetBrowserKind(RWebDisplayArgs::kFirefox);
            detected = true;
         }
      }

      return detected;
   }

   if (args.GetBrowserKind() == RWebDisplayArgs::kChrome) {
      auto &h1 = FindCreator("chrome", "ChromeCreator");
      return h1 && h1->IsActive();
   }

   if (args.GetBrowserKind() == RWebDisplayArgs::kFirefox) {
      auto &h2 = FindCreator("firefox", "FirefoxCreator");
      return h2 && h2->IsActive();
   }

#ifdef _MSC_VER
   if (args.GetBrowserKind() == RWebDisplayArgs::kEdge) {
      auto &h3 = FindCreator("edge", "ChromeCreator");
      return h3 && h3->IsActive();
   }
#endif

   return true;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
/// Returns true if image production for specified browser kind is supported
/// If browser not specified - use currently configured browser or try to test existing web browsers

bool RWebDisplayHandle::CanProduceImages(const std::string &browser)
{
   RWebDisplayArgs args(browser);

   return CheckIfCanProduceImages(args);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
/// Detect image format
/// There is special handling of ".screenshot.pdf" and ".screenshot.png" extensions
/// Creation of such files relies on headless browser functionality and fully supported only by Chrome browser

std::string RWebDisplayHandle::GetImageFormat(const std::string &fname)
{
   std::string _fname = fname;
   std::transform(_fname.begin(), _fname.end(), _fname.begin(), ::tolower);
   auto EndsWith = [&_fname](const std::string &suffix) {
      return (_fname.length() > suffix.length()) ? (0 == _fname.compare(_fname.length() - suffix.length(), suffix.length(), suffix)) : false;
   };

   if (EndsWith(".screenshot.pdf"))
      return "s.pdf"s;
   if (EndsWith(".pdf"))
      return "pdf"s;
   if (EndsWith(".json"))
      return "json"s;
   if (EndsWith(".svg"))
      return "svg"s;
   if (EndsWith(".screenshot.png"))
      return "s.png"s;
   if (EndsWith(".png"))
      return "png"s;
   if (EndsWith(".jpg") || EndsWith(".jpeg"))
      return "jpeg"s;
   if (EndsWith(".webp"))
      return "webp"s;

   return ""s;
}


///////////////////////////////////////////////////////////////////////////////////////////////////
/// Produce image file using JSON data as source
/// Invokes JSROOT drawing functionality in headless browser - Google Chrome or Mozilla Firefox

bool RWebDisplayHandle::ProduceImage(const std::string &fname, const std::string &json, int width, int height, const char *batch_file)
{
   return ProduceImages(fname, {json}, {width}, {height}, batch_file);
}


///////////////////////////////////////////////////////////////////////////////////////////////////
/// Produce vector of file names for specified file pattern
/// Depending from supported file forma

std::vector<std::string> RWebDisplayHandle::ProduceImagesNames(const std::string &fname, unsigned nfiles)
{
   auto fmt = GetImageFormat(fname);

   std::vector<std::string> fnames;

   if ((fmt == "s.pdf") || (fmt == "s.png")) {
      fnames.emplace_back(fname);
   } else {
      std::string farg = fname;

      bool has_quialifier = farg.find("%") != std::string::npos;

      if (!has_quialifier && (nfiles > 1) && (fmt != "pdf")) {
         farg.insert(farg.rfind("."), "%d");
         has_quialifier = true;
      }

      for (unsigned n = 0; n < nfiles; n++) {
         if(has_quialifier) {
            auto expand_name = TString::Format(farg.c_str(), (int) n);
            fnames.emplace_back(expand_name.Data());
         } else if (n > 0)
            fnames.emplace_back(""); // empty name is multiPdf
         else
            fnames.emplace_back(fname);
      }
   }

   return fnames;
}


///////////////////////////////////////////////////////////////////////////////////////////////////
/// Produce image file(s) using JSON data as source
/// Invokes JSROOT drawing functionality in headless browser - Google Chrome or Mozilla Firefox

bool RWebDisplayHandle::ProduceImages(const std::string &fname, const std::vector<std::string> &jsons, const std::vector<int> &widths, const std::vector<int> &heights, const char *batch_file)
{
    return ProduceImages(ProduceImagesNames(fname, jsons.size()), jsons, widths, heights, batch_file);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
/// Produce image file(s) using JSON data as source
/// Invokes JSROOT drawing functionality in headless browser - Google Chrome or Mozilla Firefox

bool RWebDisplayHandle::ProduceImages(const std::vector<std::string> &fnames, const std::vector<std::string> &jsons, const std::vector<int> &widths, const std::vector<int> &heights, const char *batch_file)
{
   if (fnames.empty() || jsons.empty())
      return false;

   std::vector<std::string> fmts;
   for (auto& fname : fnames)
      fmts.emplace_back(GetImageFormat(fname));

   bool is_any_image = false;

   for (unsigned n = 0; (n < fmts.size()) && (n < jsons.size()); n++) {
      if (fmts[n] == "json") {
         std::ofstream ofs(fnames[n]);
         ofs << jsons[n];
         fmts[n].clear();
      } else if (!fmts[n].empty())
         is_any_image = true;
   }

   if (!is_any_image)
      return true;

   std::string fdebug;
   if (fnames.size() == 1)
      fdebug  = fnames[0];
   else
      fdebug = TBufferJSON::ToJSON(&fnames, TBufferJSON::kNoSpaces);

   const char *jsrootsys = gSystem->Getenv("JSROOTSYS");
   TString jsrootsysdflt;
   if (!jsrootsys) {
      jsrootsysdflt = TROOT::GetDataDir() + "/js";
      if (gSystem->ExpandPathName(jsrootsysdflt)) {
         R__LOG_ERROR(WebGUILog()) << "Fail to locate JSROOT " << jsrootsysdflt;
         return false;
      }
      jsrootsys = jsrootsysdflt.Data();
   }

   RWebDisplayArgs args; // set default browser kind, only Chrome/Firefox/Edge or CEF/Qt5/Qt6 can be used here
   if (!CheckIfCanProduceImages(args)) {
      R__LOG_ERROR(WebGUILog()) << "Fail to detect supported browsers for image production";
      return false;
   }

   auto isChrome = (args.GetBrowserKind() == RWebDisplayArgs::kChrome),
        isChromeBased = isChrome || (args.GetBrowserKind() == RWebDisplayArgs::kEdge),
        isFirefox = args.GetBrowserKind() == RWebDisplayArgs::kFirefox;

   std::vector<std::string> draw_kinds;
   bool use_browser_draw = false, can_optimize_json = false;
   int use_home_dir = 0;
   TString jsonkind;

   // Some Chrome installation do not allow run html code from files, created in /tmp directory
   // When during session such failures happened, force usage of home directory from the beginning
   static int chrome_tmp_workaround = 0;

   if (isChrome) {
      use_home_dir = chrome_tmp_workaround;
      auto &h1 = FindCreator("chrome", "ChromeCreator");
      if (h1 && h1->IsActive() && h1->IsSnapBrowser() && (use_home_dir == 0))
         use_home_dir = 1;
   }

   if (fmts[0] == "s.png") {
      if (!isChromeBased && !isFirefox) {
         R__LOG_ERROR(WebGUILog()) << "Direct png image creation supported only by Chrome and Firefox browsers";
         return false;
      }
      use_browser_draw = true;
      jsonkind = "1111"; // special mark in canv_batch.htm
   } else if (fmts[0] == "s.pdf") {
      if (!isChromeBased) {
         R__LOG_ERROR(WebGUILog()) << "Direct creation of PDF files supported only by Chrome-based browser";
         return false;
      }
      use_browser_draw = true;
      jsonkind = "2222"; // special mark in canv_batch.htm
   } else {
      draw_kinds = fmts;
      jsonkind = TBufferJSON::ToJSON(&draw_kinds, TBufferJSON::kNoSpaces);
      can_optimize_json = true;
   }

   if (!batch_file || !*batch_file)
      batch_file = "/js/files/canv_batch.htm";

   TString origin = TROOT::GetDataDir() + batch_file;
   if (gSystem->ExpandPathName(origin)) {
      R__LOG_ERROR(WebGUILog()) << "Fail to find " << origin;
      return false;
   }

   auto filecont = THttpServer::ReadFileContent(origin.Data());
   if (filecont.empty()) {
      R__LOG_ERROR(WebGUILog()) << "Fail to read content of " << origin;
      return false;
   }

   int max_width = 0, max_height = 0, page_margin = 10;
   for (auto &w : widths)
      if (w > max_width)
         max_width = w;
   for (auto &h : heights)
      if (h > max_height)
         max_height = h;

   auto jsonw = TBufferJSON::ToJSON(&widths, TBufferJSON::kNoSpaces);
   auto jsonh = TBufferJSON::ToJSON(&heights, TBufferJSON::kNoSpaces);

   std::string mains, prev;
   for (auto &json : jsons) {
      mains.append(mains.empty() ? "[" : ", ");
      if (can_optimize_json && (json == prev)) {
         mains.append("'same'");
      } else {
         mains.append(json);
         prev = json;
      }
   }
   mains.append("]");

   if (strstr(jsrootsys, "http://") || strstr(jsrootsys, "https://") || strstr(jsrootsys, "file://"))
      filecont = std::regex_replace(filecont, std::regex("\\$jsrootsys"), jsrootsys);
   else {
      static std::string jsroot_include = "<script id=\"jsroot\" src=\"$jsrootsys/build/jsroot.js\"></script>";
      auto p = filecont.find(jsroot_include);
      if (p != std::string::npos) {
         auto jsroot_build = THttpServer::ReadFileContent(std::string(jsrootsys) + "/build/jsroot.js");
         if (!jsroot_build.empty()) {
            // insert actual jsroot file location
            jsroot_build = std::regex_replace(jsroot_build, std::regex("'\\$jsrootsys'"), std::string("'file://") + jsrootsys + "/'");
            filecont.erase(p, jsroot_include.length());
            filecont.insert(p, "<script id=\"jsroot\">" + jsroot_build + "</script>");
         }
      }

      filecont = std::regex_replace(filecont, std::regex("\\$jsrootsys"), "file://"s + jsrootsys);
   }

   filecont = std::regex_replace(filecont, std::regex("\\$page_margin"), std::to_string(page_margin) + "px");
   filecont = std::regex_replace(filecont, std::regex("\\$page_width"), std::to_string(max_width + 2*page_margin) + "px");
   filecont = std::regex_replace(filecont, std::regex("\\$page_height"), std::to_string(max_height + 2*page_margin) + "px");

   filecont = std::regex_replace(filecont, std::regex("\\$draw_kind"), jsonkind.Data());
   filecont = std::regex_replace(filecont, std::regex("\\$draw_widths"), jsonw.Data());
   filecont = std::regex_replace(filecont, std::regex("\\$draw_heights"), jsonh.Data());
   filecont = std::regex_replace(filecont, std::regex("\\$draw_objects"), mains);

   TString dump_name, html_name;

   if (!use_browser_draw && (isChromeBased || isFirefox)) {
      dump_name = "canvasdump";
      FILE *df = BrowserCreator::TemporaryFile(dump_name, use_home_dir);
      if (!df) {
         R__LOG_ERROR(WebGUILog()) << "Fail to create temporary file for dump-dom";
         return false;
      }
      fputs("placeholder", df);
      fclose(df);
   }

try_again:

   if ((args.GetBrowserKind() == RWebDisplayArgs::kCEF) || (args.GetBrowserKind() == RWebDisplayArgs::kQt6)) {
      args.SetUrl(""s);
      args.SetPageContent(filecont);

      html_name.Clear();

      R__LOG_DEBUG(0, WebGUILog()) << "Using file content_len " << filecont.length() << " to produce batch images ";

   } else {
      html_name = "canvasbody";
      FILE *hf = BrowserCreator::TemporaryFile(html_name, use_home_dir, ".html");
      if (!hf) {
         R__LOG_ERROR(WebGUILog()) << "Fail to create temporary file for batch job";
         return false;
      }
      fputs(filecont.c_str(), hf);
      fclose(hf);

      args.SetUrl("file://"s + gSystem->UnixPathName(html_name.Data()));
      args.SetPageContent(""s);

      R__LOG_DEBUG(0, WebGUILog()) << "Using " << html_name << " content_len " << filecont.length() << " to produce batch images " << fdebug;
   }

   TString wait_file_name, tgtfilename;

   args.SetStandalone(true);
   args.SetHeadless(true);
   args.SetBatchMode(true);
   args.SetSize(widths[0], heights[0]);

   if (use_browser_draw) {

      tgtfilename = fnames[0].c_str();
      if (!gSystem->IsAbsoluteFileName(tgtfilename.Data()))
         gSystem->PrependPathName(gSystem->WorkingDirectory(), tgtfilename);

      wait_file_name = tgtfilename;

      if (fmts[0] == "s.pdf")
         args.SetExtraArgs("--print-to-pdf-no-header --print-to-pdf="s + gSystem->UnixPathName(tgtfilename.Data()));
      else if (isFirefox) {
         args.SetExtraArgs("--screenshot"); // firefox does not let specify output image file
         wait_file_name = "screenshot.png";
      } else
         args.SetExtraArgs("--screenshot="s + gSystem->UnixPathName(tgtfilename.Data()));

      // remove target image file - we use it as detection when chrome is ready
      gSystem->Unlink(tgtfilename.Data());

   } else if (isFirefox) {
      // firefox will use window.dump to output produced result
      args.SetRedirectOutput(dump_name.Data());
      gSystem->Unlink(dump_name.Data());
   } else if (isChromeBased) {
      // chrome should have --dump-dom args configures
      args.SetRedirectOutput(dump_name.Data());
      gSystem->Unlink(dump_name.Data());
   }

   auto handle = RWebDisplayHandle::Display(args);

   if (!handle) {
      R__LOG_DEBUG(0, WebGUILog()) << "Cannot start " << args.GetBrowserName() << " to produce image " << fdebug;
      return false;
   }

   // delete temporary HTML file
   if (html_name.Length() > 0) {
      if (gEnv->GetValue("WebGui.PreserveBatchFiles", -1) > 0)
         ::Info("ProduceImages", "Preserve batch file %s", html_name.Data());
      else
         gSystem->Unlink(html_name.Data());
   }

   if (!wait_file_name.IsNull() && gSystem->AccessPathName(wait_file_name.Data())) {
      R__LOG_ERROR(WebGUILog()) << "Fail to produce image " << fdebug;
      return false;
   }

   if (use_browser_draw) {
      if (fmts[0] == "s.pdf")
         ::Info("ProduceImages", "PDF file %s with %d pages has been created", fnames[0].c_str(), (int) jsons.size());
      else {
         if (isFirefox)
            gSystem->Rename("screenshot.png", fnames[0].c_str());
         ::Info("ProduceImages", "PNG file %s with %d pages has been created", fnames[0].c_str(), (int) jsons.size());
      }
   } else {
      auto dumpcont = handle->GetContent();

      if ((dumpcont.length() > 20) && (dumpcont.length() < 60) && (use_home_dir < 2) && isChrome) {
         // chrome creates dummy html file with mostly no content
         // problem running chrome from /tmp directory, lets try work from home directory
         R__LOG_INFO(WebGUILog()) << "Use home directory for running chrome in batch, set TMPDIR for preferable temp directory";
         chrome_tmp_workaround = use_home_dir = 2;
         goto try_again;
      }

      if (dumpcont.length() < 100) {
         R__LOG_ERROR(WebGUILog()) << "Fail to dump HTML code into " << (dump_name.IsNull() ? "CEF" : dump_name.Data());
         return false;
      }

      std::string::size_type p = 0;

      for (unsigned n = 0; n < fmts.size(); n++) {
         if (fmts[n].empty())
            continue;
         if (fmts[n] == "svg") {
            auto p1 = dumpcont.find("<div><svg", p);
            auto p2 = dumpcont.find("</svg></div>", p1 + 8);
            p = p2 + 12;
            std::ofstream ofs(fnames[n]);
            if ((p1 != std::string::npos) && (p2 != std::string::npos) && (p1 < p2)) {
               if (p2 - p1 > 10) {
                  ofs << dumpcont.substr(p1 + 5, p2 - p1 + 1);
                  ::Info("ProduceImages", "Image file %s size %d bytes has been created", fnames[n].c_str(), (int) (p2 - p1 + 1));
               } else {
                  ::Error("ProduceImages", "Failure producing %s", fnames[n].c_str());
               }
            }
         } else {
            auto p0 = dumpcont.find("<img src=\"", p);
            auto p1 = dumpcont.find(";base64,", p0 + 8);
            auto p2 = dumpcont.find("\">", p1 + 8);
            p = p2 + 2;

            if ((p0 != std::string::npos) && (p1 != std::string::npos) && (p2 != std::string::npos) && (p1 < p2)) {
               auto base64 = dumpcont.substr(p1+8, p2-p1-8);
               if ((base64 == "failure") || (base64.length() < 10)) {
                  ::Error("ProduceImages", "Failure producing %s", fnames[n].c_str());
               } else {
                  auto binary = TBase64::Decode(base64.c_str());
                  std::ofstream ofs(fnames[n], std::ios::binary);
                  ofs.write(binary.Data(), binary.Length());
                  ::Info("ProduceImages", "Image file %s size %d bytes has been created", fnames[n].c_str(), (int) binary.Length());
               }
            } else {
               ::Error("ProduceImages", "Failure producing %s", fnames[n].c_str());
               return false;
            }
         }
      }
   }

   R__LOG_DEBUG(0, WebGUILog()) << "Create " << (fnames.size() > 1 ? "files " : "file ") << fdebug;

   return true;
}

