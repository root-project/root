// \file rootbrowse.cxx
///
/// Command line tool to open a ROOT file on a TBrowser
///
/// \author Giacomo Parolini <giacomo.parolini@cern.ch>
/// \date 2025-08-21
#include <ROOT/RLogger.hxx>

#include <TApplication.h>
#include <TBrowser.h>
#include <TError.h>
#include <TFile.h>
#include <TGFrame.h>
#include <TROOT.h>
#include <TSystem.h>

#include <chrono>
#include <cstring>
#include <iostream>
#include <memory>
#include <thread>
#include <string_view>

static const char *const kShortHelp = "usage: rootbrowse [-w WEB|-wf] <file.root>\n";
static const char *const kLongHelp = R"(
Open a ROOT file in a TBrowser

positional arguments:
  FILE           Input file

options:
  -h, --help     show this help message and exit
  -w, --web WEB  Configure webdisplay. For all possible values, see TROOT::SetWebDisplay():
                 https://root.cern/doc/latest-stable/classTROOT.html#a1749472696545b76a6b8e79769e7e773
  -wf, --webOff  Invoke the classic TBrowser (not the web version)

Examples:
- rootbrowse
  Open a TBrowser

- rootbrowse file.root
  Open the ROOT file 'file.root' in a TBrowser
)";

static ROOT::RLogChannel &RootBrowseLog()
{
   static ROOT::RLogChannel channel("RootBrowse");
   return channel;
}

struct RootBrowseArgs {
   enum class EPrintUsage {
      kNo,
      kShort,
      kLong
   };
   EPrintUsage fPrintHelp = EPrintUsage::kNo;
   std::string_view fWeb;
   std::string_view fFileName;
};

static RootBrowseArgs ParseArgs(const char **args, int nArgs)
{
   RootBrowseArgs outArgs;
   bool forcePositional = false;

   for (int i = 0; i < nArgs; ++i) {
      const char *arg = args[i];

      if (strcmp(arg, "--") == 0) {
         forcePositional = true;
         continue;
      }

      bool isFlag = !forcePositional && arg[0] == '-';
      if (isFlag) {
         ++arg;
         // Parse long or short flag and its argument into `argStr` / `nxtArgStr`.
         std::string_view argStr, nxtArgStr;
         if (arg[0] == '-') {
            ++arg;
            // long flag: may be either of the form `--web off` or `--web=off`
            const char *eq = strchr(arg, '=');
            if (eq) {
               argStr = std::string_view(arg, eq - arg);
               nxtArgStr = std::string_view(eq + 1);
            } else {
               argStr = std::string_view(arg);
               if (i < nArgs - 1 && args[i + 1][0] != '-') {
                  nxtArgStr = args[i + 1];
                  ++i;
               }
            }
         } else {
            // short flag (note that it might be more than 1 character long, like `-wf`)
            argStr = std::string_view(arg);
            if (i < nArgs - 1 && args[i + 1][0] != '-') {
               nxtArgStr = args[i + 1];
               ++i;
            }
         }

         if (argStr == "w" || argStr == "web") {
            outArgs.fWeb = nxtArgStr.empty() ? "on" : nxtArgStr;
         } else if (argStr == "h" || argStr == "help") {
            outArgs.fPrintHelp = RootBrowseArgs::EPrintUsage::kLong;
            break;
         } else if (argStr == "wf" || argStr == "--webOff") {
            outArgs.fWeb = "off";
         }

      } else if (!outArgs.fFileName.empty()) {
         outArgs.fPrintHelp = RootBrowseArgs::EPrintUsage::kShort;
         break;
      } else {
         outArgs.fFileName = arg;
      }
   }

   return outArgs;
}

int main(int argc, char **argv)
{
   auto args = ParseArgs(const_cast<const char **>(argv) + 1, argc - 1);
   if (args.fPrintHelp != RootBrowseArgs::EPrintUsage::kNo) {
      std::cerr << kShortHelp;
      if (args.fPrintHelp == RootBrowseArgs::EPrintUsage::kLong) {
         std::cerr << kLongHelp;
         return 0;
      }
      return 1;
   }

   // NOTE: we need to instantiate TApplication ourselves, otherwise TBrowser
   // will create a batch application that cannot show graphics.
   TApplication app("rootbrowse", nullptr, nullptr);

   if (!args.fWeb.empty())
      gROOT->SetWebDisplay(std::string(args.fWeb).c_str());

   std::unique_ptr<TFile> file;
   if (!args.fFileName.empty()) {
      gErrorIgnoreLevel = kError;
      file = std::unique_ptr<TFile>(TFile::Open(std::string(args.fFileName).c_str(), "READ"));
      if (!file || file->IsZombie()) {
         R__LOG_WARNING(RootBrowseLog()) << "File " << args.fFileName << " does not exist or is unreadable.";
      }
      gErrorIgnoreLevel = kUnset;
   }

   auto browser = std::make_unique<TBrowser>();

   if (gROOT->IsBatch())
      return 1;

   // For classic graphics: ensure rootbrowse quits when the window is closed
   if (auto imp = browser->GetBrowserImp()) {
      if (auto mainframe = imp->GetMainFrame()) {        
         mainframe->Connect("CloseWindow()", "TApplication", &app, "Terminate()");
      }
   }

   std::cout << "Press ctrl+c to exit.\n";
   while (!gROOT->IsInterrupted() && !gSystem->ProcessEvents()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
   }

   return 0;
}
