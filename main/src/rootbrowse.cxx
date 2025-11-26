// \file rootbrowse.cxx
///
/// Command line tool to open a ROOT file on a TBrowser
///
/// \author Giacomo Parolini <giacomo.parolini@cern.ch>
/// \date 2025-08-21
#include <ROOT/RLogger.hxx>

#include "logging.hxx"
#include "optparse.hxx"

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

struct RootBrowseArgs {
   enum class EPrintUsage {
      kNo,
      kShort,
      kLong
   };
   EPrintUsage fPrintHelp = EPrintUsage::kNo;
   std::string fWeb;
   std::string fFileName;
};

static RootBrowseArgs ParseArgs(const char **args, int nArgs)
{
   RootBrowseArgs outArgs;
   ROOT::RCmdLineOpts opts;
   opts.AddFlag({"-h", "--help"});
   opts.AddFlag({"-w", "--web"}, ROOT::RCmdLineOpts::EFlagType::kWithArg);
   opts.AddFlag({"-wf", "--webOff"});

   opts.Parse(args, nArgs);

   if (opts.ReportErrors()) {
      outArgs.fPrintHelp = RootBrowseArgs::EPrintUsage::kShort;
      return outArgs;
   }

   if (opts.GetSwitch("help")) {
      outArgs.fPrintHelp = RootBrowseArgs::EPrintUsage::kLong;
      return outArgs;
   }

   if (auto web = opts.GetFlagValue("web"); !web.empty())
      outArgs.fWeb = web;

   if (opts.GetSwitch("webOff"))
      outArgs.fWeb = "off";

   if (opts.GetArgs().empty())
      outArgs.fPrintHelp = RootBrowseArgs::EPrintUsage::kShort;
   else
      outArgs.fFileName = opts.GetArgs()[0];

   return outArgs;
}

int main(int argc, char **argv)
{
   InitLog("rootbrowse");
   
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
         Err() << "File " << args.fFileName << " does not exist or is unreadable.\n";
         return 1;
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
