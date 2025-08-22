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
#include <TROOT.h>
#include <TSystem.h>

#include <chrono>
#include <cstring>
#include <iostream>
#include <memory>
#include <thread>

static const char *const kShortHelp = "usage: rootbrowse [-w WEB|-wf] <file.root>\n";
static const char *const kLongHelp = R"(
Open a ROOT file in a TBrowser

positional arguments:
  FILE           Input file

options:
  -h, --help     show this help message and exit
  -w, --web WEB  Configure webdisplay. For all possible values, see:
                 https://root.cern/doc/v636/classTROOT.html#a1749472696545b76a6b8e79769e7e773
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
   const char *fWeb = "on";
   const char *fFileName = nullptr;
};

static RootBrowseArgs ParseArgs(const char **args, int nArgs)
{
   RootBrowseArgs outArgs;
   bool forcePositional = false;

   for (int i = 0; i < nArgs; ++i) {
      const char *arg = args[i];
      bool isFlag = !forcePositional && arg[0] == '-';
      if (isFlag) {
         if (strcmp(arg, "-w") == 0 || strcmp(arg, "--web") == 0) {
            if (i < nArgs - 1 && args[i + 1][0] != '-') {
               ++i;
               outArgs.fWeb = args[i];
            }
         } else if (strcmp(arg, "-wf") == 0 || strcmp(arg, "--webOff") == 0) {
            outArgs.fWeb = "off";
         } else if (strcmp(arg, "-h") == 0 || strcmp(arg, "--help") == 0) {
            outArgs.fPrintHelp = RootBrowseArgs::EPrintUsage::kLong;
            break;
         } else if (strcmp(arg, "--") == 0) {
            forcePositional = true;
         } else {
            outArgs.fPrintHelp = RootBrowseArgs::EPrintUsage::kShort;
            break;
         }
      } else if (outArgs.fFileName) {
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

   gROOT->SetWebDisplay(args.fWeb);

   std::unique_ptr<TFile> file;
   if (args.fFileName) {
      gErrorIgnoreLevel = kError;
      file = std::unique_ptr<TFile>(TFile::Open(args.fFileName, "READ"));
      if (!file || file->IsZombie()) {
         R__LOG_WARNING(RootBrowseLog()) << "File " << args.fFileName << " does not exist or is unreadable.";
      }
      gErrorIgnoreLevel = kUnset;
   }

   // NOTE: we need to instantiate TApplication ourselves, otherwise TBrowser
   // will create a batch application that cannot show graphics.
   TApplication app("rootbrowse", nullptr, nullptr);

   auto browser = std::make_unique<TBrowser>();
   std::cout << "Press ctrl+c to exit.\n";
   while (!gROOT->IsInterrupted() && !gSystem->ProcessEvents()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
   }

   return 0;
}
