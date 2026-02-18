/// \file rootrm.cxx
///
/// Command line tool to remove objects from ROOT files
///
/// \author Giacomo Parolini <giacomo.parolini@cern.ch>
/// \date 2026-02-18
#include <ROOT/RLogger.hxx>

#include "logging.hxx"
#include "optparse.hxx"
#include "RootObjTree.hxx"
#include "RootObjTree.cxx"

#include <TClass.h>
#include <TError.h>
#include <TFile.h>
#include <TROOT.h>
#include <TSystem.h>

#include <algorithm>
#include <iostream>
#include <memory>
#include <string_view>
#include <vector>

using namespace ROOT::CmdLine;

static const char *const kShortHelp = "usage: rootrm [-h] [-i|--interactive] [-r|--recursive] "
                                      "[-v|--verbose] FILE[:path/to/obj] [...]\n";
static const char *const kLongHelp = R"(
Remove objects from ROOT files

positional arguments:
  FILE:path             File(s) and path of objects to remove

options:
  -h, --help            show this help message and exit
  -i, --interactive     prompt before deleting each object
  -n, --dry-run         show what would be removed, but don't actually remove anything
  -r, --recursive       recurse inside directories
  -v, --verbose         be verbose

examples:
- rootrm example.root:hist
  Remove the object 'hist' from the ROOT file 'example.root'

- rootrm example.root:dir/hist
  Remove the object 'hist' from the directory 'dir' inside the ROOT file 'example.root'

- rootrm example.root
  Remove the ROOT file 'example.root'

- rootrm -i example.root:hist
  Display a confirmation request before deleting: 'remove 'hist' from 'example.root' ? (y/n) :'
)";

struct RootRmArgs {
   enum class EPrintUsage {
      kNo,
      kShort,
      kLong
   };
   EPrintUsage fPrintHelp = EPrintUsage::kNo;
   bool fInteractive = false;
   bool fRecursive = false;
   bool fDryRun = false;
   std::vector<std::string> fSources;
};

static RootRmArgs ParseArgs(const char **args, int nArgs)
{
   using ROOT::RCmdLineOpts;

   RootRmArgs outArgs;

   RCmdLineOpts opts;
   opts.AddFlag({"-h", "--help"});
   opts.AddFlag({"-i", "--interactive"});
   opts.AddFlag({"-n", "--dry-run"});
   opts.AddFlag({"-r", "--recursive"});
   opts.AddFlag({"-v", "--verbose"}, RCmdLineOpts::EFlagType::kSwitch, "", RCmdLineOpts::kFlagAllowMultiple);

   opts.Parse(args, nArgs);

   for (const auto &err : opts.GetErrors()) {
      std::cerr << err << "\n";
   }
   if (!opts.GetErrors().empty()) {
      outArgs.fPrintHelp = RootRmArgs::EPrintUsage::kShort;
      return outArgs;
   }

   if (opts.GetSwitch("help")) {
      outArgs.fPrintHelp = RootRmArgs::EPrintUsage::kLong;
      return outArgs;
   }

   outArgs.fInteractive = opts.GetSwitch("interactive");
   outArgs.fRecursive = opts.GetSwitch("recursive");
   outArgs.fDryRun = opts.GetSwitch("dry-run");

   // fDryRun implies (at least) verbosity = 2.
   int verbosity = std::max(opts.GetSwitch("v") + 1, outArgs.fDryRun + 1);
   SetLogVerbosity(verbosity);

   outArgs.fSources = opts.GetArgs();
   if (outArgs.fSources.size() < 1)
      outArgs.fPrintHelp = RootRmArgs::EPrintUsage::kShort;

   return outArgs;
}

static bool PromptForRemoval(std::string_view fileName, std::string_view objName)
{
   if (objName.empty())
      std::cout << "remove '" << fileName << "' ? (y/n) ";
   else
      std::cout << "remove '" << objName << "' from '" << fileName << "' ? (y/n) ";
   std::string answer;
   std::cin >> answer;
   return answer == 'y' || answer == 'Y';
}

static void RemoveNode(const RootSource &src, NodeIdx_t nodeIdx, const RootRmArgs &args)
{
   // nodeIdx must be in range because it always comes from a RootObjTree.
   assert(nodeIdx < src.fObjectTree.fNodes.size());
   const RootObjNode &node = src.fObjectTree.fNodes[nodeIdx];
   const bool nodeIsRootFile = !node.fKey;
   std::string_view objName = nodeIsRootFile ? std::string_view{} : node.fName;

   if (node.fDir && !args.fRecursive) {
      if (nodeIsRootFile)
         Err() << "cannot remove '" << node.fName << "': is a ROOT file. Use -r to remove it.\n";
      else
         Err() << "cannot remove '" << node.fName << "': is a directory. Use -r to remove it.\n";
      return;
   }

   const bool doRemove = !args.fInteractive || PromptForRemoval(src.fFileName, objName);
   if (!doRemove)
      return;

   if (!args.fDryRun) {
      if (nodeIsRootFile) {
         // delete the entire file.
         src.fObjectTree.fFile->Close();
         gSystem->Unlink(src.fFileName.c_str());
      } else if (node.fDir) {
         // delete the entire directory
         node.fDir->GetMotherDir()->Delete((node.fName + ";*").c_str());
      } else {
         // delete a single object
         Option_t *opt = GetLogVerbosity() > 2 ? "v" : "";
         node.fKey->Delete(opt);
      }
   }

   Info(2) << "removed '" << NodeFullPath(src.fObjectTree, nodeIdx, ENodeFullPathOpt::kIncludeFilename) << "'\n";
}

int main(int argc, char **argv)
{
   InitLog("rootrm");

   // Parse arguments
   auto args = ParseArgs(const_cast<const char **>(argv) + 1, argc - 1);
   if (args.fPrintHelp != RootRmArgs::EPrintUsage::kNo) {
      std::cerr << kShortHelp;
      if (args.fPrintHelp == RootRmArgs::EPrintUsage::kLong) {
         std::cerr << kLongHelp;
         return 0;
      }
      return 1;
   }

   // Validate and split all input sources into filename + pattern
   std::vector<std::pair<std::string_view, std::string_view>> sourcesFileAndPattern;
   sourcesFileAndPattern.reserve(args.fSources.size());
   for (const auto &src : args.fSources) {
      auto res = SplitIntoFileNameAndPattern(src);
      if (!res) {
         Err() << res.GetError()->GetReport() << "\n";
         return 1;
      }
      auto fNameAndPattern = res.Unwrap();
      sourcesFileAndPattern.push_back(fNameAndPattern);
   }

   const std::uint32_t flags = kOpenFilesAsWritable | (args.fRecursive * EGetMatchingPathsFlags::kRecursive);
   bool errors = false;
   for (const auto &[srcFname, srcPattern] : sourcesFileAndPattern) {
      auto src = ROOT::CmdLine::GetMatchingPathsInFile(srcFname, srcPattern, flags);
      if (!src.fErrors.empty()) {
         for (const auto &err : src.fErrors)
            Err() << err << "\n";

         errors = true;
         break;
      }

      // We should never register files to the global list for performance reasons.
      assert(!gROOT->GetListOfFiles()->Contains(src.fObjectTree.fFile.get()));

      // Iterate all objects we need to remove
      for (auto nodeIdx : src.fObjectTree.fLeafList) {
         RemoveNode(src, nodeIdx, args);
      }
      for (auto nodeIdx : src.fObjectTree.fDirList) {
         RemoveNode(src, nodeIdx, args);
      }
   }

   return errors;
}
