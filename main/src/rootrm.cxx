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

static const char *const kDescription = R"(Remove objects from ROOT files

examples:
- rootrm example.root:hist
  Remove the object 'hist' from the ROOT file 'example.root'

- rootrm example.root:dir/hist
  Remove the object 'hist' from the directory 'dir' inside the ROOT file 'example.root'

- rootrm example.root
  Remove the ROOT file 'example.root'

- rootrm -i example.root:hist
  Display a confirmation request before deleting: 'remove 'hist' from 'example.root' ? (y/n) :'

positional arguments:
  FILE:path             File(s) and path of objects to remove)";

struct RootRmArgs {
   std::string fHelp;
   bool fInteractive = false;
   bool fRecursive = false;
   bool fDryRun = false;
   bool fForce = false;
   std::vector<std::string> fSources;
};

static RootRmArgs ParseArgs(const char **args, int nArgs)
{
   using FT = ROOT::RCmdLineOpts::EFlagType;
   using ROOT::RCmdLineOpts;

   RootRmArgs outArgs;

   RCmdLineOpts opts;
   opts.AddFlag({"-f", "--force"}, FT::kSwitch, "ignore nonexistent files and arguments, never prompt");
   opts.AddFlag({"-h", "--help"}, FT::kSwitch, "show this help message and exit");
   opts.AddFlag({"-i", "--interactive"}, FT::kSwitch, "prompt before deleting each object");
   opts.AddFlag({"-n", "--dry-run"}, FT::kSwitch,
                "show what would be removed, but don't actually remove anything. Implies `-v`.");
   opts.AddFlag({"-r", "--recursive"}, FT::kSwitch, "recurse inside directories");
   opts.AddFlag({"-v", "--verbose"}, RCmdLineOpts::EFlagType::kSwitch,
                "be verbose (can be repeated for increased verbosity)", RCmdLineOpts::kFlagAllowMultiple);

   opts.Parse(args, nArgs);

   RCmdLineOpts::RMakeHelpOptions helpOpts;
   helpOpts.fWrapDescription = false;
   helpOpts.fDescription = kDescription;
   helpOpts.fPositionalArgs = "FILE[:path/to/obj] [...]";

   for (const auto &err : opts.GetErrors()) {
      std::cerr << err << "\n";
   }
   if (!opts.GetErrors().empty()) {
      outArgs.fHelp = opts.MakeShortHelp(Detail::gLogName, helpOpts);
      return outArgs;
   }

   if (opts.GetSwitch("help")) {
      outArgs.fHelp = opts.MakeLongHelp(Detail::gLogName, helpOpts);
      return outArgs;
   }

   outArgs.fForce = opts.GetSwitch("force");
   outArgs.fInteractive = opts.GetSwitch("interactive");
   outArgs.fRecursive = opts.GetSwitch("recursive");
   outArgs.fDryRun = opts.GetSwitch("dry-run");

   // fDryRun implies (at least) verbosity = 2.
   int verbosity = std::max(opts.GetSwitch("v") + 1, outArgs.fDryRun + 1);
   SetLogVerbosity(verbosity);

   outArgs.fSources = opts.GetArgs();
   if (outArgs.fSources.size() < 1)
      outArgs.fHelp = opts.MakeShortHelp(Detail::gLogName, helpOpts);

   if (outArgs.fForce && outArgs.fInteractive) {
      Err() << "-i and -f flags are mutually exclusive.\n";
      outArgs.fHelp = opts.MakeShortHelp(Detail::gLogName, helpOpts);
   }

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

// Returns true if the entire file was deleted.
static bool RemoveNode(RootSource &src, NodeIdx_t nodeIdx, const RootRmArgs &args)
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
      return false;
   }

   const bool doRemove = !args.fInteractive || PromptForRemoval(src.fFileName, objName);
   if (!doRemove)
      return false;

   if (!args.fDryRun) {
      if (nodeIsRootFile) {
         // delete the entire file.
         src.fObjectTree.fFile.reset();
         const auto res = gSystem->Unlink(src.fFileName.c_str());
         if (res < 0 && !args.fForce) {
            Err() << "failed to remove '" << src.fFileName << "': " << gSystem->GetErrorStr() << "\n";
         }
         return res == 0;
      } else {
         if (node.fDir) {
            // Reset the kIsDirectoryFile bit to allow deleting the directory key (normally not allowed).
            // Two considerations:
            // 1. we need to use BIT(14) because TKey::kIsDirectoryFile is private;
            // 2. this is safe to do because we're not gonna read the directory anymore until the end of the program,
            //    so we can circumvent the safety measure imposed by TKey for directories.
            node.fKey->ResetBit(BIT(14));
         }
         Option_t *opt = GetLogVerbosity() > 2 ? "v" : "";
         node.fKey->Delete(opt);
      }
   }

   Info(2) << "removed '" << NodeFullPath(src.fObjectTree, nodeIdx, ENodeFullPathOpt::kIncludeFilename) << "'\n";
   return false;
}

int main(int argc, char **argv)
{
   InitLog("rootrm");

   // Parse arguments
   auto args = ParseArgs(const_cast<const char **>(argv) + 1, argc - 1);
   if (!args.fHelp.empty()) {
      std::cerr << args.fHelp << "\n";
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
   std::vector<std::string_view> filesDeleted;
   bool errors = false;
   for (const auto &[srcFname, srcPattern] : sourcesFileAndPattern) {
      // If we already deleted this file, drop this source referring to it.
      if (std::find(filesDeleted.begin(), filesDeleted.end(), srcFname) != filesDeleted.end()) {
         continue;
      }

      auto src = ROOT::CmdLine::GetMatchingPathsInFile(srcFname, srcPattern, flags);
      if (!src.fErrors.empty()) {
         // ignore errors with -f flag
         if (args.fForce)
            continue;

         for (const auto &err : src.fErrors)
            Err() << err << "\n";

         errors = true;
         break;
      }

      // We should never register files to the global list for performance reasons.
      assert(!gROOT->GetListOfFiles()->Contains(src.fObjectTree.fFile.get()));

      // Iterate all objects we need to remove
      for (auto nodeIdx : src.fObjectTree.fLeafList) {
         if (RemoveNode(src, nodeIdx, args)) {
            filesDeleted.push_back(srcFname);
            break;
         }
      }
      for (auto nodeIdx : src.fObjectTree.fDirList) {
         if (RemoveNode(src, nodeIdx, args)) {
            filesDeleted.push_back(srcFname);
            break;
         }
      }
   }

   return errors;
}
