/// \file rootcp.cxx
///
/// Command line tool to copy objects from ROOT files to others
///
/// \author Giacomo Parolini <giacomo.parolini@cern.ch>
/// \date 2025-10-09
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
#include <TTree.h>

#include <algorithm>
#include <iostream>
#include <memory>
#include <string_view>
#include <vector>

using namespace ROOT::CmdLine;

static const char *const kShortHelp = "usage: rootcp [-h] [-c COMPRESS] [--recreate] [-r|--recursive] [--replace] "
                                      "[-v|--verbose] SOURCE [SOURCE ...] DEST\n";
static const char *const kLongHelp = R"(
Copy objects from ROOT files into another

positional arguments:
  SOURCE                Source file(s)
  DEST                  Destination file

options:
  -h, --help            show this help message and exit
  -c, --compress COMPRESS
                        change the compression settings of the destination file (if not already
                        existing).
  --recreate            recreate the destination file.
  -r, --recursive       recurse inside directories
  --replace             replace object if already existing
  -v, --verbose         be verbose

Note: If an object has been written to a file multiple times, rootcp will copy only the latest version of that object.

Source and destination files accept the syntax: `protocol://path/to/file.root:path/to/object*` to select specific
subobjects or directories in the file.

Examples:
- rootcp source.root dest.root
  Copy the latest version of each object in 'source.root' to 'dest.root'.

- rootcp source.root:hist* dest.root
  Copy all histograms whose names start with 'hist' from 'source.root' to 'dest.root'.

- rootcp source1.root:hist1 source2.root:hist2 dest.root
  Copy histograms 'hist1' from 'source1.root' and 'hist2' from 'source2.root' to 'dest.root'.

- rootcp --recreate source.root:hist dest.root
  Recreate 'dest.root' and copy the histogram named 'hist' from 'source.root' into it.

- rootcp -c 101 source.root:hist dest.root
  Change compression, if not existing, of 'dest.root' to ZLIB algorithm with compression level 1 and copy the histogram named 'hist' from 'source.root' into it.
  Meaning of the '-c' argument is given by 'compress = 100 * algorithm + level'.
  Other examples of usage:
    * -c 509 : ZSTD with compression level 9
    * -c 404 : LZ4 with compression level 4
    * -c 207 : LZMA with compression level 7
  For more information see https://root.cern.ch/doc/latest-stable/classTFile.html#ad0377adf2f3d88da1a1f77256a140d60
  and https://root.cern.ch/doc/latest-stable/structROOT_1_1RCompressionSetting.html
)";

struct RootCpArgs {
   enum class EPrintUsage {
      kNo,
      kShort,
      kLong
   };
   EPrintUsage fPrintHelp = EPrintUsage::kNo;
   std::optional<int> fCompression = std::nullopt;
   bool fRecreate = false;
   bool fReplace = false;
   bool fRecursive = false;
   std::vector<std::string> fSources;
};

static RootCpArgs ParseArgs(const char **args, int nArgs)
{
   using ROOT::RCmdLineOpts;

   RootCpArgs outArgs;

   RCmdLineOpts opts;
   opts.AddFlag({"-c", "--compress"}, RCmdLineOpts::EFlagType::kWithArg);
   opts.AddFlag({"--recreate"});
   opts.AddFlag({"--replace"});
   opts.AddFlag({"-r", "--recursive"});
   opts.AddFlag({"-h", "--help"});
   opts.AddFlag({"-v", "--verbose"});

   opts.Parse(args, nArgs);

   for (const auto &err : opts.GetErrors()) {
      std::cerr << err << "\n";
   }
   if (!opts.GetErrors().empty()) {
      outArgs.fPrintHelp = RootCpArgs::EPrintUsage::kShort;
      return outArgs;
   }

   if (opts.GetSwitch("help")) {
      outArgs.fPrintHelp = RootCpArgs::EPrintUsage::kLong;
      return outArgs;
   }

   if (auto val = opts.GetFlagValueAs<int>("compress"); val)
      outArgs.fCompression = val;
   outArgs.fRecursive = opts.GetSwitch("recursive");
   outArgs.fReplace = opts.GetSwitch("replace");
   outArgs.fRecreate = opts.GetSwitch("recreate");

   if (opts.GetSwitch("verbose"))
      SetLogVerbosity(2);

   outArgs.fSources = opts.GetArgs();
   if (outArgs.fSources.size() < 2)
      outArgs.fPrintHelp = RootCpArgs::EPrintUsage::kShort;

   return outArgs;
}

static std::unique_ptr<TFile> OpenFile(const char *fileName, const char *mode)
{
   const auto origLv = gErrorIgnoreLevel;
   gErrorIgnoreLevel = kError;
   auto file = std::unique_ptr<TFile>(TFile::Open(fileName, mode));
   if (!file || file->IsZombie()) {
      Err() << "File " << fileName << "does not exist.\n";
      return nullptr;
   }
   gErrorIgnoreLevel = origLv;
   return file;
}

namespace {

struct RootCpDestination {
   TFile *fFile;
   std::string fPath;
   std::string fFname;
   bool fIsNewObject;
};

} // namespace

// Splits `path` into a directory path (excluding the trailing '/') and a basename.
static std::pair<std::string_view, std::string_view> DecomposePath(std::string_view path)
{
   auto lastSlashIdx = path.rfind('/');
   if (lastSlashIdx == std::string_view::npos)
      return {{}, path};

   auto dirName = path.substr(0, lastSlashIdx);
   auto pathName = path.substr(lastSlashIdx + 1);
   return {dirName, pathName};
}

// Copies `nodeIdx`-th node from `src`'s object tree to the file in `dest`.
// `nodeIdx` is assumed to be in range.
static void CopyNode(const RootSource &src, const RootCpDestination &dest, NodeIdx_t nodeIdx, const RootCpArgs &args)
{
   TFile *srcfile = src.fObjectTree.fFile.get();
   // The file is guaranteed to be valid by ParseRootSource: if this crashes, it's a bug in there.
   assert(srcfile);
   // Similarly, nodeIdx must be in range because it always comes from a RootObjTree.
   assert(nodeIdx < src.fObjectTree.fNodes.size());
   const RootObjNode &node = src.fObjectTree.fNodes[nodeIdx];
   const std::string srcFullPath = NodeFullPath(src.fObjectTree, nodeIdx);
   // Directory path, excluding trailing '/' and without the "file.root:" prefix.
   const std::string_view srcDirPath =
      (node.fParent == 0) ? std::string_view{}
                          : std::string_view{srcFullPath.data(), srcFullPath.size() - node.fName.size() - 1};

   // Figure out where the output goes. If the user specified an output path (i.e. if dest.fPath is not empty), then
   // use that. Otherwise, use the same path as the source object.
   std::string destFullPath;
   std::string_view destDirPath, destBaseName;
   if (dest.fIsNewObject || dest.fPath.empty()) {
      // User gave a destination which is not an existing directory or no destination at all
      destFullPath = dest.fPath.empty() ? srcFullPath : dest.fPath;
      auto decomposed = DecomposePath(destFullPath);
      destDirPath = decomposed.first;
      destBaseName = decomposed.second;
   } else if (!dest.fPath.empty()) {
      // User gave a destination which is an existing directory
      destDirPath = dest.fPath;
      destFullPath = std::string(destDirPath) + "/" + node.fName;
   }

   Info(1) << "cp " << src.fFileName << ":" << srcFullPath << " -> " << dest.fFname << ":" << destFullPath << "\n";

   TDirectory *destDir = dest.fFile;
   if (!destDirPath.empty()) {
      Info(2) << "mkdir " << destDirPath << "\n";
      destDir = dest.fFile->mkdir(std::string(destDirPath).c_str(), /* title = */ "",
                                  /* returnPreExisting = */ true);
   }

   // Check if the destination already exists. There are 3 cases here:
   // 1. it doesn't: just go on as normal;
   // 2. it does and it is a directory: the copied object needs to be copied inside it, but this was already accounted
   //    for outside CopyNode, so just go on as normal;
   // 3. it does and it's not a directory: if we have the replace flag, replace it, otherwise error out.
   const TKey *destKey = destDir->GetKey(std::string(destBaseName).c_str());
   if (destKey && !TClass::GetClass(destKey->GetClassName())->InheritsFrom("TDirectory") && !args.fReplace) {
      Err() << "an object of type '" << destKey->GetClassName() << "' already exists at " << dest.fFname << ':'
            << destFullPath << ". Use the --replace flag to overwrite existing objects.\n";
      return;
   }

   // retrieve the object's key
   const TDirectory *srcDir = srcfile->GetDirectory(std::string(srcDirPath).c_str(), true);
   if (!srcDir) {
      Err() << "failed to get source directory '" << srcDirPath << "'\n";
      return;
   }
   const TKey *srcKey = srcDir->GetKey(node.fName.c_str());
   if (!srcKey) {
      Err() << "failed to read key of object '" << srcFullPath << "'\n";
      return;
   }

   // Verify that the class is known and supported.
   const std::string &className = node.fClassName;
   const TClass *cl = TClass::GetClass(className.c_str());
   if (!cl) {
      Err() << "unknown object type: " << className << "; object will be skipped.\n";
      return;
   }

   Info(2) << "read object \"" << srcFullPath << "\" of type " << node.fClassName << "\n";
   if (!destDir) {
      Err() << "failed to create or get destination directory \"" << dest.fFname << ":" << destDirPath << "\"\n";
      return;
   }

   // Delete previous object if we're replacing it
   if (destKey && args.fReplace)
      destDir->Delete((std::string(destBaseName) + ";*").c_str());

   //
   // Do the actual copy
   //
   if (cl->InheritsFrom("TObject")) {
      TObject *obj = node.fKey->ReadObj();
      if (!obj) {
         Err() << "failed to read object \"" << srcFullPath << "\".\n";
         return;
      }

      if (TTree *old = dynamic_cast<TTree *>(obj)) {
         // special case for TTree
         TDirectory::TContext ctx(gDirectory, destDir);
         obj = old->CloneTree(-1, "fast");
         if (dest.fIsNewObject) {
            static_cast<TTree *>(obj)->SetName(std::string(destBaseName).c_str());
         }
         obj->Write();
         old->Delete();
      } else if (cl->InheritsFrom("TDirectory")) {
         // directory
         if (!args.fRecursive) {
            Warn() << "Directory '" << srcFullPath
                   << "' will not be copied. Use the -r option if you need a recursive copy.\n";
         } else {
            destDir->mkdir(node.fName.c_str(), srcKey->GetTitle(), true);
            RootCpDestination dest2 = dest;
            dest2.fPath = dest.fPath + (dest.fPath.empty() ? "" : "/") + node.fName;
            for (auto childIdx = node.fFirstChild; childIdx < node.fFirstChild + node.fNChildren; ++childIdx)
               CopyNode(src, dest2, childIdx, args);
         }
      } else {
         // regular TObject
         destDir->WriteObject(obj, std::string(destBaseName).c_str());
      }
      obj->Delete();
   } else {
      Warn() << "object '" << node.fName << "' of type '" << node.fClassName
             << "' will not be copied, as its type is currently unsupported by rootcp.\n";
   }
}

int main(int argc, char **argv)
{
   InitLog("rootcp");

   // Parse arguments
   auto args = ParseArgs(const_cast<const char **>(argv) + 1, argc - 1);
   if (args.fPrintHelp != RootCpArgs::EPrintUsage::kNo) {
      std::cerr << kShortHelp;
      if (args.fPrintHelp == RootCpArgs::EPrintUsage::kLong) {
         std::cerr << kLongHelp;
         return 0;
      }
      return 1;
   }

   // Get destination. In general it may be a string like "prefix://file.root:path/to/dir", so check if it refers to
   // a valid location.
   // First validate the destination syntax.
   const auto destFnameAndPattern = args.fSources.back();
   auto splitRes = SplitIntoFileNameAndPattern(destFnameAndPattern);
   if (!splitRes) {
      Err() << splitRes.GetError()->GetReport() << "\n";
      return 1;
   }
   auto [destFname, destPath] = splitRes.Unwrap();

   // Check if the operation is allowed.
   args.fSources.pop_back();
   if (args.fRecreate && std::find(args.fSources.begin(), args.fSources.end(), destFname) != args.fSources.end()) {
      Err() << "cannot recreate destination file if this is also a source file\n";
      return 1;
   }

   if (args.fCompression && gSystem->AccessPathName(std::string(destFname).c_str())) {
      Err() << "can't change compression settings on existing file " << destFname << "\n";
      return 1;
   }

   const char *destFileMode =
      args.fRecreate ? "RECREATE_WITHOUT_GLOBALREGISTRATION" : "UPDATE_WITHOUT_GLOBALREGISTRATION";
   auto destFile = OpenFile(std::string(destFname).c_str(), destFileMode);
   if (!destFile)
      return 1;

   // `destPath` is the part after the colon (the input is given as `destFname:destPath`). It may be empty, but
   // if it's not, it must refer to either an existing TDirectory inside the file or to a non-existing object (it may
   // also be an existing object if --replace was passed).
   TKey *destDirKey = nullptr;
   if (!destPath.empty()) {
      destDirKey = destFile->GetKey(std::string(destPath).c_str());
      if (destDirKey && !TClass::GetClass(destDirKey->GetClassName())->InheritsFrom("TDirectory")) {
         if (!args.fReplace) {
            // This error would be caught later in CopyNode, but since we can detect it early let's bail out before
            // wasting time touching other files.
            Err() << "destination path \"" << destFname << ":" << destPath << "\" already exists (as an object of type "
                  << destDirKey->GetClassName() << "). Use the --replace flag to overwrite it.\n";
            return 1;
         } else {
            destDirKey = nullptr;
         }
      }
   }

   // If we are copying multiple objects the destination path must either be empty or a TDirectory.
   const bool destIsNewObject = !destPath.empty() && !destDirKey;
   if (destIsNewObject && args.fSources.size() > 1) {
      Err() << "multiple sources were specified, but destination path \"" << destFname << ":" << destPath
            << "\" is not a directory.\n";
      return 1;
   }

   if (args.fCompression)
      destFile->SetCompressionSettings(*args.fCompression);

   const std::uint32_t flags = args.fRecursive * EGetMatchingPathsFlags::kRecursive;
   // const bool oneFile = args.fSources.size() == 1;
   for (const auto &srcFname : args.fSources) {
      auto src = ROOT::CmdLine::ParseRootSource(srcFname, flags);
      if (!src.fErrors.empty()) {
         for (const auto &err : src.fErrors)
            Err() << err << "\n";
         return 1;
      }

      // We should never register files to the global list for performance reasons.
      assert(!gROOT->GetListOfFiles()->Contains(src.fObjectTree.fFile.get()));

      // If we are copying multiple objects the destination path must either be empty or a TDirectory.
      if (destIsNewObject && src.fObjectTree.fLeafList.size() + src.fObjectTree.fDirList.size() > 1) {
         Err() << "multiple sources were specified but destination path \"" << destFname << ":" << destPath
               << "\" is not a directory.\n";
         return 1;
      }

      // Iterate all objects we need to copy
      RootCpDestination dest;
      dest.fFile = destFile.get();
      dest.fFname = destFname;
      dest.fIsNewObject = destIsNewObject;
      dest.fPath = destPath;
      for (auto nodeIdx : src.fObjectTree.fLeafList) {
         CopyNode(src, dest, nodeIdx, args);
      }
      for (auto nodeIdx : src.fObjectTree.fDirList) {
         if (nodeIdx == 0) {
            // The root file node needs special treatment; for all other "top-level" directories, CopyNode handles them.
            const auto &node = src.fObjectTree.fNodes[nodeIdx];
            for (auto childIdx = node.fFirstChild; childIdx < node.fFirstChild + node.fNChildren; ++childIdx)
               CopyNode(src, dest, childIdx, args);
         } else {
            CopyNode(src, dest, nodeIdx, args);
         }
      }
   }

   return 0;
}
