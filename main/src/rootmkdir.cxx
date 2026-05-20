/// \file rootmkdir.cxx
///
/// Command line tool to create directories in ROOT files
///
/// \author Giacomo Parolini <giacomo.parolini@cern.ch>
/// \date 2026-03-06
#include <ROOT/RLogger.hxx>

#include "logging.hxx"
#include "optparse.hxx"
#include "RootObjTree.hxx"
#include "RootObjTree.cxx"

#include <TClass.h>
#include <TClassRef.h>
#include <TError.h>
#include <TFile.h>
#include <TROOT.h>
#include <TSystem.h>

#include <iostream>
#include <memory>
#include <string_view>
#include <vector>

using namespace ROOT::CmdLine;

static const char *const kShortHelp = "usage: rootmkdir [-p|--parents] [-v|--verbose] FILE:path/to/dir [...]\n";
static const char *const kLongHelp = R"(
Add directories in ROOT files

positional arguments:
  FILE:path             File(s) and path(s) of directories to create

options:
  -h, --help            get this help
  -p, --parents         create parent directories if needed, no error if directory already exists
  -v, --verbose         be verbose (can be repeated for increased verbosity)

Note that calling rootmkdir without a directory path can be used to create files (similar to the `touch` command).
You can also simultaneously create files and directories in it if you pass the -p flag.

examples:
- rootmkdir example.root:dir
  Add the directory 'dir' to the ROOT file 'example.root' (file must exists)

- rootmkdir example.root:dir1/dir2
  Add the directory 'dir2' in 'dir1' which exists in the ROOT file 'example.root'

- rootmkdir -p example.root:dir1/dir2/dir3
  Make parent directories of 'dir3' as needed, no error if target directory already exists

- rootmkdir non_existent.root
  Create an empty ROOT file named 'non_existent.root'

- rootmkdir -p non_existent.root:dir1
  Create a ROOT file named 'non_existent.root' and directory `dir1` in it
)";

struct RootMkdirArgs {
   enum class EPrintUsage {
      kNo,
      kShort,
      kLong
   };
   EPrintUsage fPrintHelp = EPrintUsage::kNo;
   bool fCreateParents = false;
   std::vector<std::string> fDstFileAndPaths;
};

static RootMkdirArgs ParseArgs(const char **args, int nArgs)
{
   using ROOT::RCmdLineOpts;

   RootMkdirArgs outArgs;

   RCmdLineOpts opts;
   opts.AddFlag({"-h", "--help"});
   opts.AddFlag({"-p", "--parents"});
   opts.AddFlag({"-v", "--verbose"}, RCmdLineOpts::EFlagType::kSwitch, "", RCmdLineOpts::kFlagAllowMultiple);

   opts.Parse(args, nArgs);

   for (const auto &err : opts.GetErrors()) {
      std::cerr << err << "\n";
   }
   if (!opts.GetErrors().empty()) {
      outArgs.fPrintHelp = RootMkdirArgs::EPrintUsage::kShort;
      return outArgs;
   }

   if (opts.GetSwitch("help")) {
      outArgs.fPrintHelp = RootMkdirArgs::EPrintUsage::kLong;
      return outArgs;
   }

   SetLogVerbosity(opts.GetSwitch("v"));

   outArgs.fCreateParents = opts.GetSwitch("parents");

   outArgs.fDstFileAndPaths = opts.GetArgs();
   if (outArgs.fDstFileAndPaths.size() < 1)
      outArgs.fPrintHelp = RootMkdirArgs::EPrintUsage::kShort;

   return outArgs;
}

static bool IsDirectory(const TKey &key)
{
   static const TClassRef dirClassRef("TDirectory");
   const auto *dirClass = TClass::GetClass(key.GetClassName());
   return dirClass && dirClass->InheritsFrom(dirClassRef);
}

static bool MakeDirectory(TFile &file, std::string_view dirPath, bool createParents)
{
   // Partially copy-pasted from RFile::PutUntyped

   const auto tokens = ROOT::Split(dirPath, "/");
   const auto FullPathUntil = [&tokens](auto idx) {
      return ROOT::Join("/", std::span<const std::string>{tokens.data(), idx + 1});
   };
   TDirectory *dir = &file;
   for (auto tokIdx = 0u; tokIdx < tokens.size() - 1; ++tokIdx) {
      TKey *existing = dir->GetKey(tokens[tokIdx].c_str());
      // 4 cases here:
      //  1. subdirectory exists? -> no problem.
      //  2. non-directory object exists with the name of the subdirectory? -> error.
      //  3. subdirectory does not exist and we passed -p? -> create it
      //  4. subdirectory does not exist and we didn't pass -p? -> error.
      if (existing) {
         if (!IsDirectory(*existing)) {
            Err() << "error adding '" + std::string(dirPath) + "': path '" + FullPathUntil(tokIdx) +
                        "' is already taken by an object of type '" + existing->GetClassName() + "'\n";
            return false;
         }
         dir = existing->ReadObject<TDirectory>();
      } else if (createParents) {
         dir = dir->mkdir(tokens[tokIdx].c_str(), "", false);
         if (dir)
            Info(1) << "created directory '" << file.GetName() << ':' << FullPathUntil(tokIdx) << "'\n";
      } else {
         Err() << "cannot create directory '" + std::string(dirPath) + "': parent directory '" + FullPathUntil(tokIdx) +
                     "' does not exist. If you want to create the entire hierarchy, use the -p flag.\n";
         return false;
      }

      if (!dir)
         return false;
   }

   const TKey *existing = dir->GetKey(tokens[tokens.size() - 1].c_str());
   if (existing) {
      if (!IsDirectory(*existing)) {
         Err() << "error adding '" + std::string(dirPath) + "': path is already taken by an object of type '" +
                     existing->GetClassName() + "'\n";
         return false;
      } else if (!createParents) {
         Err() << "error adding '" + std::string(dirPath) + "': a directory already exists at that path.\n";
         return false;
      }

      // directory already exists and we passed -p.
      return true;
   }

   auto newDir = dir->mkdir(tokens[tokens.size() - 1].c_str(), "", false);
   if (newDir)
      Info(1) << "created directory '" << file.GetName() << ':' << std::string(dirPath) << "'\n";

   return newDir != nullptr;
}

static bool ValidateDirPath(std::string_view path)
{
   if (path.rfind(';') != std::string_view::npos) {
      Err() << "cannot specify cycle for the directory to create.\n";
      return false;
   }
   return true;
}

int main(int argc, char **argv)
{
   InitLog("rootmkdir");

   // Parse arguments
   auto args = ParseArgs(const_cast<const char **>(argv) + 1, argc - 1);
   if (args.fPrintHelp != RootMkdirArgs::EPrintUsage::kNo) {
      std::cerr << kShortHelp;
      if (args.fPrintHelp == RootMkdirArgs::EPrintUsage::kLong) {
         std::cerr << kLongHelp;
         return 0;
      }
      return 1;
   }

   // Validate and split all arguments into filename + dirpath
   std::vector<std::pair<std::string_view, std::string_view>> dstFilesAndPaths;
   dstFilesAndPaths.reserve(args.fDstFileAndPaths.size());
   for (const auto &src : args.fDstFileAndPaths) {
      auto res = SplitIntoFileNameAndPattern(src);
      if (!res) {
         Err() << res.GetError()->GetReport() << "\n";
         return 1;
      }
      auto fnameAndPath = res.Unwrap();
      if (!ValidateDirPath(fnameAndPath.second))
         return 1;
      dstFilesAndPaths.push_back(fnameAndPath);
   }

   // Create the directories
   bool errors = false;
   for (const auto &[dstFname, dstPath] : dstFilesAndPaths) {
      const std::string fname{dstFname};
      const bool fileExists = gSystem->AccessPathName(fname.c_str(), kFileExists) == 0;
      // If the file does not exist we attempt to create it in the following cases:
      //   1. dstPath is empty
      //   2. dstPath is non-empty and the user passed the -p flag.
      if (!fileExists && !(dstPath.empty() || args.fCreateParents)) {
         Err() << "cannot create directory '" << dstFname << ":" << dstPath
               << "': file does not exist. Use the -p flag if you want to create the file alongside the directories.\n";
         errors = true;
         continue;
      } else if (fileExists && dstPath.empty() && !args.fCreateParents) {
         Err() << "cannot create file '" << fname << "': already exists.\n";
         errors = true;
         continue;
      }

      auto file = std::unique_ptr<TFile>(TFile::Open(fname.c_str(), "UPDATE"));
      if (!file || file->IsZombie()) {
         Err() << "failed to open '" << fname << "' for writing.\n";
         errors = true;
         continue;
      } else if (!fileExists) {
         Info(1) << "created file '" << fname << "'\n";
      }

      if (!dstPath.empty())
         errors |= !MakeDirectory(*file, dstPath, args.fCreateParents);
   }

   return errors;
}
