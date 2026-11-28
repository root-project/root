// rootmerge
//
// An alternative frontend to hadd/TFileMerger.
// @author Giacomo Parolini
//
#include "Compression.h"
#include "TClass.h"
#include "TFile.h"
#include "TFileMerger.h"
#include "THashList.h"
#include "TKey.h"
#include "TSystem.h"
#include "TUUID.h"

#include <ROOT/RConfig.hxx>
#include <ROOT/StringConv.hxx>
#include <ROOT/TIOFeatures.hxx>
#include <ROOT/RFile.hxx>

#include "haddCommandLineOptionsHelp.h"
#include "logging.hxx"

#include "optparse.hxx"

#include <climits>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <streambuf>
#include <variant>

#ifndef R__WIN32
#include "ROOT/TProcessExecutor.hxx"
#endif

////////////////////////////////////////////////////////////////////////////////

// NOTE: TFileMerger will use PrintLevel = GetLogVerbosity() - 1. If PrintLevel is < 1, it will print nothing, otherwise
// it will print everything. To give some granularity to hadd, we do the following:
// LogVerbosity = 0: only print hadd errors
// LogVerbosity = 1: only print hadd errors + warnings
// LogVerbosity = 2: print hadd errors + warnings and TFileMerger messages
// LogVerbosity > 2: print all hadd and TFileMerger messages.
static constexpr int kDefaultVerbosity = 2;

using IntFlag_t = uint32_t;
struct CompressionKeepFirstInput {};

enum class EObjectFilterType {
   kUnspecified,
   kOnlyListed = TFileMerger::kOnlyListed,
   kSkipListed = TFileMerger::kSkipListed,
};

enum class EHelpType {
   kNone,
   kShort,
   kLong,
};

struct HAddArgs {
   bool fAppend = false;
   bool fForce = false;
   bool fSkipErrors = false;
   std::string fHelp;

   struct TTreeOpts {
      bool fNoTrees = false;
      bool fReoptimize = false;
      bool fKeepCompressionAsIs = false;
   } fTTree;

   std::string fWorkingDir;
   std::optional<IntFlag_t> fNProcesses;
   std::string fObjectFilterFile;
   EObjectFilterType fObjectFilterType = EObjectFilterType::kUnspecified;
   TString fCacheSize;
   std::optional<IntFlag_t> fMaxOpenedFiles;
   IntFlag_t fVerbosity;
   std::variant<std::monostate, IntFlag_t, CompressionKeepFirstInput> fCompressionSettings;
   ROOT::TIOFeatures fFeatures;

   std::string fTarget;
   std::vector<std::string> fSources;
};

enum class EFlagResult {
   kIgnored,
   kParsed,
   kErr
};

template <typename T>
struct FlagConvResult {
   T fValue;
   EFlagResult fResult;
};

EFlagResult ConvertIOFeatures(std::string_view arg, ROOT::TIOFeatures &features)
{
   std::stringstream ss;
   ss << arg;
   std::string item;
   while (std::getline(ss, item, ',')) {
      if (!features.Set(item))
         Warn() << "ignoring unknown feature request: " << item << "\n";
   }
   return EFlagResult::kParsed;
}

static FlagConvResult<TString> ConvertCacheSize(std::string_view arg)
{
   TString cacheSize;
   int size;
   auto parseResult = ROOT::FromHumanReadableSize(arg, size);
   if (parseResult == ROOT::EFromHumanReadableSize::kParseFail) {
      Err() << "could not parse the cache size passed after --cachesize: '" << arg << "'\n";
      return {"", EFlagResult::kErr};
   } else if (parseResult == ROOT::EFromHumanReadableSize::kOverflow) {
      double m;
      const char *munit = nullptr;
      ROOT::ToHumanReadableSize(INT_MAX, false, &m, &munit);
      Warn() << "the cache size passed after -cachesize is too large: " << arg << " is greater than " << m << munit
             << ". We will use the maximum value.\n";
      return {std::to_string(m) + munit, EFlagResult::kParsed};
   } else {
      cacheSize = "cachesize=";
      cacheSize.Append(std::string(arg));
   }
   return {cacheSize, EFlagResult::kParsed};
}

static bool ValidCompressionSettings(int compSettings)
{
   // Must be a number between 0 and 509 (with a 0 in the middle)
   if (compSettings == 0)
      return true;
   // We also accept [1-9] as aliases of [101-109], but it's discouraged.
   if (compSettings >= 1 && compSettings <= 9) {
      Warn() << "interpreting " << compSettings << " as " << 100 + compSettings
             << "."
                " This behavior is deprecated, please use the full compression settings.\n";
      return true;
   }
   return (compSettings >= 100 && compSettings <= 509) && ((compSettings / 10) % 10 == 0);
}

// Returns the flags to add to the file merger's flags, or -1 in case of errors.
static int ParseFilterFile(std::string_view filterFileName, EObjectFilterType objectFilterType, TFileMerger &fileMerger)
{
   if (filterFileName.empty())
      return 0;

   std::ifstream filterFile{std::string(filterFileName)};
   if (!filterFile) {
      Err() << "error opening filter file '" << filterFileName << "'\n";
      return -1;
   }
   TString filteredObjects;
   std::string line;
   std::string objPath;
   int nObjects = 0;
   while (std::getline(filterFile, line)) {
      std::istringstream ss(line);
      // only read exactly 1 token per line (strips any whitespaces and such)
      objPath.clear();
      ss >> objPath;
      if (!objPath.empty() && objPath[0] != '#') {
         filteredObjects.Append(objPath + ' ');
         ++nObjects;
      }
   }

   if (nObjects) {
      Info(2) << "added " << nObjects << " object from filter file '" << filterFileName << "'\n";
      fileMerger.AddObjectNames(filteredObjects);
   } else {
      Warn() << "no objects were added from filter file '" << filterFileName << "'\n";
   }

   assert(static_cast<int>(objectFilterType) == TFileMerger::kSkipListed ||
          static_cast<int>(objectFilterType) == TFileMerger::kOnlyListed);
   return static_cast<int>(objectFilterType);
}

static bool FilesAreEquivalent(std::string_view source, std::string_view target)
{
   const bool sourceHasProtocol = source.find_first_of("://") == std::string_view::npos;
   const bool targetHasProtocol = target.find_first_of("://") == std::string_view::npos;
   if (sourceHasProtocol != targetHasProtocol)
      return false;

   // We cannot use std::filesystem functions for file paths that have a protocol.
   if (!sourceHasProtocol)
      return source == target;

   std::error_code ec;
   const bool equiv = std::filesystem::equivalent(source, target, ec);
   return !ec && equiv;
}

static EObjectFilterType ConvertObjectFilterType(std::string_view filtType)
{
   if (filtType.empty())
      return EObjectFilterType::kUnspecified;
   if (filtType == "OnlyListed")
      return EObjectFilterType::kOnlyListed;
   if (filtType == "SkipListed")
      return EObjectFilterType::kSkipListed;

   throw std::runtime_error(std::string("invalid value for --list-type: ") + std::string(filtType) +
                            ". Valid values are: OnlyListed, SkipListed.");
}

static HAddArgs ParseArgs(const char **args, int nArgs)
{
   using FT = ROOT::RCmdLineOpts::EFlagType;

   ROOT::RCmdLineOpts opts;
   opts.AddFlag({"-a", "--append"}, FT::kSwitch, "Append to the output");
   opts.AddFlag({"-c", "--compression"}, FT::kWithArg,
                "Set target compression. The argument must be a number of the format `i0j`, where `i` represents the "
                "compression algorithm and `j` the compression level used."
                " 0 means 'uncompressed'. For the compression algorithm: 1=Zlib, 2=LZMA, 4=LZ4, 5=ZSTD. For the "
                "compression level, any number from 0 to 9 is accepted. See ROOT::RCompressionSetting and TFile::TFile "
                "documentation for more details. If \"same\" is passed instead of a number, use the same compression "
                "as the first input file.");
   opts.AddFlag({"--cachesize"}, FT::kWithArg,
                "Resize the prefetching cache used to speed up I/O operations (0 to disable)");
   opts.AddFlag({"--workdir"}, FT::kWithArg, "Carry out the partial multiprocess execution in the specified directory");
   opts.AddFlag({"-f", "--force-overwrite"}, FT::kSwitch, "Force overwriting of output file");
   opts.AddFlag({"-j", "--jobs"}, FT::kWithArg,
                "Parallelise the execution in N processes. Pass 0 to use the system maximum");
   opts.AddFlag({"-k", "--skip-errors"}, FT::kSwitch, "Skip corrupt or non-existent files, do not exit");
   opts.AddFlag({"-L", "--from-list"}, FT::kWithArg,
                "Read the list of objects from FILE and either only merge or skip those objects depending on"
                " the value of \"--list-type\". FILE must contain one object name per line, which cannot contain"
                " whitespaces or '/'. You can also pass TDirectory names, which apply to the entire directory"
                " content. Lines beginning with '#' are ignored. If this flag is passed, \"--list-type\" MUST be"
                " passed as well.");
   opts.AddFlag({"--Ltype", "--list-type"}, FT::kWithArg,
                "Sets the type of operation performed on the objects listed in the file given with the \"-L\" flag."
                "\"SkipListed\" will skip all the listed objects; \"OnlyListed\" will only merge those"
                " objects. If this flag is passed, \"-L\" must be passed as well.");
   opts.AddFlag({"-n", "--max-files"}, FT::kWithArg,
                "Open at most N files at once (pass 0 to use the system maximum - which is also the default)."
                " This number includes both the input reading files as well as the output file."
                " Thus, if set to 1, it will be automatically replaced to a minimum of 2. If set to too large of a "
                " value, it will be capped to the system maximum.");
   opts.AddFlag({"-v"}, FT::kSwitch, "Set verbosity level. Pass multiple times to increase verbosity.",
                ROOT::RCmdLineOpts::kFlagAllowMultiple);
   opts.AddFlag({"--experimental-io-features"}, FT::kWithArg,
                "Enables the corresponding experimental feature for the output",
                ROOT::RCmdLineOpts::kFlagAllowMultiple);
   // TTree-specific flags
   opts.AddFlag({"--ttree-reoptimize-basket-size"}, FT::kSwitch, "Re-optimize basket size when merging TTrees");
   opts.AddFlag({"--ttree-skip"}, FT::kSwitch, "Do not merge TTrees");
   // TODO: make shared with RNTuple
   opts.AddFlag({"--ttree-keep-basket-compression"}, FT::kSwitch,
                "Sets the target file to contain TTree baskets with the same compression as the input files"
                " (unless --ttree-reoptimize-basket-size is specified). Compresses the meta data using the compression "
                "level specified"
                " in the first input or the compression setting given with -c (for example 505 when using -c 505 "
                "--ttree-keep-basket-compression)");
   opts.AddFlag({"-h", "--help"}, FT::kSwitch, "Get a help message");

   opts.Parse(args, nArgs);
   assert(opts.GetSettings().fFlagTreatment == ROOT::RCmdLineOpts::EFlagTreatment::kGrouped);

   HAddArgs parsedArgs;
   auto doHelp = EHelpType::kNone;
   if (opts.GetErrors().empty()) {
      try {
         parsedArgs.fAppend = opts.GetSwitch("append");
         if (const auto cacheSize = opts.GetFlagValue("cachesize"); !cacheSize.empty()) {
            if (const auto parsedCacheSize = ConvertCacheSize(cacheSize);
                parsedCacheSize.fResult == EFlagResult::kParsed)
               parsedArgs.fCacheSize = parsedCacheSize.fValue;
         }
         parsedArgs.fWorkingDir = opts.GetFlagValue("workdir");
         for (const auto ioFeatures : opts.GetFlagValues("experimental-io-features")) {
            if (ConvertIOFeatures(ioFeatures, parsedArgs.fFeatures) != EFlagResult::kParsed)
               doHelp = EHelpType::kShort;
         }
         parsedArgs.fForce = opts.GetSwitch("force-overwrite");
         if (const auto compSettings = opts.GetFlagValue("compression"); compSettings == "same") {
            parsedArgs.fCompressionSettings = CompressionKeepFirstInput{};
         } else {
            if (const auto compSettingsInt = opts.GetFlagValueAs<IntFlag_t>("compression")) {
               if (ValidCompressionSettings(*compSettingsInt)) {
                  parsedArgs.fCompressionSettings = *compSettingsInt;
               } else {
                  doHelp = EHelpType::kShort;
               }
            }
         }
         parsedArgs.fTTree.fKeepCompressionAsIs = opts.GetSwitch("ttree-keep-basket-compression");
         parsedArgs.fNProcesses = opts.GetFlagValueAs<IntFlag_t>("jobs");
         parsedArgs.fSkipErrors = opts.GetSwitch("skip-errors");
         parsedArgs.fObjectFilterFile = opts.GetFlagValue("from-list");
         parsedArgs.fObjectFilterType = ConvertObjectFilterType(opts.GetFlagValue("list-type"));
         parsedArgs.fTTree.fNoTrees = opts.GetSwitch("ttree-skip");
         parsedArgs.fMaxOpenedFiles = opts.GetFlagValueAs<IntFlag_t>("max-files");
         parsedArgs.fTTree.fReoptimize = opts.GetSwitch("ttree-reoptimize-basket-size");
         parsedArgs.fTTree.fNoTrees = opts.GetSwitch("ttree-skip");
         parsedArgs.fVerbosity = opts.GetSwitch("v");
         if (parsedArgs.fVerbosity == 0)
            parsedArgs.fVerbosity = kDefaultVerbosity;

         parsedArgs.fSources = opts.GetArgs();
         if (parsedArgs.fSources.size() < 2) {
            if (parsedArgs.fSources.empty())
               Err() << "missing output file.";
            else
               Err() << "missing input file.";
            doHelp = EHelpType::kShort;
         } else {
            parsedArgs.fTarget = parsedArgs.fSources.front();
            parsedArgs.fSources.erase(parsedArgs.fSources.begin());
         }
      } catch (const std::runtime_error &ex) {
         Err() << ex.what() << "\n";
         doHelp = EHelpType::kShort;
      }
   } else {
      for (const auto &err : opts.GetErrors()) {
         Err() << err << "\n";
      }
      doHelp = EHelpType::kShort;
   }

   // Done at the end so it doesn't get overridden.
   if (opts.GetSwitch("help"))
      doHelp = EHelpType::kLong;

   ROOT::RCmdLineOpts::RMakeHelpOptions helpOpts;
   helpOpts.fPositionalArgs = "<target> <source1> [source2...]";
   if (doHelp == EHelpType::kShort)
      parsedArgs.fHelp = opts.MakeShortHelp(Detail::gLogName, helpOpts);
   else if (doHelp == EHelpType::kLong)
      parsedArgs.fHelp = opts.MakeLongHelp(Detail::gLogName, helpOpts);

   return parsedArgs;
}

int main(int argc, const char **argv)
{
   InitLog("rootmerge", kDefaultVerbosity);

   auto args = ParseArgs(argv + 1, argc - 1);
   if (!args.fHelp.empty()) {
      std::cerr << args.fHelp << "\n";
      return 0;
   }

   const auto maxOpenedFiles = args.fMaxOpenedFiles.value_or(0);
   SetLogVerbosity(args.fVerbosity);

   // For the -j flag (nProcesses), we check if the flag is present and, if so, if it has a
   // valid value (i.e. any value > 0).
   // If the flag is present at all, we do multiprocessing. If the value of nProcesses is invalid,
   // we default to the number of cpus on the machine.
   bool multiprocess = args.fNProcesses.has_value();
   int nProcesses;
   if (args.fNProcesses && *args.fNProcesses > 0) {
      nProcesses = *args.fNProcesses;
   } else {
      SysInfo_t s;
      gSystem->GetSysInfo(&s);
      nProcesses = s.fCpus;
   }
   if (multiprocess)
      Info(2) << "parallelizing  with " << nProcesses << " processes.\n";

   // If the user specified a workingDir, use that. Otherwise, default to the system temp dir.
   if (args.fWorkingDir.empty()) {
      args.fWorkingDir = gSystem->TempDirectory();
   } else if (!args.fWorkingDir.empty() && gSystem->AccessPathName(args.fWorkingDir.c_str())) {
      Err() << "could not access the directory specified: " << args.fWorkingDir << ".\n";
      return 1;
   }

   // Verify that -L and -Ltype are either both present or both absent.
   if (args.fObjectFilterFile.empty() != (args.fObjectFilterType == EObjectFilterType::kUnspecified)) {
      Err() << "-L must always be passed along with --Ltype.\n";
      return 1;
   }

   if (args.fCacheSize.size() > 0)
      Info(2) << "Using " << args.fCacheSize << "\n";

   ////////////////////////////// end flags processing /////////////////////////////////

   gSystem->Load("libTreePlayer");

   TFileMerger fileMerger(kFALSE, kFALSE);
   fileMerger.SetMsgPrefix(Detail::gLogName);
   fileMerger.SetPrintLevel(GetLogVerbosity() - 1);
   if (maxOpenedFiles > 0) {
      fileMerger.SetMaxOpenedFiles(maxOpenedFiles);
   }

   // Expand all indirect files (those specified with '@' at the start) and validate all sources
   // (must be accessible and different from the target).
   for (auto it = args.fSources.begin(); it != args.fSources.end(); ++it) {
      // NOTE: intentionally copying the string so we don't have to worry about the conditional erase() later
      const auto source = *it;
      assert(!source.empty());
      if (source[0] == '@') {
         std::ifstream indirectFile(source.substr(1));
         // remove the indirect file itself from the list of sources, it'll be replaced by its content.
         it = args.fSources.erase(it);
         if (!indirectFile.is_open()) {
            Err() << "could not open indirect file \"" << source.substr(1) << "\"" << std::endl;
            if (!args.fSkipErrors)
               return 1;
         } else {
            std::string line;
            while (indirectFile) {
               if (std::getline(indirectFile, line) && line.length()) {
                  if (gSystem->AccessPathName(line.c_str(), kReadPermission) == kTRUE) {
                     Err() << "could not validate the file name \"" << line << "\" within indirect file "
                           << source.substr(1) << std::endl;
                     if (!args.fSkipErrors)
                        return 1;
                  } else if (FilesAreEquivalent(line, args.fTarget)) {
                     Err() << "file \"" << line << "\" cannot be both the target and an input!\n";
                     if (!args.fSkipErrors)
                        return 1;
                  } else {
                     it = args.fSources.insert(it, line);
                  }
               }
            }
         }
      } else if (gSystem->AccessPathName(source.c_str(), kReadPermission) == kTRUE) {
         Err() << "could not validate argument \"" << source << "\" as input file " << std::endl;
         if (!args.fSkipErrors)
            return 1;
      } else if (FilesAreEquivalent(source, args.fTarget)) {
         Err() << "file \"" << source << "\" cannot be both the target and an input!\n";
         if (!args.fSkipErrors)
            return 1;
      }
   }

   if (args.fSources.empty()) {
      Err() << "no valid input file." << std::endl;
      return 1;
   }

   // Determine output compression
   struct {
      int operator()(std::monostate)
      {
         // no compression specified: use default.
         fFileMerger.SetMergeOptions(TString("DefaultCompression"));
         return ROOT::RCompressionSetting::EDefaults::kUseCompiledDefault;
      }
      int operator()(IntFlag_t compSettings) { return compSettings; }
      int operator()(CompressionKeepFirstInput)
      {
         // use compression from the first file.
         fFileMerger.SetMergeOptions(TString("FirstSrcCompression"));
         try {
            const auto firstInput = ROOT::RFile::Open(fArgs.fSources.front());
            return firstInput->GetFileHeader().GetCompression();
         } catch (...) {
            return ROOT::RCompressionSetting::EDefaults::kUseCompiledDefault;
         }
      }
      TFileMerger &fFileMerger;
      HAddArgs &fArgs;
   } visitor{fileMerger, args};

   const auto newCompression = std::visit(visitor, args.fCompressionSettings);

   {
      std::string msg = "compression settings for output file: " + std::to_string(newCompression);
      if (args.fTTree.fKeepCompressionAsIs && !args.fTTree.fReoptimize)
         msg += " (TTree data will keep existing compression)";
      Info(2) << msg << "\n";
   }

   if (args.fAppend) {
      if (!fileMerger.OutputFile(args.fTarget.c_str(), "UPDATE", newCompression)) {
         Err() << "error opening target file for update :" << args.fTarget << ".\n";
         return 2;
      }
   } else if (!fileMerger.OutputFile(args.fTarget.c_str(), args.fForce, newCompression)) {
      std::stringstream ss;
      ss << "error opening target file (does " << args.fTarget << " exist?).\n";
      if (!args.fForce)
         ss << "pass \"-f\" argument to force re-creation of output file.\n";
      Err() << ss.str();
      return 1;
   }

   auto filesPerProcess = (args.fSources.size() + nProcesses - 1) / nProcesses;
   if (multiprocess && filesPerProcess < 3) {
      // At least 3 files per process
      filesPerProcess = 3;
      nProcesses = (args.fSources.size() + filesPerProcess - 1) / filesPerProcess;
      Info(2) << "each process should handle at least 3 files for efficiency."
                 " Setting the number of processes to: "
              << nProcesses << std::endl;
   }
   if (nProcesses == 1)
      multiprocess = false;

   std::vector<std::string> partialFiles;

#ifndef R__WIN32
   // this is commented out only to try to prevent false positive detection
   // from several anti-virus engines on Windows, and multiproc is not
   // supported on Windows anyway
   if (multiprocess) {
      TUUID uuid;
      auto partialTail = uuid.AsString();
      for (auto i = 0; (i * filesPerProcess) < args.fSources.size(); i++) {
         std::stringstream buffer;
         buffer << args.fWorkingDir << "/partial" << i << "_" << partialTail << ".root";
         partialFiles.emplace_back(buffer.str());
      }
   }
#endif

   auto mergeFiles = [&](TFileMerger &merger) {
      if (args.fTTree.fReoptimize) {
         merger.SetFastMethod(kFALSE);
      } else {
         if (!args.fTTree.fKeepCompressionAsIs && merger.HasCompressionChange()) {
            // Don't warn if the user has requested any re-optimization.
            Warn() << "Sources and Target have different compression settings: merging will be slower\n";
         }
      }
      merger.SetNotrees(args.fTTree.fNoTrees);
      merger.SetMergeOptions(TString(merger.GetMergeOptions()) + " " + args.fCacheSize);
      merger.SetErrorBehavior(args.fSkipErrors ? TFileMerger::EErrorBehavior::kSkipOnError
                                               : TFileMerger::EErrorBehavior::kFailOnError);
      merger.SetIOFeatures(args.fFeatures);
      auto fileMergerFlags = static_cast<int>(TFileMerger::kAll);
      const auto extraFlags = ParseFilterFile(args.fObjectFilterFile, args.fObjectFilterType, merger);
      if (extraFlags < 0)
         return false;
      fileMergerFlags |= extraFlags;
      if (args.fAppend)
         fileMergerFlags |= TFileMerger::kIncremental;
      else
         fileMergerFlags |= TFileMerger::kRegular;
      Bool_t status = merger.PartialMerge(fileMergerFlags);
      return status;
   };

   auto sequentialMerge = [&args, &mergeFiles](TFileMerger &merger, int start, int nFiles) {
      const auto &sources = args.fSources;
      for (auto i = start; i < (start + nFiles) && i < static_cast<int>(sources.size()); i++) {
         if (!merger.AddFile(sources[i].c_str())) {
            if (args.fSkipErrors) {
               Warn() << "skipping file with error: " << sources[i] << std::endl;
            } else {
               Err() << "exiting due to error in " << sources[i] << std::endl;
               return kFALSE;
            }
         }
      }
      return mergeFiles(merger);
   };

   auto parallelMerge = [&](int start) {
      TFileMerger mergerP(kFALSE, kFALSE);
      mergerP.SetMsgPrefix("hadd");
      mergerP.SetPrintLevel(GetLogVerbosity() - 1);
      if (maxOpenedFiles > 0) {
         mergerP.SetMaxOpenedFiles(maxOpenedFiles / nProcesses);
      }
      const auto &fileName = partialFiles[start / filesPerProcess];
      if (!mergerP.OutputFile(fileName.c_str(), args.fForce, newCompression)) {
         Err() << "error opening target partial file\n";
         exit(1);
      }
      return sequentialMerge(mergerP, start, filesPerProcess);
   };

   Bool_t status;

#ifndef R__WIN32
   if (multiprocess) {
      ROOT::TProcessExecutor p(nProcesses);
      auto res = p.Map(parallelMerge, ROOT::TSeqI(0, args.fSources.size(), filesPerProcess));
      status = std::accumulate(res.begin(), res.end(), 0U) == partialFiles.size();
      if (status) {
         for (const auto &pf : partialFiles) {
            fileMerger.AddFile(pf.c_str());
         }
         status = mergeFiles(fileMerger);
      } else {
         Err() << "failed at the parallel stage\n";
      }
   } else {
      status = sequentialMerge(fileMerger, 0, args.fSources.size());
   }
#else
   status = sequentialMerge(fileMerger, 0, args.fSources.size());
#endif

   if (status) {
      Info(3) << "merged " << args.fSources.size() << " (" << fileMerger.GetMergeList()->GetEntries()
              << ") input (partial) files into " << args.fTarget << "\n";
      return 0;
   } else {
      Err() << "failure during the merge of " << args.fSources.size() << " (" << fileMerger.GetMergeList()->GetEntries()
            << ") input (partial) files into " << args.fTarget << "\n";
      return 1;
   }
}
