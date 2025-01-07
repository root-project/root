/**
  \file hadd.cxx
  \brief This program will add histograms (see note) and Trees from a list of root files and write them to a target root file.
  The target file is newly created and must not be
  identical to one of the source files.

  Syntax:
  ```{.cpp}
       hadd [flags] targetfile source1 source2 ...
  ```

  Flags can be passed before or after the positional arguments.
  The first positional (non-flag) argument will be interpreted as the targetfile.
  After that, the first sequence of positional arguments will be interpreted as the input files.
  If two sequences of positional arguments are separated by flags, all sequences but the first
  will be ignored.
  If a flag requires an argument, the argument can be specified in any of these ways:

     # All equally valid:
     -j 16
     -j16
     -j=16
   
   The first syntax is the preferred one since it's backward-compatible with previous versions of hadd.
   The -f flag is an exception to this rule: it only supports the `-f[0-9]` syntax.

   Note that merging multiple flags is NOT supported: `-jfa` will be interpreted as -j=fa, which is invalid!

   The flags are as follows:

  \param -a   Append to the output
  \param -cachesize Resize the prefetching cache use to speed up I/O operations (use 0 to disable).
  \param -d   Carry out the partial multiprocess execution in the specified directory
  \param -dbg Enable verbosity. If -j was specified, do not not delete partial files stored inside working directory.
  \param -experimental-io-features `<feature>` Enables the corresponding experimental feature for output trees. \see ROOT::Experimental::EIOFeatures
  \param -f   Force overwriting of output file.
  \param -f[0-9] Set target compression level. 0 = uncompressed, 9 = highly compressed. Default is 1 (kDefaultZLIB).
                 You can also specify the full compresion algorithm, e.g. -f206
  \param -fk  Sets the target file to contain the baskets with the same compression
              as the input files (unless -O is specified). Compresses the meta data
              using the compression level specified in the first input or the
              compression setting after fk (for example 206 when using -fk206)
  \param -ff  The compression level used is the one specified in the first input
  \param -j   Parallelise the execution in `J` processes. If the number of processes is not specified, use the system maximum.
  \param -k   Skip corrupt or non-existent files, do not exit
  \param -n   Open at most `N` files at once (use 0 to request to use the system maximum)
  \param -O   Re-optimize basket size when merging TTree
  \param -T   Do not merge Trees
  \param -v   Explicitly set the verbosity level: 0 request no output, 99 is the default
  \return hadd returns a status code: 0 if OK, -1 otherwise

  For example assume 3 files f1, f2, f3 containing histograms hn and Trees Tn
   - f1 with h1 h2 h3 T1
   - f2 with h1 h4 T1 T2
   - f3 with h5
  the result of
  ```
    hadd -f x.root f1.root f2.root f3.root
  ```
  will be a file x.root with h1 h2 h3 h4 h5 T1 T2
  where
   - h1 will be the sum of the 2 histograms in f1 and f2
   - T1 will be the merge of the Trees in f1 and f2

  The files may contain sub-directories.

  If the source files contains histograms and Trees, one can skip
  the Trees with
  ```
       hadd -T targetfile source1 source2 ...
  ```

  Wildcarding and indirect files are also supported
  ```
      hadd result.root  myfil*.root
  ```
  will merge all files in myfil*.root
  ```
      hadd result.root file1.root @list.txt file2. root myfil*.root
  ```
  will merge file1.root, file2.root, all files in myfil*.root
  and all files in the indirect text file list.txt ("@" as the first
  character of the file indicates an indirect file. An indirect file
  is a text file containing a list of other files, including other
  indirect files, one line per file).

  If the sources and and target compression levels are identical (default),
  the program uses the TChain::Merge function with option "fast", ie
  the merge will be done without  unzipping or unstreaming the baskets
  (i.e. direct copy of the raw byte on disk). The "fast" mode is typically
  5 times faster than the mode unzipping and unstreaming the baskets.

  If the option -cachesize is used, hadd will resize (or disable if 0) the
  prefetching cache use to speed up I/O operations.

  For options that take a size as argument, a decimal number of bytes is expected.
  If the number ends with a `k`, `m`, `g`, etc., the number is multiplied
  by 1000 (1K), 1000000 (1MB), 1000000000 (1G), etc.
  If this prefix is followed by `i`, the number is multiplied by the traditional
  1024 (1KiB), 1048576 (1MiB), 1073741824 (1GiB), etc.
  The prefix can be optionally followed by B whose casing is ignored,
  eg. 1k, 1K, 1Kb and 1KB are the same.

  \note By default histograms are added. However hadd does not support the case where
         histograms have their bit TH1::kIsAverage set.

  \authors Rene Brun, Dirk Geppert, Sven A. Schmidt, Toby Burnett
*/
#include "Compression.h"
#include <ROOT/RConfig.hxx>
#include "ROOT/TIOFeatures.hxx"
#include "TFile.h"
#include "THashList.h"
#include "TKey.h"
#include "TClass.h"
#include "TSystem.h"
#include "TUUID.h"
#include "ROOT/StringConv.hxx"
#include "snprintf.h"

#include <string>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <climits>
#include <sstream>
#include <optional>
#include "haddCommandLineOptionsHelp.h"

#include "TFileMerger.h"
#ifndef R__WIN32
#include "ROOT/TProcessExecutor.hxx"
#endif

////////////////////////////////////////////////////////////////////////////////

struct HaddArgs {
  bool fNoTrees;
  bool fAppend;
  bool fForce;
  bool fSkipErrors; 
  bool fReoptimize;
  bool fDebug;
  bool fKeepCompressionAsIs;
  bool fUseFirstInputCompression;

  std::optional<std::string>       fWorkingDir;
  std::optional<int>               fNProcesses;
  std::optional<TString>           fCacheSize;
  std::optional<ROOT::TIOFeatures> fFeatures;
  std::optional<int>               fMaxOpenedFiles;
  std::optional<int>               fVerbosity;
  std::optional<int>               fCompressionSettings;

  int fOutputArgIdx;
  int fFirstInputIdx;
};

enum class EFlagResult {
   Ignored,
   Parsed,
   Error
};

static EFlagResult FlagToggle(const char *arg, const char *flagStr, bool &flagOut) {
   if (strncmp(arg, flagStr, strlen(flagStr)) == 0) {
      if (flagOut)
         std::cerr << "[warn] Duplicate flag: " << flagStr << "\n";
      flagOut = true;
   }
   return flagOut ? EFlagResult::Parsed : EFlagResult::Ignored;
}

// NOTE: not using std::stoi or similar because they have bad error checking.
// std::stoi will happily parse "120notvalid" as 120.
static std::optional<int> StrToInt(const char *str) {
   if (!str) return {};
   
   int res = 0;
   do {
      if (!isdigit(*str)) return {};
      res *= 10;
      res += *str - '0' ;
   } while (*++str);

   return res;
}

template <typename T>
static std::optional<T> ConvertArg(const char *);

template <>
std::optional<std::string> ConvertArg<std::string>(const char *arg) { return arg; }

template <>
std::optional<int> ConvertArg<int>(const char *arg) {  return StrToInt(arg); }

template <>
std::optional<ROOT::TIOFeatures> ConvertArg<ROOT::TIOFeatures>(const char *arg)
{
   ROOT::TIOFeatures features;
   std::stringstream ss;
   ss.str(arg);
   std::string item;
   while (std::getline(ss, item, ',')) {
      if (!features.Set(item))
         std::cerr << "Ignoring unknown feature request: " << item << std::endl;
   }
   return features;
}

static std::optional<int> ConvertNProcesses(const char *arg)
{
   auto np = ConvertArg<int>(arg);
   if (np) return np;
   else {
      std::cerr << "Error: could not parse the number of processes to run in parallel passed after -j: "
          << arg << ". We will use the system maximum.\n";
      // By returning a non-nullopt, we enable multiprocessing.
      // Since 0 is not a valid number of processes, hadd will default to the number of cpus.
      return 0;
   }
}

static std::optional<TString> ConvertCacheSize(const char *arg)
{
   TString cacheSize;
   int size;
   auto parseResult = ROOT::FromHumanReadableSize(arg, size);
   if (parseResult == ROOT::EFromHumanReadableSize::kParseFail) {
      std::cerr << "Error: could not parse the cache size passed after -cachesize: "
                << arg << ". We will use the default value.\n";
      return {};
   } else if (parseResult == ROOT::EFromHumanReadableSize::kOverflow) {
      double m;
      const char *munit = nullptr;
      ROOT::ToHumanReadableSize(INT_MAX, false, &m, &munit);
      std::cerr << "Error: the cache size passed after -cachesize is too large: "
                << arg << " is greater than " << m << munit
                << ". We will use the default value.\n";
      return {};
   } else {
      cacheSize = "cachesize=";
      cacheSize.Append(arg);
   }
   return cacheSize;
}

template <typename T>
static EFlagResult FlagArg(int argc, char **argv, int &argIdxInOut, const char *flagStr, 
                           std::optional<T> &flagOut,
                           std::optional<T> (*conv)(const char *) = ConvertArg<T>)
{
   int argIdx = argIdxInOut;
   const char *arg = argv[argIdx] + 1;
   int argLen = strlen(arg);
   int flagLen = strlen(flagStr);
   const char *nxtArg = nullptr;

   if (strncmp(arg, flagStr, flagLen) != 0)
      return EFlagResult::Ignored;
   
   if (argLen > flagLen) {
      // interpret anything after the flag as the argument.
      nxtArg = arg + flagLen;
      // Ignore one '=', if present
      if (nxtArg[0] && nxtArg[0] == '=')
         ++nxtArg;
   } else if (argLen == flagLen) {
      if (argIdx + 1 < argc) {
         nxtArg = argv[argIdx + 1];
      } else {
         std::cerr << "[err] Expected argument after '-" << flagStr << "' flag.\n";
         return EFlagResult::Error;
      }
   } else {
      return EFlagResult::Ignored;
   }

   flagOut = conv(nxtArg);
   ++argIdxInOut;
   return EFlagResult::Parsed;
}

static bool ValidCompressionSettings(int compSettings)
{
   for (int alg = 0; alg <= 5; ++alg) {
      for (int j=0; j<=9; ++j) {
         int comp = (alg*100)+j;
         if (compSettings == comp)
            return true;
      }
   }
   return false;
}

// The -f flag has a somewhat complicated logic.
// We have 4 cases:
//   1. -f
//   2. -ff
//   3. -fk
//   4. -f[0-509]
//
// and a combination thereof (e.g. -fk101, -ff202, -ffk, -fk209)
// -ff and -f[0-509] are incompatible.
//
// ALL these flags imply '-f' ("force overwrite"), but only if they parse successfully.
// This means that if we see a -f[something] and that "something" doesn't parse to a valid
// number between 0 and 509, or f or k, we consider the flag invalid and skip it without
// setting any state.
//
// Note that we don't allow `-f [0-9]` because that would be a backwards-incompatible
// change with the previous arg parsing semantic, changing the meaning of a cmdline like:
//
// $ hadd -f 200 f.root g.root  # <- '200' is the output file, not an argument to -f!
static EFlagResult FlagF(const char *arg, HaddArgs &args)
{
   if (arg[0] != 'f')
      return EFlagResult::Ignored;

   args.fForce = true;
   const char *cur = arg + 1;
   while (*cur) {
      switch (cur[0]) {
      case 'f':
         if (args.fUseFirstInputCompression)
            std::cerr << "[warn] Duplicate flag: -ff\n";
         if (args.fCompressionSettings) {
            std::cerr << "[err] Cannot specify both -ff and -f[0-9]. Either use the first input compression or specify it.\n";
            return EFlagResult::Error;
         } else
            args.fUseFirstInputCompression = true;
         break;
      case 'k':
         if (args.fKeepCompressionAsIs)
            std::cerr << "[warn] Duplicate flag: -fk\n";
         args.fKeepCompressionAsIs = true;
         break;
      default:
         if (isdigit(cur[0])) {
            if (args.fUseFirstInputCompression) {
               std::cerr << "[err] Cannot specify both -ff and -f[0-9]. Either use the first input compression or specify it.\n";
               return EFlagResult::Error;
            } else if (!args.fCompressionSettings) {
               if (auto compLv = StrToInt(cur)) {
                  if (ValidCompressionSettings(*compLv)) {
                     args.fCompressionSettings = *compLv;
                  } else {
                     std::cerr << "[err] " << *compLv << " is not a supported compression settings.\n";
                     return EFlagResult::Error;
                  }
               } else {
                  std::cerr << "[err] Failed to parse compression settings '" << cur << "' as an integer.\n";
                  return EFlagResult::Error;
               }
            }
         } else {
            std::cerr << "[err] Invalid flag: " << arg << "\n";
            return EFlagResult::Error;
         }
      }
      ++cur;
   }
   
   return EFlagResult::Parsed;
}

// Returns nullopt if any of the flags failed to parse.
// If an unknown flag is encountered, it will print a warning and go on.
static std::optional<HaddArgs> ParseArgs(int argc, char **argv)
{
   HaddArgs args {};

   for (int argIdx = 1; argIdx < argc; ++argIdx) {
      const char *argRaw = argv[argIdx];
      if (argRaw[0] == '-' && argRaw[1] != '\0') {
         // parse flag
         const char *arg = argRaw + 1;
         bool validFlag = false;

#define PARSE_FLAG(func, ...) do {                 \
      if (!validFlag) {                            \
         const auto res = func(__VA_ARGS__);       \
         if (res == EFlagResult::Error) return {}; \
         validFlag = res == EFlagResult::Parsed;   \
      }                                            \
   } while (0)

         PARSE_FLAG(FlagToggle, arg, "T", args.fNoTrees);
         PARSE_FLAG(FlagToggle, arg, "a", args.fAppend);
         PARSE_FLAG(FlagToggle, arg, "k", args.fSkipErrors);
         PARSE_FLAG(FlagToggle, arg, "O", args.fReoptimize);
         PARSE_FLAG(FlagToggle, arg, "dbg", args.fDebug);
         PARSE_FLAG(FlagArg, argc, argv, argIdx, "d", args.fWorkingDir);
         PARSE_FLAG(FlagArg, argc, argv, argIdx, "j", args.fNProcesses, ConvertNProcesses);
         PARSE_FLAG(FlagArg, argc, argv, argIdx, "cachesize", args.fCacheSize, ConvertCacheSize);
         PARSE_FLAG(FlagArg, argc, argv, argIdx, "experimental-io-features", args.fFeatures);
         PARSE_FLAG(FlagArg, argc, argv, argIdx, "n", args.fMaxOpenedFiles);
         PARSE_FLAG(FlagArg, argc, argv, argIdx, "v", args.fVerbosity);
         PARSE_FLAG(FlagF, arg, args);

#undef PARSE_FLAG

         if (!validFlag)
            std::cerr << "[warn] Unknown flag: " << argRaw << "\n";

      } else if (!args.fOutputArgIdx) {
         args.fOutputArgIdx = argIdx;
      } else if (!args.fFirstInputIdx) {
         args.fFirstInputIdx = argIdx;
      }
   }

   return args;
}

int main( int argc, char **argv )
{
   if ( argc < 3 || "-h" == std::string(argv[1]) || "--help" == std::string(argv[1]) ) {
         fprintf(stderr, kCommandLineOptionsHelp);
         return (argc == 2 && ("-h" == std::string(argv[1]) || "--help" == std::string(argv[1]))) ? 0 : 1;
   }

   const auto argsOpt = ParseArgs(argc, argv);
   if (!argsOpt)
      return 1;
   const HaddArgs &args = *argsOpt;

   ROOT::TIOFeatures features = args.fFeatures.value_or(ROOT::TIOFeatures{});
   Int_t maxopenedfiles = args.fMaxOpenedFiles.value_or(0);
   Int_t verbosity = args.fVerbosity.value_or(99);
   TString cacheSize = args.fCacheSize.value_or("");
   if (args.fCacheSize)
      std::cerr << "Using " << cacheSize << "\n";
   Bool_t multiproc = args.fNProcesses.has_value();
   int nProcesses;
   if (args.fNProcesses && *args.fNProcesses > 0) {
      nProcesses = *args.fNProcesses;
   } else {
      SysInfo_t s;
      gSystem->GetSysInfo(&s);
      nProcesses = s.fCpus;
   }
   if (multiproc)
      std::cout << "Parallelizing  with " << nProcesses << " processes.\n";
   std::string workingDir;
   if (!args.fWorkingDir) {
      workingDir = gSystem->TempDirectory();
   } else if (args.fWorkingDir && gSystem->AccessPathName(args.fWorkingDir->c_str())) {
      std::cerr << "Error: could not access the directory specified: " << *args.fWorkingDir
          << ". We will use the system's temporary directory.\n";
      workingDir = gSystem->TempDirectory();
   } else {
      workingDir = *args.fWorkingDir;
   }
   Int_t newcomp = args.fCompressionSettings.value_or(-1);
   
   gSystem->Load("libTreePlayer");

   const char *targetname = 0;
   if (!args.fOutputArgIdx) {
      std::cerr << "Missing output file.\n";
      return 1;
   }
   targetname = argv[args.fOutputArgIdx];

   if (verbosity > 1) {
      std::cout << "hadd Target file: " << targetname << std::endl;
   }

   TFileMerger fileMerger(kFALSE, kFALSE);
   fileMerger.SetMsgPrefix("hadd");
   fileMerger.SetPrintLevel(verbosity - 1);
   if (maxopenedfiles > 0) {
      fileMerger.SetMaxOpenedFiles(maxopenedfiles);
   }
   // The following section will collect all input filenames into a vector,
   // including those listed within an indirect file.
   // If any file can not be accessed, it will error out, unless args.fSkipErrors is true
   std::vector<std::string> allSubfiles;
   for (int a = args.fFirstInputIdx; a < argc; ++a) {
      if (argv[a] && argv[a][0] == '-') {
         break;
      }
      if (argv[a] && argv[a][0] == '@') {
         std::ifstream indirect_file(argv[a] + 1);
         if (!indirect_file.is_open()) {
            std::cerr << "hadd could not open indirect file " << (argv[a] + 1) << std::endl;
            if (!args.fSkipErrors)
               return 1;
         } else {
            std::string line;
            while (indirect_file) {
               if( std::getline(indirect_file, line) && line.length() ) {
                  if (gSystem->AccessPathName(line.c_str(), kReadPermission) == kTRUE) {
                     std::cerr << "hadd could not validate the file name \"" << line << "\" within indirect file "
                               << (argv[a] + 1) << std::endl;
                     if (!args.fSkipErrors)
                        return 1;
                  } else
                     allSubfiles.emplace_back(line);
               }
            }
         }
      } else {
         const std::string line = argv[a];
         if (gSystem->AccessPathName(line.c_str(), kReadPermission) == kTRUE) {
            std::cerr << "hadd could not validate argument \"" << line << "\" as input file " << std::endl;
            if (!args.fSkipErrors)
               return 1;
         } else
            allSubfiles.emplace_back(line);
      }
   }
   if (allSubfiles.empty()) {
      std::cerr << "hadd could not find any valid input file " << std::endl;
      return 1;
   }
   // The next snippet determines the output compression if unset
   if (newcomp == -1) {
      if (args.fUseFirstInputCompression || args.fKeepCompressionAsIs) {
         // grab from the first file.
         TFile *firstInput = TFile::Open(allSubfiles.front().c_str());
         if (firstInput && !firstInput->IsZombie())
            newcomp = firstInput->GetCompressionSettings();
         else
            newcomp = ROOT::RCompressionSetting::EDefaults::kUseCompiledDefault;
         delete firstInput;
         fileMerger.SetMergeOptions(TString("first_source_compression"));
      } else {
         newcomp = ROOT::RCompressionSetting::EDefaults::kUseCompiledDefault;
         fileMerger.SetMergeOptions(TString("default_compression"));
      }
   }
   if (verbosity > 1) {
      if (args.fKeepCompressionAsIs && !args.fReoptimize)
         std::cout << "hadd compression setting for meta data: " << newcomp << '\n';
      else
         std::cout << "hadd compression setting for all output: " << newcomp << '\n';
   }
   if (args.fAppend) {
      if (!fileMerger.OutputFile(targetname, "UPDATE", newcomp)) {
         std::cerr << "hadd error opening target file for update :" << targetname << "." << std::endl;
         exit(2);
      }
   } else if (!fileMerger.OutputFile(targetname, args.fForce, newcomp)) {
      std::cerr << "hadd error opening target file (does " << targetname << " exist?)." << std::endl;
      if (!args.fForce) std::cerr << "Pass \"-f\" argument to force re-creation of output file." << std::endl;
      exit(1);
   }

   auto step = (allSubfiles.size() + nProcesses - 1) / nProcesses;
   if (multiproc && step < 3) {
      // At least 3 files per process
      step = 3;
      nProcesses = (allSubfiles.size() + step - 1) / step;
      std::cout << "Each process should handle at least 3 files for efficiency.";
      std::cout << " Setting the number of processes to: " << nProcesses << std::endl;
   }
   if (nProcesses == 1)
      multiproc = kFALSE;

   std::vector<std::string> partialFiles;

#ifndef R__WIN32
   // this is commented out only to try to prevent false positive detection
   // from several anti-virus engines on Windows, and multiproc is not
   // supported on Windows anyway
   if (multiproc) {
      auto uuid = TUUID();
      auto partialTail = uuid.AsString();
      for (auto i = 0; (i * step) < allSubfiles.size(); i++) {
         std::stringstream buffer;
         buffer << workingDir << "/partial" << i << "_" << partialTail << ".root";
         partialFiles.emplace_back(buffer.str());
      }
   }
#endif

   auto mergeFiles = [&](TFileMerger &merger) {
      if (args.fReoptimize) {
         merger.SetFastMethod(kFALSE);
      } else {
         if (!args.fKeepCompressionAsIs && merger.HasCompressionChange()) {
            // Don't warn if the user has requested any re-optimization.
            Warn() << "Sources and Target have different compression settings\n"
               "hadd merging will be slower\n";
         }
      }
      merger.SetNotrees(args.fNoTrees);
      merger.SetMergeOptions(TString(merger.GetMergeOptions()) + " " + cacheSize);
      merger.SetIOFeatures(features);
      Bool_t status;
      if (args.fAppend)
         status = merger.PartialMerge(TFileMerger::kIncremental | TFileMerger::kAll);
      else
         status = merger.Merge();
      return status;
   };

   auto sequentialMerge = [&](TFileMerger &merger, int start, int nFiles) {
      for (auto i = start; i < (start + nFiles) && i < static_cast<int>(allSubfiles.size()); i++) {
         if (!merger.AddFile(allSubfiles[i].c_str())) {
            if (args.fSkipErrors) {
               std::cerr << "hadd skipping file with error: " << allSubfiles[i] << std::endl;
            } else {
               std::cerr << "hadd exiting due to error in " << allSubfiles[i] << std::endl;
               return kFALSE;
            }
         }
      }
      return mergeFiles(merger);
   };

   auto parallelMerge = [&](int start) {
      TFileMerger mergerP(kFALSE, kFALSE);
      mergerP.SetMsgPrefix("hadd");
      mergerP.SetPrintLevel(verbosity - 1);
      if (maxopenedfiles > 0) {
         mergerP.SetMaxOpenedFiles(maxopenedfiles / nProcesses);
      }
      if (!mergerP.OutputFile(partialFiles[start / step].c_str(), newcomp)) {
         std::cerr << "hadd error opening target partial file" << std::endl;
         exit(1);
      }
      return sequentialMerge(mergerP, start, step);
   };

   auto reductionFunc = [&]() {
      for (const auto &pf : partialFiles) {
         fileMerger.AddFile(pf.c_str());
      }
      return mergeFiles(fileMerger);
   };

   Bool_t status;

#ifndef R__WIN32
   if (multiproc) {
      ROOT::TProcessExecutor p(nProcesses);
      auto res = p.Map(parallelMerge, ROOT::TSeqI(0, allSubfiles.size(), step));
      status = std::accumulate(res.begin(), res.end(), 0U) == partialFiles.size();
      if (status) {
         status = reductionFunc();
      } else {
         std::cout << "hadd failed at the parallel stage" << std::endl;
      }
      if (!args.fDebug) {
         for (const auto &pf : partialFiles) {
            gSystem->Unlink(pf.c_str());
         }
      }
   } else {
      status = sequentialMerge(fileMerger, 0, allSubfiles.size());
   }
#else
   status = sequentialMerge(fileMerger, 0, allSubfiles.size());
#endif

   if (status) {
      if (verbosity == 1) {
         std::cout << "hadd merged " << allSubfiles.size() << " (" << fileMerger.GetMergeList()->GetEntries()
                   << ") input (partial) files into " << targetname << ".\n";
      }
      return 0;
   } else {
      if (verbosity == 1) {
         std::cout << "hadd failure during the merge of " << allSubfiles.size() << " ("
                   << fileMerger.GetMergeList()->GetEntries() << ") input (partial) files into " << targetname << ".\n";
      }
      return 1;
   }
}
