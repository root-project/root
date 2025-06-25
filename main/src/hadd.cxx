/**
  \file hadd.cxx
  \brief This program will merge compatible ROOT objects, such as histograms, Trees and RNTuples,
         from a list of root files and write them to a target root file.
         In order for a ROOT object to be mergeable, it must implement the Merge() function.
         Non-mergeable objects will have all instances copied as-is into the target file.
         The target file must not be identical to one of the source files.

  Syntax:
  ```{.cpp}
       hadd [flags] targetfile source1 source2 ... [flags]
  ```

  Flags can be passed before or after the positional arguments.
  The first positional (non-flag) argument will be interpreted as the targetfile.
  After that, the first sequence of positional arguments will be interpreted as the input files.
  If two sequences of positional arguments are separated by flags, hadd will emit an error and abort.

  By default, any argument starting with `-` is interpreted as a flag. If you want to pass filenames
  starting with `-` you need to pass them after `--`:
  ```{.cpp}
       hadd [flags] -- -file1 -file2 ...
  ```
  Note that in this case you need to pass ALL positional arguments after `--`.

  If a flag requires an argument, the argument can be specified in any of these ways:

     # All equally valid:
     -j 16
     -j16
     -j=16

   The first syntax is the preferred one since it's backward-compatible with previous versions of hadd.
   The -f flag is an exception to this rule: it only supports the `-f[0-9]` syntax.

   Note that merging multiple flags is NOT supported: `-jfa` will be interpreted as -j=fa, which is invalid!

   The flags are as follows:

  \param -a                Append to the output
  \param -cachesize <SIZE> Resize the prefetching cache used to speed up I/O operations (use 0 to disable).
  \param -d <DIR>          Carry out the partial multiprocess execution in the specified directory
  \param -dbg              Enable verbosity. If -j was specified, do not not delete partial files
                           stored inside working directory.
  \param -experimental-io-features <FEATURES> Enables the corresponding experimental feature for output trees.
                                              \see ROOT::Experimental::EIOFeatures
  \param -f            Force overwriting of output file.
  \param -f[0-9]       Set target compression algorithm `i` and level `j` passing the number `i*100 + j`, e.g. `-f505`.
                       The last digit (`j`) can be set from 0 = uncompressed to 9 = highly compressed.
                       The first digit (`i`) is 1 for ZLIB, 2 for LZMA, 4 for LZ4 and 5 for ZSTD.
                       Recommended numbers are 101 (ZLIB), 207 (LZMA), 404 (LZ4), 505 (ZSTD),
                       The default value for this flag is 101 (kDefaultZLIB).
                       See ROOT::RCompressionSetting and TFile::TFile documentation for more details.
  \param -fk           Sets the target file to contain the baskets with the same compression as the input files
                       (unless -O is specified). Compresses the meta data using the compression level specified
                       in the first input or the compression setting after fk (for example 505 when using -fk505)
  \param -ff           The compression level used is the one specified in the first input
  \param -j [N_JOBS]   Parallelise the execution in `N_JOBS` processes. If the number of processes is not specified,
                       or is 0, use the system maximum.
  \param -k            Skip corrupt or non-existent files, do not exit
  \param -L <FILE>     Read the list of objects from FILE and either only merge or skip those objects depending on
                       the value of "-Ltype". FILE must contain one object name per line, which cannot contain
                       whitespaces or '/'. You can also pass TDirectory names, which apply to the entire directory
                       content. Lines beginning with '#' are ignored. If this flag is passed, "-Ltype" MUST be
                       passed as well.
  \param -Ltype <SkipListed|OnlyListed> Sets the type of operation performed on the objects listed in FILE given with the
                       "-L" flag. "SkipListed" will skip all the listed objects; "OnlyListed" will only merge those
                       objects. If this flag is passed, "-L" must be passed as well.
  \param -n <N_FILES>  Open at most `N` files at once (use 0 to request to use the system maximum - which is also
                       the default). This number includes both the input reading files as well as the output file.
                       Thus, if set to 1, it will be automatically replaced to a minimum of 2. If set to a too large value,
                       it will be clipped to the system maximum.
  \param -O            Re-optimize basket size when merging TTree
  \param -T            Do not merge Trees
  \param -v [LEVEL]    Explicitly set the verbosity level: 0 request no output, 99 is the default
  \return hadd returns a status code: 0 if OK, 1 otherwise

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

#include "haddCommandLineOptionsHelp.h"

#include <climits>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>

#ifndef R__WIN32
#include "ROOT/TProcessExecutor.hxx"
#endif

////////////////////////////////////////////////////////////////////////////////

inline std::ostream &Err()
{
   std::cerr << "Error in <hadd>: ";
   return std::cerr;
}

inline std::ostream &Warn()
{
   std::cerr << "Warning in <hadd>: ";
   return std::cerr;
}

inline std::ostream &Info()
{
   std::cerr << "Info in <hadd>: ";
   return std::cerr;
}

using IntFlag_t = uint32_t;

struct HAddArgs {
   bool fNoTrees;
   bool fAppend;
   bool fForce;
   bool fSkipErrors;
   bool fReoptimize;
   bool fDebug;
   bool fKeepCompressionAsIs;
   bool fUseFirstInputCompression;

   std::optional<std::string> fWorkingDir;
   std::optional<IntFlag_t> fNProcesses;
   std::optional<std::string> fObjectFilterFile;
   std::optional<Int_t> fObjectFilterType;
   std::optional<TString> fCacheSize;
   std::optional<ROOT::TIOFeatures> fFeatures;
   std::optional<IntFlag_t> fMaxOpenedFiles;
   std::optional<IntFlag_t> fVerbosity;
   std::optional<IntFlag_t> fCompressionSettings;

   int fOutputArgIdx;
   int fFirstInputIdx;
   // This is set to true if and only if the user passed `--`. In this special
   // case, we must not stop parsing positional arguments even if we find one
   // that starts with a `-`.
   bool fNoFlagsAfterPositionalArguments;
};

enum class EFlagResult { kIgnored, kParsed, kErr };

static EFlagResult FlagToggle(const char *arg, const char *flagStr, bool &flagOut)
{
   const auto argLen = strlen(arg);
   const auto flagLen = strlen(flagStr);
   if (argLen == flagLen && strncmp(arg, flagStr, flagLen) == 0) {
      if (flagOut)
         Warn() << "duplicate flag: " << flagStr << "\n";
      flagOut = true;
      return EFlagResult::kParsed;
   }
   return EFlagResult::kIgnored;
}

// NOTE: not using std::stoi or similar because they have bad error checking.
// std::stoi will happily parse "120notvalid" as 120.
static std::optional<IntFlag_t> StrToUInt(const char *str)
{
   if (!str)
      return {};

   uint32_t res = 0;
   do {
      if (!isdigit(*str))
         return {};
      if (res * 10 < res) // overflow is an error
         return {};
      res *= 10;
      res += *str - '0';
   } while (*++str);

   return res;
}

template <typename T>
struct FlagConvResult {
   T fValue;
   EFlagResult fResult;
};

template <typename T>
static FlagConvResult<T> ConvertArg(const char *);

template <>
FlagConvResult<std::string> ConvertArg<std::string>(const char *arg)
{
   return {arg, EFlagResult::kParsed};
}

template <>
FlagConvResult<IntFlag_t> ConvertArg<IntFlag_t>(const char *arg)
{
   // Don't even try to parse arg if it doesn't look like a number.
   if (!isdigit(*arg))
      return {0, EFlagResult::kIgnored};

   auto intOpt = StrToUInt(arg);
   if (intOpt)
      return {*intOpt, EFlagResult::kParsed};

   Err() << "error parsing integer argument '" << arg << "'\n";
   return {0, EFlagResult::kErr};
}

template <>
FlagConvResult<ROOT::TIOFeatures> ConvertArg<ROOT::TIOFeatures>(const char *arg)
{
   ROOT::TIOFeatures features;
   std::stringstream ss;
   ss.str(arg);
   std::string item;
   while (std::getline(ss, item, ',')) {
      if (!features.Set(item))
         Warn() << "ignoring unknown feature request: " << item << "\n";
   }
   return {features, EFlagResult::kParsed};
}

static FlagConvResult<TString> ConvertCacheSize(const char *arg)
{
   TString cacheSize;
   int size;
   auto parseResult = ROOT::FromHumanReadableSize(arg, size);
   if (parseResult == ROOT::EFromHumanReadableSize::kParseFail) {
      Err() << "could not parse the cache size passed after -cachesize: '" << arg << "'\n";
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
      cacheSize.Append(arg);
   }
   return {cacheSize, EFlagResult::kParsed};
}

static FlagConvResult<Int_t> ConvertFilterType(const char *arg)
{
   if (strcmp(arg, "SkipListed") == 0)
      return {TFileMerger::kSkipListed, EFlagResult::kParsed};
   if (strcmp(arg, "OnlyListed") == 0)
      return {TFileMerger::kOnlyListed, EFlagResult::kParsed};

   Err() << "invalid argument for -Ltype: '" << arg << "'. Can only be 'SkipListed' or 'OnlyListed' (case matters).\n";
   return {{}, EFlagResult::kErr};
}

// Parses a flag that is followed by an argument of type T.
// If `defaultVal` is provided, the following argument is optional and will be set to `defaultVal` if missing.
// `conv` is used to convert the argument from string to its type T.
template <typename T>
static EFlagResult
FlagArg(int argc, char **argv, int &argIdxInOut, const char *flagStr, std::optional<T> &flagOut,
        std::optional<T> defaultVal = std::nullopt, FlagConvResult<T> (*conv)(const char *) = ConvertArg<T>)
{
   int argIdx = argIdxInOut;
   const char *arg = argv[argIdx] + 1;
   int argLen = strlen(arg);
   int flagLen = strlen(flagStr);
   const char *nxtArg = nullptr;

   if (strncmp(arg, flagStr, flagLen) != 0)
      return EFlagResult::kIgnored;

   bool argIsSeparate = false;
   if (argLen > flagLen) {
      // interpret anything after the flag as the argument.
      nxtArg = arg + flagLen;
      // Ignore one '=', if present
      if (nxtArg[0] == '=')
         ++nxtArg;
   } else if (argLen == flagLen) {
      argIsSeparate = true;
      if (argIdx + 1 < argc) {
         ++argIdxInOut;
         nxtArg = argv[argIdxInOut];
      } else {
         Err() << "expected argument after '-" << flagStr << "' flag.\n";
         return EFlagResult::kErr;
      }
   } else {
      return EFlagResult::kIgnored;
   }

   auto converted = conv(nxtArg);
   if (converted.fResult == EFlagResult::kParsed) {
      flagOut = converted.fValue;
   } else if (converted.fResult == EFlagResult::kIgnored) {
      if (defaultVal && argIsSeparate) {
         flagOut = defaultVal;
         // If we had tried parsing the next argument, step back one arg idx.
         argIdxInOut -= (argIdxInOut > argIdx);
      } else {
         Err() << "the argument after '-" << flagStr << "' flag was not of the expected type.\n";
         return EFlagResult::kErr;
      }
   } else {
      return EFlagResult::kErr;
   }

   return EFlagResult::kParsed;
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
static EFlagResult FlagF(const char *arg, HAddArgs &args)
{
   if (arg[0] != 'f')
      return EFlagResult::kIgnored;

   args.fForce = true;
   const char *cur = arg + 1;
   while (*cur) {
      switch (cur[0]) {
      case 'f':
         if (args.fUseFirstInputCompression)
            Warn() << "duplicate flag: -ff\n";
         if (args.fCompressionSettings) {
            std::cerr
               << "[err] Cannot specify both -ff and -f[0-9]. Either use the first input compression or specify it.\n";
            return EFlagResult::kErr;
         } else
            args.fUseFirstInputCompression = true;
         break;
      case 'k':
         if (args.fKeepCompressionAsIs)
            Warn() << "duplicate flag: -fk\n";
         args.fKeepCompressionAsIs = true;
         break;
      default:
         if (isdigit(cur[0])) {
            if (args.fUseFirstInputCompression) {
               Err() << "cannot specify both -ff and -f[0-9]. Either use the first input compression or "
                        "specify it.\n";
               return EFlagResult::kErr;
            } else if (!args.fCompressionSettings) {
               if (auto compLv = StrToUInt(cur)) {
                  if (ValidCompressionSettings(*compLv)) {
                     args.fCompressionSettings = *compLv;
                     // we can't see any other argument after the number, so we return here to avoid
                     // incorrectly parsing the rest of the characters in `arg`.
                     return EFlagResult::kParsed;
                  } else {
                     Err() << *compLv << " is not a supported compression settings.\n";
                     return EFlagResult::kErr;
                  }
               } else {
                  Err() << "failed to parse compression settings '" << cur << "' as an integer.\n";
                  return EFlagResult::kErr;
               }
            } else {
               Err() << "cannot specify -f[0-9] multiple times!\n";
               return EFlagResult::kErr;
            }
         } else {
            Err() << "invalid flag: " << arg << "\n";
            return EFlagResult::kErr;
         }
      }
      ++cur;
   }

   return EFlagResult::kParsed;
}

// Returns nullopt if any of the flags failed to parse.
// If an unknown flag is encountered, it will print a warning and go on.
static std::optional<HAddArgs> ParseArgs(int argc, char **argv)
{
   HAddArgs args{};

   enum {
      kParseStart,
      kParseFirstFlagGroup,
      kParseFirstPosArgGroup,
      kParseSecondFlagGroup,
   } parseState = kParseStart;

   for (int argIdx = 1; argIdx < argc; ++argIdx) {
      const char *argRaw = argv[argIdx];
      if (!*argRaw)
         continue;

      if (!args.fNoFlagsAfterPositionalArguments && argRaw[0] == '-' && argRaw[1] != '\0') {
         if (argRaw[1] == '-' && argRaw[2] == '\0') {
            // special case `--`: force parsing to consider all future args as positional arguments.
            if (parseState > kParseFirstFlagGroup) {
               Err()
                  << "found `--`, but we've already parsed (or are still parsing) a sequence of positional arguments!"
                     " This is not supported: you must have exactly one sequence of positional arguments, so if you"
                     " need to use `--` make sure to pass *all* positional arguments after it.";
               return {};
            }
            args.fNoFlagsAfterPositionalArguments = true;
            continue;
         }

         // parse flag
         parseState = (parseState == kParseFirstPosArgGroup) ? kParseSecondFlagGroup : kParseFirstFlagGroup;

         const char *arg = argRaw + 1;
         bool validFlag = false;

#define PARSE_FLAG(func, ...)                     \
   do {                                           \
      if (!validFlag) {                           \
         const auto res = func(__VA_ARGS__);      \
         if (res == EFlagResult::kErr)            \
            return {};                            \
         validFlag = res == EFlagResult::kParsed; \
      }                                           \
   } while (0)

         // NOTE: if two flags have the same prefix (e.g. -Ltype and -L) always put the longest one first!
         PARSE_FLAG(FlagToggle, arg, "T", args.fNoTrees);
         PARSE_FLAG(FlagToggle, arg, "a", args.fAppend);
         PARSE_FLAG(FlagToggle, arg, "k", args.fSkipErrors);
         PARSE_FLAG(FlagToggle, arg, "O", args.fReoptimize);
         PARSE_FLAG(FlagToggle, arg, "dbg", args.fDebug);
         PARSE_FLAG(FlagArg, argc, argv, argIdx, "d", args.fWorkingDir);
         PARSE_FLAG(FlagArg, argc, argv, argIdx, "j", args.fNProcesses, {0});
         PARSE_FLAG(FlagArg, argc, argv, argIdx, "Ltype", args.fObjectFilterType, {}, ConvertFilterType);
         PARSE_FLAG(FlagArg, argc, argv, argIdx, "L", args.fObjectFilterFile);
         PARSE_FLAG(FlagArg, argc, argv, argIdx, "cachesize", args.fCacheSize, {}, ConvertCacheSize);
         PARSE_FLAG(FlagArg, argc, argv, argIdx, "experimental-io-features", args.fFeatures);
         PARSE_FLAG(FlagArg, argc, argv, argIdx, "n", args.fMaxOpenedFiles);
         PARSE_FLAG(FlagArg, argc, argv, argIdx, "v", args.fVerbosity, {99});
         PARSE_FLAG(FlagF, arg, args);

#undef PARSE_FLAG

         if (!validFlag)
            Warn() << "unknown flag: " << argRaw << "\n";

      } else if (!args.fOutputArgIdx) {
         // First positional argument is the output
         args.fOutputArgIdx = argIdx;
         assert(parseState < kParseFirstPosArgGroup);
         parseState = kParseFirstPosArgGroup;
      } else {
         // We should be in the same positional argument group as the output, error otherwise
         if (parseState == kParseFirstPosArgGroup) {
            if (!args.fFirstInputIdx) {
               args.fFirstInputIdx = argIdx;
            }
         } else {
            Err() << "seen a positional argument '" << argRaw
                  << "' after some flags."
                     " Positional arguments were already parsed at this point (from '"
                  << argv[args.fOutputArgIdx]
                  << "' onwards), and you can only have one sequence of them, so you cannot pass more."
                     " Please group your positional arguments all together so that hadd works as you expect.\n"
                     "Cmdline: ";
            for (int i = 0; i < argc; ++i)
               std::cerr << argv[i] << " ";
            std::cerr << "\n";

            return {};
         }
      }
   }

   return args;
}

// Returns the flags to add to the file merger's flags, or -1 in case of errors.
static Int_t ParseFilterFile(const std::optional<std::string> &filterFileName,
                             std::optional<Int_t> objectFilterType, TFileMerger &fileMerger)
{
   if (filterFileName) {
      std::ifstream filterFile(*filterFileName);
      if (!filterFile) {
         Err() << "error opening filter file '" << *filterFileName << "'\n";
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
         Info() << "added " << nObjects << " object from filter file '" << *filterFileName << "'\n";
         fileMerger.AddObjectNames(filteredObjects);
      } else {
         Warn() << "no objects were added from filter file '" << *filterFileName << "'\n";
      }

      assert(objectFilterType.has_value());
      const auto filterFlag = *objectFilterType;
      assert(filterFlag == TFileMerger::kSkipListed || filterFlag == TFileMerger::kOnlyListed);
      return filterFlag;
   }
   return 0;
}

int main(int argc, char **argv)
{
   if (argc < 3 || "-h" == std::string(argv[1]) || "--help" == std::string(argv[1])) {
      fprintf(stderr, kCommandLineOptionsHelp);
      return (argc == 2 && ("-h" == std::string(argv[1]) || "--help" == std::string(argv[1]))) ? 0 : 1;
   }

   const auto argsOpt = ParseArgs(argc, argv);
   if (!argsOpt)
      return 1;
   const HAddArgs &args = *argsOpt;

   ROOT::TIOFeatures features = args.fFeatures.value_or(ROOT::TIOFeatures{});
   Int_t maxopenedfiles = args.fMaxOpenedFiles.value_or(0);
   Int_t verbosity = args.fVerbosity.value_or(99);
   Int_t newcomp = args.fCompressionSettings.value_or(-1);
   TString cacheSize = args.fCacheSize.value_or("");

   // For the -j flag (nProcesses), we check if the flag is present and, if so, if it has a
   // valid value (i.e. any value > 0).
   // If the flag is present at all, we do multiprocessing. If the value of nProcesses is invalid,
   // we default to the number of cpus on the machine.
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
      Info() << "parallelizing  with " << nProcesses << " processes.\n";

   // If the user specified a workingDir, use that. Otherwise, default to the system temp dir.
   std::string workingDir;
   if (!args.fWorkingDir) {
      workingDir = gSystem->TempDirectory();
   } else if (args.fWorkingDir && gSystem->AccessPathName(args.fWorkingDir->c_str())) {
      Err() << "could not access the directory specified: " << *args.fWorkingDir << ".\n";
      return 1;
   } else {
      workingDir = *args.fWorkingDir;
   }

   // Verify that -L and -Ltype are either both present or both absent.
   if (args.fObjectFilterFile.has_value() != args.fObjectFilterType.has_value()) {
      Err() << "-L must always be passed along with -Ltype.\n";
      return 1;
   }

   const char *targetname = 0;
   if (!args.fOutputArgIdx) {
      Err() << "missing output file.\n";
      return 1;
   }
   if (!args.fFirstInputIdx) {
      Err() << "missing input file.\n";
      return 1;
   }
   targetname = argv[args.fOutputArgIdx];

   if (verbosity > 1)
      Info() << "target file: " << targetname << "\n";

   if (args.fCacheSize)
      Info() << "Using " << cacheSize << "\n";

   ////////////////////////////// end flags processing /////////////////////////////////

   gSystem->Load("libTreePlayer");

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
      if (!args.fNoFlagsAfterPositionalArguments && argv[a] && argv[a][0] == '-') {
         break;
      }
      if (argv[a] && argv[a][0] == '@') {
         std::ifstream indirect_file(argv[a] + 1);
         if (!indirect_file.is_open()) {
            Err() << "could not open indirect file " << (argv[a] + 1) << std::endl;
            if (!args.fSkipErrors)
               return 1;
         } else {
            std::string line;
            while (indirect_file) {
               if (std::getline(indirect_file, line) && line.length()) {
                  if (gSystem->AccessPathName(line.c_str(), kReadPermission) == kTRUE) {
                     Err() << "could not validate the file name \"" << line << "\" within indirect file "
                           << (argv[a] + 1) << std::endl;
                     if (!args.fSkipErrors)
                        return 1;
                  } else if (std::filesystem::exists(targetname) && std::filesystem::equivalent(line, targetname)) {
                     Err() << "file " << line << " cannot be both the target and an input!\n";
                     if (!args.fSkipErrors)
                        return 1;
                  } else {
                     allSubfiles.emplace_back(line);
                  }
               }
            }
         }
      } else {
         const std::string line = argv[a];
         if (gSystem->AccessPathName(line.c_str(), kReadPermission) == kTRUE) {
            Err() << "could not validate argument \"" << line << "\" as input file " << std::endl;
            if (!args.fSkipErrors)
               return 1;
         } else if (std::filesystem::exists(targetname) && std::filesystem::equivalent(line, targetname)) {
            Err() << "file " << line << " cannot be both the target and an input!\n";
            if (!args.fSkipErrors)
               return 1;
         } else {
            allSubfiles.emplace_back(line);
         }
      }
   }
   if (allSubfiles.empty()) {
      Err() << "could not find any valid input file " << std::endl;
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
         fileMerger.SetMergeOptions(TString("FirstSrcCompression"));
      } else {
         newcomp = ROOT::RCompressionSetting::EDefaults::kUseCompiledDefault;
         fileMerger.SetMergeOptions(TString("DefaultCompression"));
      }
   }
   if (verbosity > 1) {
      if (args.fKeepCompressionAsIs && !args.fReoptimize)
         Info() << "compression setting for meta data: " << newcomp << '\n';
      else
         Info() << "compression setting for all output: " << newcomp << '\n';
   }
   if (args.fAppend) {
      if (!fileMerger.OutputFile(targetname, "UPDATE", newcomp)) {
         Err() << "error opening target file for update :" << targetname << ".\n";
         return 2;
      }
   } else if (!fileMerger.OutputFile(targetname, args.fForce, newcomp)) {
      Err() << "error opening target file (does " << targetname << " exist?).\n";
      if (!args.fForce)
         Info() << "pass \"-f\" argument to force re-creation of output file.\n";
      return 1;
   }

   auto step = (allSubfiles.size() + nProcesses - 1) / nProcesses;
   if (multiproc && step < 3) {
      // At least 3 files per process
      step = 3;
      nProcesses = (allSubfiles.size() + step - 1) / step;
      Info() << "each process should handle at least 3 files for efficiency."
                " Setting the number of processes to: "
             << nProcesses << std::endl;
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
      merger.SetErrorBehavior(args.fSkipErrors ? TFileMerger::EErrorBehavior::kSkipOnError
                                               : TFileMerger::EErrorBehavior::kFailOnError);
      merger.SetIOFeatures(features);
      Int_t fileMergerFlags = TFileMerger::kAll;
      Int_t extraFlags = ParseFilterFile(args.fObjectFilterFile, args.fObjectFilterType, merger);
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

   auto sequentialMerge = [&](TFileMerger &merger, int start, int nFiles) {
      for (auto i = start; i < (start + nFiles) && i < static_cast<int>(allSubfiles.size()); i++) {
         if (!merger.AddFile(allSubfiles[i].c_str())) {
            if (args.fSkipErrors) {
               Warn() << "skipping file with error: " << allSubfiles[i] << std::endl;
            } else {
               Err() << "exiting due to error in " << allSubfiles[i] << std::endl;
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
         Err() << "error opening target partial file\n";
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
         Err() << "failed at the parallel stage\n";
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
         Info() << "merged " << allSubfiles.size() << " (" << fileMerger.GetMergeList()->GetEntries()
                << ") input (partial) files into " << targetname << ".\n";
      }
      return 0;
   } else {
      if (verbosity == 1) {
         Err() << "failure during the merge of " << allSubfiles.size() << " ("
               << fileMerger.GetMergeList()->GetEntries() << ") input (partial) files into " << targetname << ".\n";
      }
      return 1;
   }
}
