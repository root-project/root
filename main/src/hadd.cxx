/*

  This program will add histograms (see note) and Trees from a list of root files and write them
  to a target root file. The target file is newly created and must not be
  identical to one of the source files.

  Syntax:

       hadd targetfile source1 source2 ...
    or
       hadd -f targetfile source1 source2 ...
         (targetfile is overwritten if it exists)

  When the -f option is specified, one can also specify the compression
  level of the target file. By default the compression level is 1, but
  if "-f0" is specified, the target file will not be compressed.
  if "-f6" is specified, the compression level 6 will be used.

  For example assume 3 files f1, f2, f3 containing histograms hn and Trees Tn
    f1 with h1 h2 h3 T1
    f2 with h1 h4 T1 T2
    f3 with h5
   the result of
     hadd -f x.root f1.root f2.root f3.root
   will be a file x.root with h1 h2 h3 h4 h5 T1 T2
   where h1 will be the sum of the 2 histograms in f1 and f2
         T1 will be the merge of the Trees in f1 and f2

  The files may contain sub-directories.

  if the source files contains histograms and Trees, one can skip
  the Trees with
       hadd -T targetfile source1 source2 ...

  Wildcarding and indirect files are also supported
    hadd result.root  myfil*.root
   will merge all files in myfil*.root
    hadd result.root file1.root @list.txt file2. root myfil*.root
    will merge file1. root, file2. root, all files in myfil*.root
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

  For options that takes a size as argument, a decimal number of bytes is expected.
  If the number ends with a ``k'', ``m'', ``g'', etc., the number is multiplied
  by 1000 (1K), 1000000 (1MB), 1000000000 (1G), etc.
  If this prefix is followed by i, the number is multipled by the traditional
  1024 (1KiB), 1048576 (1MiB), 1073741824 (1GiB), etc.
  The prefix can be optionally followed by B whose casing is ignored,
  eg. 1k, 1K, 1Kb and 1KB are the same.

  NOTE1: By default histograms are added. However hadd does not support the case where
         histograms have their bit TH1::kIsAverage set.

  NOTE2: hadd returns a status code: 0 if OK, -1 otherwise

  Authors: Rene Brun, Dirk Geppert, Sven A. Schmidt, sven.schmidt@cern.ch
         : rewritten from scratch by Rene Brun (30 November 2005)
            to support files with nested directories.
           Toby Burnett implemented the possibility to use indirect files.
 */

#include "RConfig.h"
#include <string>
#include "TFile.h"
#include "THashList.h"
#include "TKey.h"
#include "TObjString.h"
#include "Riostream.h"
#include "TClass.h"
#include "TSystem.h"
#include "TUUID.h"
#include "ROOT/StringConv.hxx"
#include <stdlib.h>
#include <climits>

#include "TFileMerger.h"
#include "ROOT/TProcessExecutor.hxx"

////////////////////////////////////////////////////////////////////////////////

int main( int argc, char **argv )
{
   if ( argc < 3 || "-h" == std::string(argv[1]) || "--help" == std::string(argv[1]) ) {
      std::cout << "Usage: " << argv[0] << " [-f[fk][0-9]] [-k] [-T] [-O] [-a] \n"
      "            [-n maxopenedfiles] [-cachesize size] [-v [verbosity]] \n"
      "            targetfile source1 [source2 source3 ...]\n" << std::endl;
      std::cout << "This program will add histograms from a list of root files and write them" << std::endl;
      std::cout << "   to a target root file. The target file is newly created and must not" << std::endl;
      std::cout << "   exist, or if -f (\"force\") is given, must not be one of the source files." << std::endl;
      std::cout << "   Supply at least two source files for this to make sense... ;-)" << std::endl;
      std::cout << "If the option -a is used, hadd will append to the output." << std::endl;
      std::cout << "If the option -k is used, hadd will not exit on corrupt or non-existant input\n"
                   "   files but skip the offending files instead." << std::endl;
      std::cout << "If the option -T is used, Trees are not merged" <<std::endl;
      std::cout << "If the option -O is used, when merging TTree, the basket size is re-optimized" <<std::endl;
      std::cout << "If the option -v is used, explicitly set the verbosity level;\n"\
                   "   0 request no output, 99 is the default" <<std::endl;
      std::cout << "If the option -j is used, the execution will be parallelized in multiple processes\n" << std::endl;
      std::cout << "If the option -dbg is used, the execution will be parallelized in multiple processes in debug mode."
                   " This will not delete the partial files stored in the working directory\n"
                << std::endl;
      std::cout << "If the option -d is used, the partial multiprocess execution will be carried out in the specified "
                   "directory\n"
                << std::endl;
      std::cout << "If the option -n is used, hadd will open at most 'maxopenedfiles' at once, use 0\n"
                   "   to request to use the system maximum." << std::endl;
      std::cout << "If the option -cachesize is used, hadd will resize (or disable if 0) the\n"
                   "   prefetching cache use to speed up I/O operations." << std::endl;
      std::cout << "When -the -f option is specified, one can also specify the compression level of\n"
                   "   the target file.  By default the compression level is 1." <<std::endl;
      std::cout << "If \"-fk\" is specified, the target file contain the baskets with the same\n"
                   "   compression as in the input files unless -O is specified.  The meta data will\n"
                   "   be compressed using the compression level specified in the first input or the\n"
                   "   compression setting specified follow fk (206 when using -fk206 for example)" <<std::endl;
      std::cout << "If \"-ff\" is specified, the compression level use is the one specified in the\n"
                   "   first input." <<std::endl;
      std::cout << "If \"-f0\" is specified, the target file will not be compressed." <<std::endl;
      std::cout << "If \"-f6\" is specified, the compression level 6 will be used.  \n"
                   "   See TFile::SetCompressionSettings for the support range of value." <<std::endl;
      std::cout << "If Target and source files have different compression settings a slower method\n"
                   "   is used.\n"<<std::endl;
      std::cout << "For options that takes a size as argument, a decimal number of bytes is expected.\n"
                   "If the number ends with a ``k'', ``m'', ``g'', etc., the number is multiplied\n"
                   "   by 1000 (1K), 1000000 (1MB), 1000000000 (1G), etc. \n"
                   "If this prefix is followed by i, the number is multipled by the traditional\n"
                   "   1024 (1KiB), 1048576 (1MiB), 1073741824 (1GiB), etc. \n"
                   "The prefix can be optionally followed by B whose casing is ignored,\n"
                   "   eg. 1k, 1K, 1Kb and 1KB are the same."<<std::endl;

      return 1;
   }

   Bool_t append = kFALSE;
   Bool_t force = kFALSE;
   Bool_t skip_errors = kFALSE;
   Bool_t reoptimize = kFALSE;
   Bool_t noTrees = kFALSE;
   Bool_t keepCompressionAsIs = kFALSE;
   Bool_t useFirstInputCompression = kFALSE;
   Bool_t multiproc = kFALSE;
   Bool_t debug = kFALSE;
   Int_t maxopenedfiles = 0;
   Int_t verbosity = 99;
   TString cacheSize;
   SysInfo_t s;
   gSystem->GetSysInfo(&s);
   auto nProcesses = s.fCpus;
   auto workingDir = gSystem->TempDirectory();
   int outputPlace = 0;
   int ffirst = 2;
   Int_t newcomp = -1;
   for( int a = 1; a < argc; ++a ) {
      if ( strcmp(argv[a],"-T") == 0 ) {
         noTrees = kTRUE;
         ++ffirst;
      } else if ( strcmp(argv[a],"-a") == 0 ) {
         append = kTRUE;
         ++ffirst;
      } else if ( strcmp(argv[a],"-f") == 0 ) {
         force = kTRUE;
         ++ffirst;
      } else if ( strcmp(argv[a],"-k") == 0 ) {
         skip_errors = kTRUE;
         ++ffirst;
      } else if ( strcmp(argv[a],"-O") == 0 ) {
         reoptimize = kTRUE;
         ++ffirst;
      } else if (strcmp(argv[a], "-dbg") == 0) {
         debug = kTRUE;
         verbosity = kTRUE;
         ++ffirst;
      } else if (strcmp(argv[a], "-d") == 0) {
         if (a + 1 != argc && argv[a + 1][0] != '-') {
            if (gSystem->AccessPathName(argv[a + 1])) {
               std::cerr << "Error: could not access the directory specified: " << argv[a + 1]
                         << ". We will use the system's temporal directory.\n";
            } else {
               workingDir = argv[a + 1];
            }
            ++a;
            ++ffirst;
         } else {
            std::cout << "-d: no directory specified.  We will use the system's temporal directory.\n";
         }
         ++ffirst;
      } else if (strcmp(argv[a], "-j") == 0) {
         // If the number of processes is not specified, use the default.
         if (a + 1 != argc && argv[a + 1][0] != '-') {
            // number of processes specified
            Long_t request = 1;
            for (char *c = argv[a + 1]; *c != '\0'; ++c) {
               if (!isdigit(*c)) {
                  // Wrong number of Processes. Use the default:
                  std::cerr << "Error: could not parse the number of processes to run in parallel passed after -j: "
                            << argv[a + 1] << ". We will use the system maximum.\n";
                  request = 0;
                  break;
               }
            }
            if (request == 1) {
               request = strtol(argv[a + 1], 0, 10);
               if (request < kMaxLong && request >= 0) {
                  nProcesses = (Int_t)request;
                  ++a;
                  ++ffirst;
                  std::cout << "Parallelizing  with " << nProcesses << " processes.\n";
               } else {
                  std::cerr << "Error: could not parse the number of processes to use passed after -j: " << argv[a + 1]
                            << ". We will use the default value (number of logical cores).\n";
               }
            }
         }
         multiproc = kTRUE;
         ++ffirst;
      } else if ( strcmp(argv[a],"-cachesize=") == 0 ) {
         int size;
         static const size_t arglen = strlen("-cachesize=");
         auto parseResult = ROOT::FromHumanReadableSize(argv[a]+arglen,size);
         if (parseResult == ROOT::EFromHumanReadableSize::kParseFail) {
            std::cerr << "Error: could not parse the cache size passed after -cachesize: "
                      << argv[a + 1] << ". We will use the default value.\n";
         } else if (parseResult == ROOT::EFromHumanReadableSize::kOverflow) {
            double m;
            const char *munit = nullptr;
            ROOT::ToHumanReadableSize(INT_MAX,false,&m,&munit);
            std::cerr << "Error: the cache size passed after -cachesize is too large: "
                      << argv[a + 1] << " is greater than " << m << munit
                      << ". We will use the default value.\n";
         } else {
            cacheSize = "cachesize=";
            cacheSize.Append(argv[a]+1);
         }
         ++ffirst;
      } else if ( strcmp(argv[a],"-cachesize") == 0 ) {
         if (a+1 >= argc) {
            std::cerr << "Error: no cache size number was provided after -cachesize.\n";
         } else {
            int size;
            auto parseResult = ROOT::FromHumanReadableSize(argv[a+1],size);
            if (parseResult == ROOT::EFromHumanReadableSize::kParseFail) {
               std::cerr << "Error: could not parse the cache size passed after -cachesize: "
                         << argv[a + 1] << ". We will use the default value.\n";
            } else if (parseResult == ROOT::EFromHumanReadableSize::kOverflow) {
               double m;
               const char *munit = nullptr;
               ROOT::ToHumanReadableSize(INT_MAX,false,&m,&munit);
               std::cerr << "Error: the cache size passed after -cachesize is too large: "
                         << argv[a + 1] << " is greater than " << m << munit
                         << ". We will use the default value.\n";
               ++a;
               ++ffirst;
            } else {
               cacheSize = "cachesize=";
               cacheSize.Append(argv[a+1]);
               ++a;
               ++ffirst;
            }
         }
         ++ffirst;
      } else if ( strcmp(argv[a],"-n") == 0 ) {
         if (a+1 >= argc) {
            std::cerr << "Error: no maximum number of opened was provided after -n.\n";
         } else {
            Long_t request = strtol(argv[a+1], 0, 10);
            if (request < kMaxLong && request >= 0) {
               maxopenedfiles = (Int_t)request;
               ++a;
               ++ffirst;
            } else {
               std::cerr << "Error: could not parse the max number of opened file passed after -n: " << argv[a+1] << ". We will use the system maximum.\n";
            }
         }
         ++ffirst;
      } else if ( strcmp(argv[a],"-v") == 0 ) {
         if (a+1 == argc || argv[a+1][0] == '-') {
            // Verbosity level was not specified use the default:
            verbosity = 99;
//         if (a+1 >= argc) {
//            std::cerr << "Error: no verbosity level was provided after -v.\n";
         } else {
            Long_t request = -1;
            for (char *c = argv[a+1]; *c != '\0'; ++c) {
               if (!isdigit(*c)) {
                  // Verbosity level was not specified use the default:
                  request = 99;
                  break;
               }
            }
            if (request == 1) {
               request = strtol(argv[a+1], 0, 10);
               if (request < kMaxLong && request >= 0) {
                  verbosity = (Int_t)request;
                  ++a;
                  ++ffirst;
                  std::cerr << "Error: from " << argv[a+1] << " guess verbosity level : " << verbosity << "\n";
               } else {
                  std::cerr << "Error: could not parse the verbosity level passed after -v: " << argv[a+1] << ". We will use the default value (99).\n";
               }
            }
         }
         ++ffirst;
      } else if ( argv[a][0] == '-' ) {
         bool farg = false;
         if (force && argv[a][1] == 'f') {
            // Bad argument
            std::cerr << "Error: Using option " << argv[a] << " more than once is not supported.\n";
            ++ffirst;
            farg = true;
         }
         const char *prefix = "";
         if (argv[a][1] == 'f' && argv[a][2] == 'k') {
            farg = true;
            force = kTRUE;
            keepCompressionAsIs = kTRUE;
            prefix = "k";
         }
         if (argv[a][1] == 'f' && argv[a][2] == 'f') {
            farg = true;
            force = kTRUE;
            useFirstInputCompression = kTRUE;
            if (argv[a][3] != '\0') {
               std::cerr << "Error: option -ff should not have any suffix: " << argv[a] << " (suffix has been ignored)\n";
            }
         }
         char ft[7];
         for (int alg = 0; !useFirstInputCompression && alg <= 4; ++alg) {
            for( int j=0; j<=9; ++j ) {
               const int comp = (alg*100)+j;
               snprintf(ft,7,"-f%s%d",prefix,comp);
               if (!strcmp(argv[a],ft)) {
                  farg = true;
                  force = kTRUE;
                  newcomp = comp;
                  break;
               }
            }
         }
         if (!farg) {
            // Bad argument
            std::cerr << "Error: option " << argv[a] << " is not a supported option.\n";
         }
         ++ffirst;
      } else if (!outputPlace) {
         outputPlace = a;
      }
   }

   gSystem->Load("libTreePlayer");

   const char *targetname = 0;
   if (outputPlace) {
      targetname = argv[outputPlace];
   } else {
      targetname = argv[ffirst-1];
   }

   if (verbosity > 1) {
      std::cout << "hadd Target file: " << targetname << std::endl;
   }

   TFileMerger fileMerger(kFALSE, kFALSE);
   fileMerger.SetMsgPrefix("hadd");
   fileMerger.SetPrintLevel(verbosity - 1);
   if (maxopenedfiles > 0) {
      fileMerger.SetMaxOpenedFiles(maxopenedfiles);
   }
   if (newcomp == -1) {
      if (useFirstInputCompression || keepCompressionAsIs) {
         // grab from the first file.
         TFile *firstInput = nullptr;
         if (argv[ffirst] && argv[ffirst][0]=='@') {
            std::ifstream indirect_file(argv[ffirst]+1);
            if( ! indirect_file.is_open() ) {
               std::cerr<< "hadd could not open indirect file " << (argv[ffirst]+1) << std::endl;
               return 1;
            }
            std::string line;
            while( indirect_file ){
               if( std::getline(indirect_file, line) && line.length() ) {
                  firstInput = TFile::Open(line.c_str());
                  break;
               }
            }
         } else {
            firstInput = TFile::Open(argv[ffirst]);
         }
         if (firstInput && !firstInput->IsZombie())
            newcomp = firstInput->GetCompressionSettings();
         else
            newcomp = 1;
         delete firstInput;
      } else newcomp = 1; // default compression level.
   }
   if (verbosity > 1) {
      if (keepCompressionAsIs && !reoptimize)
         std::cout << "hadd compression setting for meta data: " << newcomp << '\n';
      else
         std::cout << "hadd compression setting for all ouput: " << newcomp << '\n';
   }
   if (append) {
      if (!fileMerger.OutputFile(targetname, "UPDATE", newcomp)) {
         std::cerr << "hadd error opening target file for update :" << argv[ffirst-1] << "." << std::endl;
         exit(2);
      }
   } else if (!fileMerger.OutputFile(targetname, force, newcomp)) {
      std::cerr << "hadd error opening target file (does " << argv[ffirst-1] << " exist?)." << std::endl;
      if (!force) std::cerr << "Pass \"-f\" argument to force re-creation of output file." << std::endl;
      exit(1);
   }

   auto filesToProcess = argc - ffirst;
   auto step = (filesToProcess + nProcesses - 1) / nProcesses;
   if (multiproc && step < 3) {
      // At least 3 files per process
      step = 3;
      nProcesses = (filesToProcess + step - 1) / step;
      std::cout << "Each process should handle at least 3 files for efficiency.";
      std::cout << " Setting the number of processes to: " << nProcesses << std::endl;
   }
   std::vector<std::string> partialFiles;

   if (multiproc) {
      auto uuid = TUUID();
      auto partialTail = uuid.AsString();
      for (auto i = 0; (i * step) < filesToProcess; i++) {
         std::stringstream buffer;
         buffer << workingDir << "/partial" << i << "_" << partialTail << ".root";
         partialFiles.emplace_back(buffer.str());
      }
   }

   auto mergeFiles = [&](TFileMerger &merger) {
      if (reoptimize) {
         merger.SetFastMethod(kFALSE);
      } else {
         if (!keepCompressionAsIs && merger.HasCompressionChange()) {
            // Don't warn if the user any request re-optimization.
            std::cout << "hadd Sources and Target have different compression levels" << std::endl;
            std::cout << "hadd merging will be slower" << std::endl;
         }
      }
      merger.SetNotrees(noTrees);
      merger.SetMergeOptions(cacheSize);
      Bool_t status;
      if (append)
         status = merger.PartialMerge(TFileMerger::kIncremental | TFileMerger::kAll);
      else
         status = merger.Merge();
      return status;
   };

   auto sequentialMerge = [&](TFileMerger &merger, int start, int nFiles) {

      for (auto i = start; i < (start + nFiles) && i < argc; i++) {
         if (argv[i] && argv[i][0] == '@') {
            std::ifstream indirect_file(argv[i] + 1);
            if (!indirect_file.is_open()) {
               std::cerr << "hadd could not open indirect file " << (argv[i] + 1) << std::endl;
               return kFALSE;
            }
            while (indirect_file) {
               std::string line;
               if (std::getline(indirect_file, line) && line.length() && !merger.AddFile(line.c_str())) {
                  return kFALSE;
               }
            }
         } else if (!merger.AddFile(argv[i])) {
            if (skip_errors) {
               std::cerr << "hadd skipping file with error: " << argv[i] << std::endl;
            } else {
               std::cerr << "hadd exiting due to error in " << argv[i] << std::endl;
            }
            return kFALSE;
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
      if (!mergerP.OutputFile(partialFiles[(start - ffirst) / step].c_str(), newcomp)) {
         std::cerr << "hadd error opening target partial file" << std::endl;
         exit(1);
      }
      return sequentialMerge(mergerP, start, step);
   };

   auto reductionFunc = [&]() {
      for (auto pf : partialFiles) {
         fileMerger.AddFile(pf.c_str());
      }
      return mergeFiles(fileMerger);
   };

   Bool_t status;

   if (multiproc) {
      ROOT::TProcessExecutor p(nProcesses);
      auto res = p.Map(parallelMerge, ROOT::TSeqI(ffirst, argc, step));
      status = std::accumulate(res.begin(), res.end(), 0U) == partialFiles.size();
      if (status) {
         status = reductionFunc();
      } else {
         std::cout << "hadd failed at the parallel stage" << std::endl;
      }
      if (!debug) {
         for (auto pf : partialFiles) {
            gSystem->Unlink(pf.c_str());
         }
      }
   } else {
      status = sequentialMerge(fileMerger, ffirst, filesToProcess);
   }

   if (status) {
      if (verbosity == 1) {
         std::cout << "hadd merged " << fileMerger.GetMergeList()->GetEntries() << " input files in " << targetname
                   << ".\n";
      }
      return 0;
   } else {
      if (verbosity == 1) {
         std::cout << "hadd failure during the merge of " << fileMerger.GetMergeList()->GetEntries()
                   << " input files in " << targetname << ".\n";
      }
      return 1;
   }
}
