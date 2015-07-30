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
#include <stdlib.h>

#include "TFileMerger.h"

////////////////////////////////////////////////////////////////////////////////

int main( int argc, char **argv )
{
   if ( argc < 3 || "-h" == std::string(argv[1]) || "--help" == std::string(argv[1]) ) {
      std::cout << "Usage: " << argv[0] << " [-f[fk][0-9]] [-k] [-T] [-O] [-a] [-n maxopenedfiles] [-v [verbosity]] targetfile source1 [source2 source3 ...]" << std::endl;
      std::cout << "This program will add histograms from a list of root files and write them" << std::endl;
      std::cout << "to a target root file. The target file is newly created and must not " << std::endl;
      std::cout << "exist, or if -f (\"force\") is given, must not be one of the source files." << std::endl;
      std::cout << "Supply at least two source files for this to make sense... ;-)" << std::endl;
      std::cout << "If the option -a is used, hadd will append to the output." << std::endl;
      std::cout << "If the option -k is used, hadd will not exit on corrupt or non-existant input files but skip the offending files instead." << std::endl;
      std::cout << "If the option -T is used, Trees are not merged" <<std::endl;
      std::cout << "If the option -O is used, when merging TTree, the basket size is re-optimized" <<std::endl;
      std::cout << "If the option -v is used, explicitly set the verbosity level; 0 request no output, 99 is the default" <<std::endl;
      std::cout << "If the option -n is used, hadd will open at most 'maxopenedfiles' at once, use 0 to request to use the system maximum." << std::endl;
      std::cout << "When -the -f option is specified, one can also specify the compression level of the target file.\n"
                   "By default the compression level is 1, but" <<std::endl;
      std::cout << "if \"-fk\" is specified, the target file contain the baskets with the same compression as in the input files \n"
                   "  unless -O is specified.  The meta data will be compressed using the compression level specified in the first\n"
                   "  input or the compression setting specified follow fk (206 when using -fk206 for example)" <<std::endl;
      std::cout << "if \"-ff\" is specified, the compression level use is the one specified in the first input." <<std::endl;
      std::cout << "if \"-f0\" is specified, the target file will not be compressed." <<std::endl;
      std::cout << "if \"-f6\" is specified, the compression level 6 will be used.  See  TFile::SetCompressionSettings for the support range of value." <<std::endl;
      std::cout << "if Target and source files have different compression settings"<<std::endl;
      std::cout << " a slower method is used"<<std::endl;
      return 1;
   }

   Bool_t append = kFALSE;
   Bool_t force = kFALSE;
   Bool_t skip_errors = kFALSE;
   Bool_t reoptimize = kFALSE;
   Bool_t noTrees = kFALSE;
   Bool_t keepCompressionAsIs = kFALSE;
   Bool_t useFirstInputCompression = kFALSE;
   Int_t maxopenedfiles = 0;
   Int_t verbosity = 99;

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
         if (force && argv[a][1] == 'f') {
            // Bad argument
            std::cerr << "Error: Using option " << argv[a] << " more than once is not supported.\n";
            ++ffirst;
         }
         const char *prefix = "";
         if (argv[a][1] == 'f' && argv[a][2] == 'k') {
            force = kTRUE;
            keepCompressionAsIs = kTRUE;
            prefix = "k";
         }
         if (argv[a][1] == 'f' && argv[a][2] == 'f') {
            force = kTRUE;
            useFirstInputCompression = kTRUE;
            if (argv[a][3] != '\0') {
               std::cerr << "Error: option -ff should not have any suffix: " << argv[a] << " (suffix has been ignored)\n";
            }
         }
         char ft[7];
         for ( int alg = 0; !useFirstInputCompression && alg <= 2; ++alg ) {
            for( int j=0; j<=9; ++j ) {
               const int comp = (alg*100)+j;
               snprintf(ft,7,"-f%s%d",prefix,comp);
               if (!strcmp(argv[a],ft)) {
                  force = kTRUE;
                  newcomp = comp;
                  break;
               }
            }
         }
         if (!force) {
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

   TFileMerger merger(kFALSE,kFALSE);
   merger.SetMsgPrefix("hadd");
   merger.SetPrintLevel(verbosity - 1);
   if (maxopenedfiles > 0) {
      merger.SetMaxOpenedFiles(maxopenedfiles);
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
      if (!merger.OutputFile(targetname,"UPDATE",newcomp)) {
         std::cerr << "hadd error opening target file for update :" << argv[ffirst-1] << "." << std::endl;
         exit(2);
      }
   } else if (!merger.OutputFile(targetname,force,newcomp) ) {
      std::cerr << "hadd error opening target file (does " << argv[ffirst-1] << " exist?)." << std::endl;
      if (!force) std::cerr << "Pass \"-f\" argument to force re-creation of output file." << std::endl;
      exit(1);
   }


   for ( int i = ffirst; i < argc; i++ ) {
      if (argv[i] && argv[i][0]=='@') {
         std::ifstream indirect_file(argv[i]+1);
         if( ! indirect_file.is_open() ) {
            std::cerr<< "hadd could not open indirect file " << (argv[i]+1) << std::endl;
            return 1;
         }
         while( indirect_file ){
            std::string line;
            if( std::getline(indirect_file, line) && line.length() &&  !merger.AddFile(line.c_str()) ) {
               return 1;
            }
         }
      } else if( ! merger.AddFile(argv[i]) ) {
         if ( skip_errors ) {
            std::cerr << "hadd skipping file with error: " << argv[i] << std::endl;
         } else {
            std::cerr << "hadd exiting due to error in " << argv[i] << std::endl;
            return 1;
         }
      }
   }
   if (reoptimize) {
      merger.SetFastMethod(kFALSE);
   } else {
      if (!keepCompressionAsIs && merger.HasCompressionChange()) {
         // Don't warn if the user any request re-optimization.
         std::cout <<"hadd Sources and Target have different compression levels"<<std::endl;
         std::cout <<"hadd merging will be slower"<<std::endl;
      }
   }
   merger.SetNotrees(noTrees);
   Bool_t status;
   if (append) status = merger.PartialMerge(TFileMerger::kIncremental | TFileMerger::kAll);
   else status = merger.Merge();

   if (status) {
      if (verbosity == 1) {
         std::cout << "hadd merged " << merger.GetMergeList()->GetEntries() << " input files in " << targetname << ".\n";
      }
      return 0;
   } else {
      if (verbosity == 1) {
         std::cout << "hadd failure during the merge of " << merger.GetMergeList()->GetEntries() << " input files in " << targetname << ".\n";
      }
      return 1;
   }
}
