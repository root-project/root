/*

  This program will add histograms (see note) and Trees from a list of root files and write them
  to a target root file. The target file is newly created and must not be
  identical to one of the source files.

  Syntax:

       hadd targetfile source1 source2 ...
    or
       hadd -f targetfile source1 source2 ...
         (targetfile is overwritten if it exists)

  When -the -f option is specified, one can also specify the compression
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

//___________________________________________________________________________
int main( int argc, char **argv )
{

   if ( argc < 3 || "-h" == string(argv[1]) || "--help" == string(argv[1]) ) {
      cout << "Usage: " << argv[0] << " [-f[0-9]] [-k] [-T] [-O] [-n maxopenedfiles] targetfile source1 [source2 source3 ...]" << endl;
      cout << "This program will add histograms from a list of root files and write them" << endl;
      cout << "to a target root file. The target file is newly created and must not " << endl;
      cout << "exist, or if -f (\"force\") is given, must not be one of the source files." << endl;
      cout << "Supply at least two source files for this to make sense... ;-)" << endl;
      cout << "If the option -k is used, hadd will not exit on corrupt or non-existant input files but skip the offending files instead." << endl;
      cout << "If the option -T is used, Trees are not merged" <<endl;
      cout << "If the option -O is used, when merging TTree, the basket size is re-optimized" <<endl;
      cout << "If the option -n is used, hadd will open at most 'maxopenedfiles' at once, use 0 to request to use the system maximum." << endl;
      cout << "When -the -f option is specified, one can also specify the compression" <<endl;
      cout << "level of the target file. By default the compression level is 1, but" <<endl;
      cout << "if \"-f0\" is specified, the target file will not be compressed." <<endl;
      cout << "if \"-f6\" is specified, the compression level 6 will be used." <<endl;
      cout << "if Target and source files have different compression levels"<<endl;
      cout << " a slower method is used"<<endl;
      return 1;
   }

   Bool_t force = kFALSE;
   Bool_t skip_errors = kFALSE;
   Bool_t reoptimize = kFALSE;
   Bool_t noTrees = kFALSE;
   Int_t maxopenedfiles = 0;

   int outputPlace = 0;
   int ffirst = 2;
   Int_t newcomp = 1;
   for( int a = 1; a < argc; ++a ) {
      if ( strcmp(argv[a],"-T") == 0 ) {
         noTrees = kTRUE;
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
            cerr << "Error: no maximum number of opened was provided after -n.\n";
         } else {
            long request = strtol(argv[a+1], 0, 10);
            if (request < kMaxLong && request >= 0) {
               maxopenedfiles = (Int_t)request;
               ++a;
               ++ffirst;
            } else {
               cerr << "Error: could not parse the max number of opened file passed after -n: " << argv[a+1] << ". We will use the system maximum.\n";
            }
         }
         ++ffirst;
      } else if ( argv[a][0] == '-' ) {
         char ft[4];
         for( int j=0; j<=9; ++j ) {
            snprintf(ft,4,"-f%d",j);
            if (!strcmp(argv[a],ft)) {
               force = kTRUE;
               newcomp = j;
               ++ffirst;
               break;
            }
         }
         if (!force) {
            // Bad argument
            cerr << "Error: option " << argv[a] << " is not a supported option.\n";
            ++ffirst;
         }
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
      
   cout << "Target file: " << targetname << endl;

   TFileMerger merger(kFALSE,kFALSE);
   merger.SetPrintLevel(99);
   if (maxopenedfiles > 0) {
      merger.SetMaxOpenedFiles(maxopenedfiles);
   }
   if (!merger.OutputFile(targetname,force,newcomp) ) {
      cerr << "Error opening target file (does " << argv[ffirst-1] << " exist?)." << endl;
      cerr << "Pass \"-f\" argument to force re-creation of output file." << endl;
      exit(1);
   }

   for ( int i = ffirst; i < argc; i++ ) {
      if (argv[i] && argv[i][0]=='@') {
         std::ifstream indirect_file(argv[i]+1);
         if( ! indirect_file.is_open() ) {
            std::cerr<< "Could not open indirect file " << (argv[i]+1) << std::endl;
            return 1;
         }
         while( indirect_file ){
            std::string line;
            std::getline(indirect_file, line);
            if( !merger.AddFile(line.c_str()) ) {
               return 1;
            }
         }         
      } else if( ! merger.AddFile(argv[i]) ) {
         if ( skip_errors ) {
            cerr << "Skipping file with error: " << argv[i] << endl;
         } else {
            cerr << "Exiting due to error in " << argv[i] << endl;
            return 1;
         }
      }
   }
   if (reoptimize) {
      merger.SetFastMethod(kFALSE);
   } else {
      if (merger.HasCompressionChange()) {
         // Don't warn if the user any request re-optimization.
         cout <<"Sources and Target have different compression levels"<<endl;
         cout <<"Merging will be slower"<<endl;
      }
   }
   merger.SetNotrees(noTrees);
   Bool_t status = merger.Merge();

   if (status) {
      return 0;
   } else {
      return 1;
   }
}
