// @(#)root/roofitcore:$name:  $:$id$
// Authors: Wouter Verkerke  November 2007

// C/C++ headers
#include <list>
#include <iostream>
#include <iomanip>

// Math headers
#include "Math/MinimizerOptions.h"

// ROOT headers
#include "TWebFile.h"
#include "TSystem.h"
#include "TString.h"
#include "TROOT.h"
#include "TFile.h"
#include "TBenchmark.h"

// RooFit headers
#include "RooNumIntConfig.h"
#include "RooResolutionModel.h"
#include "RooAbsData.h"
#include "RooTrace.h"
#include "RooUnitTest.h"

// Tests file
#include "stressHistFactory_tests.cxx"

using namespace std ;
using namespace RooFit ;


//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*//
//                                                                           //
// RooStats Unit Test S.T.R.E.S.S. Suite                                     //
// Authors: Ioan Gabriel Bucur, Lorenzo Moneta, Wouter Verkerke              //
//                                                                           //
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*//


////////////////////////////////////////////////////////////////////////////////
/// Print test program number and its title

void StatusPrint(const int id, const TString &title, const int status, const int lineWidth)
{
   TString header = TString::Format("Test %d : %s ", id, title.Data());
   cout << left << setw(lineWidth) << setfill('.') << header << " " << (status > 0 ? "OK" : (status < 0 ? "SKIPPED" : "FAILED")) << endl;
}

////////////////////////////////////////////////////////////////////////////////
/// width of lines when printing test results

int stressHistFactory(const char* refFile, bool writeRef, int verbose, bool allTests, bool oneTest, int testNumber, bool dryRun)
{
   const int lineWidth = 120;

   // Save memory directory location
   RooUnitTest::setMemDir(gDirectory) ;

   std::cout << "using reference file " << refFile << std::endl;
   TFile* fref = 0 ;
   if (!dryRun) {
      if (TString(refFile).Contains("http:")) {
         if (writeRef) {
            cout << "stressHistFactory ERROR: reference file must be local file in writing mode" << endl ;
            return -1 ;
         }
         fref = new TWebFile(refFile) ;
      } else {
         fref = new TFile(refFile, writeRef ? "RECREATE" : "") ;
      }
      if (fref->IsZombie()) {
         cout << "stressHistFactory ERROR: cannot open reference file " << refFile << endl ;
         return -1;
      }
   }

   if (dryRun) {
      // Preload singletons here so they don't show up in trace accounting
      RooNumIntConfig::defaultConfig() ;
      RooResolutionModel::identity() ;

      RooTrace::active(1) ;
   }

   // Add dedicated logging stream for errors that will remain active in silent mode
   RooMsgService::instance().addStream(RooFit::ERROR) ;


   cout << left << setw(lineWidth) << setfill('*') << "" << endl;
   cout << "*" << setw(lineWidth - 2) << setfill(' ') << " Histfactory S.T.R.E.S.S. suite " << "*" << endl;
   cout << setw(lineWidth) << setfill('*') << "" << endl;
   cout << setw(lineWidth) << setfill('*') << "" << endl;


   TStopwatch timer;
   timer.Start();

   list<RooUnitTest*> testList;
   testList.push_back(new PdfComparison(fref, writeRef, verbose));

   TString suiteType = TString::Format(" Starting S.T.R.E.S.S. %s",
                                       allTests ? "full suite" : (oneTest ? TString::Format("test %d", testNumber).Data() : "basic suite")
                                      );

   cout << "*" << setw(lineWidth - 3) << setfill(' ') << suiteType << " *" << endl;
   cout << setw(lineWidth) << setfill('*') << "" << endl;

   gBenchmark->Start("stressHistFactory");

   int nFailed = 0;
   {
      int i;
      list<RooUnitTest*>::iterator iter;

      if (oneTest && (testNumber <= 0 || (UInt_t) testNumber > testList.size())) {
         cout << "Tests are numbered from 1 to " << testList.size() << endl;
         return -1;
      } else {
         for (iter = testList.begin(), i = 1; iter != testList.end(); iter++, i++) {
            if (!oneTest || testNumber == i) {
               int status = (*iter)->isTestAvailable() ? (*iter)->runTest() : -1;
               StatusPrint(i, (*iter)->GetName(), status , lineWidth);
               if (status <= 0) nFailed++; // successfull tests return a positive number
            }
            delete *iter;
         }
      }
   }

   if (dryRun) {
      RooTrace::dump();
   }

   gBenchmark->Stop("stressHistFactory");


   //Print table with results
   bool UNIX = strcmp(gSystem->GetName(), "Unix") == 0;
   cout << setw(lineWidth) << setfill('*') << "" << endl;
   if (UNIX) {
      TString sp = gSystem->GetFromPipe("uname -a");
      cout << "* SYS: " << sp << endl;
      if (strstr(gSystem->GetBuildNode(), "Darwin")) {
         sp  = gSystem->GetFromPipe("sw_vers -productVersion");
         sp += " Mac OS X ";
         cout << "* SYS: " << sp << endl;
      }
   } else {
      const Char_t *os = gSystem->Getenv("OS");
      if (!os) cout << "*  SYS: Windows 95" << endl;
      else     cout << "*  SYS: " << os << " " << gSystem->Getenv("PROCESSOR_IDENTIFIER") << endl;
   }

   cout << setw(lineWidth) << setfill('*') << "" << endl;
   gBenchmark->Print("stressHistFactory");

   Double_t reftime = 2.4; //maclm compiled
   const Double_t rootmarks = 860 * reftime / gBenchmark->GetCpuTime("stressHistFactory");

   cout << setw(lineWidth) << setfill('*') << "" << endl;
   cout << TString::Format("*  ROOTMARKS = %6.1f  *  Root %-8s %d/%d", rootmarks, gROOT->GetVersion(),
                           gROOT->GetVersionDate(), gROOT->GetVersionTime()) << endl;
   cout << setw(lineWidth) << setfill('*') << "" << endl;

   // NOTE: The function TStopwatch::CpuTime() calls Tstopwatch::Stop(), so you do not need to stop the timer separately.
   cout << "Time at the end of job = " << timer.CpuTime() << " seconds" << endl;

   if (fref) {
      fref->Close() ;
      delete fref ;
   }

   delete gBenchmark ;
   gBenchmark = 0 ;

   return nFailed;
}

//_____________________________batch only_____________________
#ifndef __CINT__

int main(int argc, const char *argv[])
{
   bool doWrite     = false;
   int  verbose     =      0;
   bool allTests    = false;
   bool oneTest     = false;
   int testNumber   =      0;
   bool dryRun      = false;

   string refFileName = "stressHistFactory_ref.root" ;
   string minimizerName = "Minuit";


   // Parse command line arguments
   for (int i = 1 ;  i < argc ; i++) {
      string arg = argv[i] ;

      if (arg == "-f") {
         cout << "stressHistFactory: using reference file " << argv[i + 1] << endl ;
         refFileName = argv[++i] ;
      } else if (arg == "-w") {
         cout << "stressHistFactory: running in writing mode to update reference file" << endl ;
         doWrite = true ;
      } else if (arg == "-mc") {
         cout << "stressHistFactory: running in memcheck mode, no regression tests are performed" << endl;
         dryRun = true;
      } else if (arg == "-min" || arg == "-minim") {
         cout << "stressHistFactory: running using minimizer " << argv[i +1]  << endl;
         minimizerName = argv[++i] ;
      } else if (arg == "-v") {
         cout << "stressHistFactory: running in verbose mode" << endl;
         verbose = 1;
      } else if (arg == "-vv") {
         cout << "stressHistFactory: running in very verbose mode" << endl;
         verbose = 2;
      } else if (arg == "-a") {
         cout << "stressHistFactory: deploying full suite of tests" << endl;
         allTests = true;
      } else if (arg == "-n") {
         cout << "stressHistFactory: running single test" << endl;
         oneTest = true;
         testNumber = atoi(argv[++i]);
      } else if (arg == "-d") {
         cout << "stressHistFactory: setting gDebug to " << argv[i + 1] << endl;
         gDebug = atoi(argv[++i]);
      } else if (arg == "-h" || arg == "--help") {
         cout << R"(usage: stressHistFactory [ options ]

       -f <file>   : use given reference file instead of default ("stressHistFactory_ref.root")
       -w          : write reference file, instead of reading file and running comparison tests
       -n N        : only run test with sequential number N
       -a          : run full suite of tests (default is basic suite); this overrides the -n single test option
       -mc         : memory check mode, no regression test are performed. Set this flag when running with valgrind
       -min <name> : minimizer name (default is Minuit, not Minuit2)
       -vs         : use vector-based storage for all datasets (default is tree-based storage)
       -v/-vv      : set verbose mode (show result of each regression test) or very verbose mode (show all roofit output as well)
       -d N        : set ROOT gDebug flag to N
)";
         return 0 ;
      }

   }

   // set minimizer
   ROOT::Math::MinimizerOptions::SetDefaultMinimizer(minimizerName.c_str());

   gBenchmark = new TBenchmark();
   return stressHistFactory(refFileName.c_str(), doWrite, verbose, allTests, oneTest, testNumber, dryRun);
}

////////////////////////////////////////////////////////////////////////////////

int stressHistFactory()
{
   bool doWrite     = false;
   int  verbose     =     0;
   bool allTests    = false;
   bool oneTest     = false;
   int testNumber   =     0;
   bool dryRun      = false;
   string refFileName = "stressHistFactory_ref.root" ;

   // in interpreted mode, the minimizer is hardcoded to Minuit 1
   ROOT::Math::MinimizerOptions::SetDefaultMinimizer("Minuit");

   return stressHistFactory(refFileName.c_str(), doWrite, verbose, allTests, oneTest, testNumber, dryRun);
}

#endif
