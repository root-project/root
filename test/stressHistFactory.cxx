// @(#)root/roofitcore:$name:  $:$id$
// Authors: Wouter Verkerke  November 2007

// C/C++ headers
#include <list>
#include <iostream>
#include <iomanip>

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

void StatusPrint(const Int_t id, const TString &title, const Int_t status, const Int_t lineWidth)
{
   TString header = TString::Format("Test %d : %s ", id, title.Data());
   cout << left << setw(lineWidth) << setfill('.') << header << " " << (status > 0 ? "OK" : (status < 0 ? "SKIPPED" : "FAILED")) << endl;
}

////////////////////////////////////////////////////////////////////////////////
/// width of lines when printing test results

Int_t stressHistFactory(const char* refFile, Bool_t writeRef, Int_t verbose, Bool_t allTests, Bool_t oneTest, Int_t testNumber, Bool_t dryRun)
{
   const Int_t lineWidth = 120;

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
      Int_t i;
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
   Bool_t UNIX = strcmp(gSystem->GetName(), "Unix") == 0;
   cout << setw(lineWidth) << setfill('*') << "" << endl;
   if (UNIX) {
      TString sp = gSystem->GetFromPipe("uname -a");
      cout << "* SYS: " << sp << endl;
      if (strstr(gSystem->GetBuildNode(), "Linux")) {
         sp = gSystem->GetFromPipe("lsb_release -d -s");
         cout << "* SYS: " << sp << endl;
      }
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
   Bool_t doWrite     = kFALSE;
   Int_t  verbose     =      0;
   Bool_t allTests    = kFALSE;
   Bool_t oneTest     = kFALSE;
   Int_t testNumber   =      0;
   Bool_t dryRun      = kFALSE;

   string refFileName = "$ROOTSYS/test/stressHistFactory_ref.root" ;


   // Parse command line arguments
   for (Int_t i = 1 ;  i < argc ; i++) {
      string arg = argv[i] ;

      if (arg == "-f") {
         cout << "stressHistFactory: using reference file " << argv[i + 1] << endl ;
         refFileName = argv[++i] ;
      } else if (arg == "-w") {
         cout << "stressHistFactory: running in writing mode to update reference file" << endl ;
         doWrite = kTRUE ;
      } else if (arg == "-mc") {
         cout << "stressHistFactory: running in memcheck mode, no regression tests are performed" << endl;
         dryRun = kTRUE;
      } else if (arg == "-v") {
         cout << "stressHistFactory: running in verbose mode" << endl;
         verbose = 1;
      } else if (arg == "-vv") {
         cout << "stressHistFactory: running in very verbose mode" << endl;
         verbose = 2;
      } else if (arg == "-a") {
         cout << "stressHistFactory: deploying full suite of tests" << endl;
         allTests = kTRUE;
      } else if (arg == "-n") {
         cout << "stressHistFactory: running single test" << endl;
         oneTest = kTRUE;
         testNumber = atoi(argv[++i]);
      } else if (arg == "-d") {
         cout << "stressHistFactory: setting gDebug to " << argv[i + 1] << endl;
         gDebug = atoi(argv[++i]);
      } else if (arg == "-h") {
         cout << "usage: stressHistFactory [ options ] " << endl;
         cout << "" << endl;
         cout << "       -f <file> : use given reference file instead of default (" << refFileName << ")" << endl;
         cout << "       -w        : write reference file, instead of reading file and running comparison tests" << endl;
         cout << "       -n N      : only run test with sequential number N" << endl;
         cout << "       -a        : run full suite of tests (default is basic suite); this overrides the -n single test option" << endl;
         cout << "       -mc       : memory check mode, no regression test are performed. Set this flag when running with valgrind" << endl;
         cout << "       -vs       : use vector-based storage for all datasets (default is tree-based storage)" << endl;
         cout << "       -v/-vv    : set verbose mode (show result of each regression test) or very verbose mode (show all roofit output as well)" << endl;
         cout << "       -d N      : set ROOT gDebug flag to N" << endl ;
         cout << " " << endl ;
         return 0 ;
      }

   }

   gBenchmark = new TBenchmark();
   return stressHistFactory(refFileName.c_str(), doWrite, verbose, allTests, oneTest, testNumber, dryRun);
}

////////////////////////////////////////////////////////////////////////////////

Int_t stressHistFactory()
{
   Bool_t doWrite     = kFALSE;
   Int_t  verbose     =      0;
   Bool_t allTests    = kFALSE;
   Bool_t oneTest     = kFALSE;
   Int_t testNumber   =      0;
   Bool_t dryRun      = kFALSE;

   string refFileName = "$ROOTSYS/test/stressHistFactory_ref.root" ;
   return stressHistFactory(refFileName.c_str(), doWrite, verbose, allTests, oneTest, testNumber, dryRun);
}

#endif
