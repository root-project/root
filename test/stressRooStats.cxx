// @(#)root/roofitcore:$name:  $:$id$
// Authors: Wouter Verkerke  November 2007

#include "TWebFile.h"
#include "TSystem.h"
#include "TString.h"
#include "TStopwatch.h"
#include "TROOT.h"
#include "TLine.h"
#include "TFile.h"
#include "TClass.h"
#include "TCanvas.h"
#include "TH1.h"
#include "TF1.h"
#include "TBenchmark.h"
#include "RooGlobalFunc.h"
#include "RooNumIntConfig.h"
#include "RooMsgService.h"
#include "RooResolutionModel.h"
#include "RooPlot.h"
#include "RooFitResult.h"
#include "RooDouble.h"
#include "RooWorkspace.h"
#include "Roo1DTable.h"
#include "RooCurve.h"
#include "RooHist.h"
#include "RooRandom.h"
#include "RooTrace.h"
#include "RooMath.h"
#include <string>
#include <list>
#include <iostream>
#include <math.h>

using namespace std ;
using namespace RooFit ;

//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*//
//                                                                           //
// RooStats Examples, Wouter Verkerke, Lorenzo Moneta, Ioan Gabriel Bucur    //
//                                                                           //
//                                                                           //
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*_*//


//------------------------------------------------------------------------
void StatusPrint(const Int_t id, const TString &title, const Int_t status, const Int_t lineWidth)
{
   // Print test program number and its title
   TString header = TString::Format("Test %d : %s ", id, title.Data());
   cout << left << setw(lineWidth) << setfill('.') << header << " " << (status > 0 ? "OK" : (status < 0 ? "SKIPPED" : "FAILED")) << endl;
}

#include "stressRooStats_tests.cxx"

//______________________________________________________________________________
Int_t stressRooStats(const char* refFile, Bool_t writeRef, Int_t verbose, Int_t oneTest, Bool_t dryRun, Bool_t doDump, Bool_t doTreeStore)
{
   // width of lines when printing test results
   const Int_t lineWidth = 80;

   // Save memory directory location
   RooUnitTest::setMemDir(gDirectory) ;

   if (doTreeStore) {
      RooAbsData::setDefaultStorageType(RooAbsData::Tree) ;
   }

   TFile* fref = 0 ;
   if (!dryRun) {
      if (TString(refFile).Contains("http:")) {
         if (writeRef) {
            cout << "stressRooStats ERROR: reference file must be local file in writing mode" << endl ;
            return kFALSE ;
         }
         fref = new TWebFile(refFile) ;
      } else {
         fref = new TFile(refFile, writeRef ? "RECREATE" : "") ;
      }
      if (fref->IsZombie()) {
         cout << "stressRooStats ERROR: cannot open reference file " << refFile << endl ;
         return kFALSE ;
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
   cout << "*" << setw(lineWidth - 2) << setfill(' ') << " RooStats S.T.R.E.S.S. suite " << "*" << endl;
   cout << setw(lineWidth) << setfill('*') << "" << endl;
   cout << setw(lineWidth) << setfill('*') << "" << endl;


   TStopwatch timer;
   timer.Start();

   list<RooUnitTest*> testList ;

   // TEST PROFILE LIKELIHOOD CALCULATOR 1 : Confidence Level range is (0,1)
   testList.push_back(new TestProfileLikelihoodCalculator1(fref, writeRef, verbose, 0.99999)); // boundary case CL -> 1
   testList.push_back(new TestProfileLikelihoodCalculator1(fref, writeRef, verbose, 2 * ROOT::Math::normal_cdf(3) - 1)); // 3 sigma
   testList.push_back(new TestProfileLikelihoodCalculator1(fref, writeRef, verbose, 2 * ROOT::Math::normal_cdf(2) - 1)); // 2 sigma
   testList.push_back(new TestProfileLikelihoodCalculator1(fref, writeRef, verbose, 2 * ROOT::Math::normal_cdf(1) - 1)); // 1 sigma
   testList.push_back(new TestProfileLikelihoodCalculator1(fref, writeRef, verbose, 0.00001)); // boundary case CL -> 0

   // TEST PROFILE LIKELIHOOD CALCULATOR 2 : Observed value range is [0,1000]
   testList.push_back(new TestProfileLikelihoodCalculator2(fref, writeRef, verbose, 0)); // boundary Poisson value (0)
   testList.push_back(new TestProfileLikelihoodCalculator2(fref, writeRef, verbose, 1));
   testList.push_back(new TestProfileLikelihoodCalculator2(fref, writeRef, verbose, 5));
   testList.push_back(new TestProfileLikelihoodCalculator2(fref, writeRef, verbose, 100));
   testList.push_back(new TestProfileLikelihoodCalculator2(fref, writeRef, verbose, 800)); // boundary Poisson value

   // TEST PROFILE LIKELIHOOD CALCULATOR 3 : Observed value range is [0,40] for x=s+b and [0,120] for y=2*s*1.2^beta
   testList.push_back(new TestProfileLikelihoodCalculator3(fref, writeRef, verbose, 0, 0));
   testList.push_back(new TestProfileLikelihoodCalculator3(fref, writeRef, verbose, 10, 10));
   testList.push_back(new TestProfileLikelihoodCalculator3(fref, writeRef, verbose, 15, 30));
   testList.push_back(new TestProfileLikelihoodCalculator3(fref, writeRef, verbose, 30, 5));
   testList.push_back(new TestProfileLikelihoodCalculator3(fref, writeRef, verbose, 40, 120));

   testList.push_back(new TestProfileLikelihoodCalculator4(fref, writeRef, verbose));

   // TEST PROFILE LIKELIHOOD CALCULATOR 1 : Observed value range is [0,100]
   testList.push_back(new TestBayesianCalculator1(fref, writeRef, verbose, 1));
   testList.push_back(new TestBayesianCalculator1(fref, writeRef, verbose, 3));
   testList.push_back(new TestBayesianCalculator1(fref, writeRef, verbose, 10));
   testList.push_back(new TestBayesianCalculator1(fref, writeRef, verbose, 50));

   testList.push_back(new TestBayesianCalculator2(fref, writeRef, verbose));

///   testList.push_back(new TestBasic101(fref, writeRef, verbose));
//   testList.push_back(new TestHypoTestCalculator(fref, writeRef, verbose));
//   testList.push_back(new TestHypoTestCalculator2(fref, writeRef, verbose));
//   testList.push_back(new TestHypoTestCalculator3(fref, writeRef, verbose));
//  testList.push_back(new TestBayesianCalculator3(fref, writeRef, verbose));
//   testList.push_back(new TestMCMCCalculator(fref, writeRef, verbose));
//   testList.push_back(new TestHypoTestInverter1(fref, writeRef, verbose, HypoTestInverter::kAsymptotic, kSimpleLR));
//   testList.push_back(new TestHypoTestInverter1(fref, writeRef, verbose, HypoTestInverter::kHybrid, kSimpleLR));
//   testList.push_back(new TestHypoTestInverter1(fref, writeRef, verbose, HypoTestInverter::kFrequentist, kSimpleLR));
//   testList.push_back(new TestHypoTestInverter1(fref, writeRef, verbose, HypoTestInverter::kFrequentist, kRatioLR));
//   testList.push_back(new TestHypoTestInverter1(fref, writeRef, verbose, HypoTestInverter::kFrequentist, kProfileLR));
//   testList.push_back(new TestHypoTestInverter1(fref, writeRef, verbose, HypoTestInverter::kFrequentist, kProfileLROneSided));
//   testList.push_back(new TestHypoTestInverter1(fref, writeRef, verbose, HypoTestInverter::kFrequentist, kProfileLRSigned));
//   testList.push_back(new TestHypoTestInverter1(fref, writeRef, verbose, HypoTestInverter::kFrequentist, kMLE));
//   testList.push_back(new TestHypoTestInverter1(fref, writeRef, verbose, HypoTestInverter::kFrequentist, kNObs));
//   testList.push_back(new TestHypoTestInverter1(fref, writeRef, verbose, HypoTestInverter::kHybrid, kSimpleLR));
//   testList.push_back(new TestHypoTestInverter1(fref, writeRef, verbose, HypoTestInverter::kHybrid, kRatioLR));
//   testList.push_back(new TestHypoTestInverter1(fref, writeRef, verbose, HypoTestInverter::kHybrid, kProfileLR));
//   testList.push_back(new TestHypoTestInverter1(fref, writeRef, verbose, HypoTestInverter::kHybrid, kProfileLROneSided));
//   testList.push_back(new TestHypoTestInverter1(fref, writeRef, verbose, HypoTestInverter::kHybrid, kProfileLRSigned));
//   testList.push_back(new TestHypoTestInverter1(fref, writeRef, verbose, HypoTestInverter::kHybrid, kMLE));
//   testList.push_back(new TestHypoTestInverter1(fref, writeRef, verbose, HypoTestInverter::kHybrid, kNObs));
//   testList.push_back(new TestHypoTestInverter2(fref, writeRef, verbose, HypoTestInverter::kFrequentist, kSimpleLR));
//   testList.push_back(new TestHypoTestInverter2(fref, writeRef, verbose, HypoTestInverter::kFrequentist, kRatioLR));
//   testList.push_back(new TestHypoTestInverter2(fref, writeRef, verbose, HypoTestInverter::kFrequentist, kProfileLR));
//   testList.push_back(new TestHypoTestInverter2(fref, writeRef, verbose, HypoTestInverter::kFrequentist, kProfileLROneSided));
//   testList.push_back(new TestHypoTestInverter2(fref, writeRef, verbose, HypoTestInverter::kFrequentist, kProfileLRSigned));
//   testList.push_back(new TestHypoTestInverter2(fref, writeRef, verbose, HypoTestInverter::kFrequentist, kMLE));
//   testList.push_back(new TestHypoTestInverter2(fref, writeRef, verbose, HypoTestInverter::kFrequentist, kNObs));

   cout << "*" << setw(lineWidth - 2) << setfill(' ') << " Starting S.T.R.E.S.S. basic suite " << "*" << endl;
   cout << setw(lineWidth) << setfill('*') << "" << endl;

   if (doDump) {
      TFile fdbg("stressRooStats_DEBUG.root", "RECREATE") ;
   }

   gBenchmark->Start("StressRooStats");

   Int_t i(1) ;
   for (list<RooUnitTest*>::iterator iter = testList.begin() ; iter != testList.end() ; ++iter) {
      if (oneTest < 0 || oneTest == i) {
         if (doDump) {
            (*iter)->setDebug(kTRUE);
         }
         StatusPrint(i, (*iter)->GetName(), (*iter)->isTestAvailable() ? (*iter)->runTest() : -1, lineWidth);
      }
      delete *iter;
      i++;
   }

   if (dryRun) {
      RooTrace::dump();
   }

   gBenchmark->Stop("StressRooStats");


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
   gBenchmark->Print("StressFit");
#ifdef __CINT__
   Double_t reftime = 186.34; //pcbrun4 interpreted
#else
   Double_t reftime = 93.59; //pcbrun4 compiled
#endif
   const Double_t rootmarks = 860 * reftime / gBenchmark->GetCpuTime("StressRooStats");

   cout << setw(lineWidth) << setfill('*') << "" << endl;
   cout << TString::Format("*  ROOTMARKS = %6.1f   *  Root%-8s  %d/%d", rootmarks, gROOT->GetVersion(),
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

   return 0;
}

//_____________________________batch only_____________________
#ifndef __CINT__

int main(int argc, const char *argv[])
{
   Bool_t doWrite     = kFALSE ;
   Int_t verbose    = 0 ;
   Int_t oneTest      = -1 ;
   Bool_t dryRun      = kFALSE ;
   Bool_t doDump      = kFALSE ;
   Bool_t doTreeStore = kFALSE ;

   string refFileName = "http://root.cern.ch/files/stressRooStats_v534_ref.root" ;

   // Parse command line arguments
   for (Int_t i = 1 ;  i < argc ; i++) {
      string arg = argv[i] ;

      if (arg == "-f") {
         cout << "stressRooStats: using reference file " << argv[i + 1] << endl ;
         refFileName = argv[++i] ;
      }

      if (arg == "-w") {
         cout << "stressRooStats: running in writing mode to update reference file" << endl ;
         doWrite = kTRUE ;
      }

      if (arg == "-mc") {
         cout << "stressRooStats: running in memcheck mode, no regression tests are performed" << endl ;
         dryRun = kTRUE ;
      }

      if (arg == "-ts") {
         cout << "stressRooStats: setting tree-based storage for datasets" << endl ;
         doTreeStore = kTRUE ;
      }

      if (arg == "-v") {
         cout << "stressRooStats: running in verbose mode" << endl ;
         verbose = 1 ;
      }

      if (arg == "-vv") {
         cout << "stressRooStats: running in very verbose mode" << endl ;
         verbose = 2 ;
      }

      if (arg == "-n") {
         cout << "stressRooStats: running single test " << argv[i + 1] << endl ;
         oneTest = atoi(argv[++i]) ;
      }

      if (arg == "-d") {
         cout << "stressRooStats: setting gDebug to " << argv[i + 1] << endl ;
         gDebug = atoi(argv[++i]) ;
      }

      if (arg == "-c") {
         cout << "stressRooStats: dumping comparison file for failed tests " << endl ;
         doDump = kTRUE ;
      }

      if (arg == "-h") {
         cout << "usage: stressRooStats [ options ] " << endl ;
         cout << "" << endl ;
         cout << "       -f <file> : use given reference file instead of default (" <<  refFileName << ")" << endl ;
         cout << "       -w        : write reference file, instead of reading file and running comparison tests" << endl ;
         cout << " " << endl ;
         cout << "       -n N      : Only run test with sequential number N instead of full suite of tests" << endl ;
         cout << "       -c        : dump file stressRooStats_DEBUG.root to which results of both current result and reference for each failed test are written" << endl ;
         cout << "       -mc       : memory check mode, no regression test are performed. Set this flag when running with valgrind" << endl ;
         cout << "       -vs       : Use vector-based storage for all datasets (default is tree-based storage)" << endl ;
         cout << "       -v/-vv    : set verbose mode (show result of each regression test) or very verbose mode (show all roofit output as well)" << endl ;
         cout << "       -d N      : set ROOT gDebug flag to N" << endl ;
         cout << " " << endl ;
         return 0 ;
      }

   }

   if (doWrite && refFileName.find("http:") == 0) {

      // Locate file name part in URL and update refFileName accordingly
      char* buf = new char[refFileName.size() + 1];
      strcpy(buf, refFileName.c_str());
      char *ptr = strrchr(buf, '/');
      if (!ptr) {
         ptr = strrchr(buf, ':');
      }
      refFileName = ptr + 1;
      delete[] buf;

      cout << "stressRooStats: WARNING running in write mode, but reference file is web file, writing local file instead: " << refFileName << endl;
   }

   // Disable caching of complex error function calculation, as we don't
   // want to write out the cache file as part of the validation procedure
   RooMath::cacheCERF(kFALSE) ;

   gBenchmark = new TBenchmark();
   stressRooStats(refFileName.c_str(), doWrite, verbose, oneTest, dryRun, doDump, doTreeStore);
   return 0;
}

#endif
