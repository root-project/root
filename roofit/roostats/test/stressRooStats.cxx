// @(#)root/roofitcore:$name:  $:$id$
// Authors: Wouter Verkerke  November 2007

// C/C++ headers
#include <string>
#include <list>
#include <iostream>
#include <iomanip>
#include <cmath>

// Math headers
#include "Math/MinimizerOptions.h"

// ROOT headers
#include "TWebFile.h"
#include "TSystem.h"
#include "TString.h"
#include "TStopwatch.h"
#include "TROOT.h"
#include "TLine.h"
#include "TFile.h"
#include "TClass.h"
#include "TF1.h"
#include "TBenchmark.h"

// RooFit headers
#include "RooGlobalFunc.h"
#include "RooNumIntConfig.h"
#include "RooMsgService.h"
#include "RooResolutionModel.h"
#include "RooRandom.h"
#include "RooTrace.h"

// Tests file
#include "stressRooStats_tests.h"

using std::string, std::list, std::setw, std::setfill, std::left;
using namespace RooFit;

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
   std::cout << left << setw(lineWidth) << setfill('.') << header << " "
        << (status > 0 ? "OK" : (status < 0 ? "SKIPPED" : "FAILED")) << std::endl;
}

////////////////////////////////////////////////////////////////////////////////
/// width of lines when printing test results

int stressRooStats(const char *refFile, bool writeRef, int verbose, bool allTests, bool oneTest, int testNumber,
                   bool dryRun, bool doDump, bool doTreeStore)
{
   const int lineWidth = 120;

   // Save memory directory location
   auto memDir = gDirectory;
   RooUnitTest::setMemDir(gDirectory);

   if (doTreeStore) {
      RooAbsData::setDefaultStorageType(RooAbsData::Tree);
   }

   TFile *fref = nullptr;
   if (!dryRun) {
      if (TString(refFile).Contains("http:")) {
         if (writeRef) {
            std::cout << "stressRooStats ERROR: reference file must be local file in writing mode" << std::endl;
            return -1;
         }
         fref = new TWebFile(refFile);
      } else {
         fref = new TFile(refFile, writeRef ? "RECREATE" : "");
      }
      if (fref->IsZombie()) {
         std::cout << "stressRooStats ERROR: cannot open reference file " << refFile << std::endl;
         return -1;
      }
   }

   if (dryRun) {
      // Preload singletons here so they don't show up in trace accounting
      RooNumIntConfig::defaultConfig();
      RooResolutionModel::identity();

      RooTrace::active(true);
   }

   // Add dedicated logging stream for errors that will remain active in silent mode
   RooMsgService::instance().addStream(RooFit::ERROR);

   std::cout << left << setw(lineWidth) << setfill('*') << "" << std::endl;
   std::cout << "*" << setw(lineWidth - 2) << setfill(' ') << " RooStats S.T.R.E.S.S. suite "
        << "*" << std::endl;
   std::cout << setw(lineWidth) << setfill('*') << "" << std::endl;
   std::cout << setw(lineWidth) << setfill('*') << "" << std::endl;

   TStopwatch timer;
   timer.Start();

   list<RooUnitTest *> testList;

   // 1-5 TEST PLC CONFINT SIMPLE GAUSSIAN : Confidence Level range is (0,1)
   testList.push_back(new TestProfileLikelihoodCalculator1(fref, writeRef, verbose, 0.99999)); // boundary case CL -> 1
   testList.push_back(
      new TestProfileLikelihoodCalculator1(fref, writeRef, verbose, 2 * ROOT::Math::normal_cdf(3) - 1)); // 3 sigma
   testList.push_back(
      new TestProfileLikelihoodCalculator1(fref, writeRef, verbose, 2 * ROOT::Math::normal_cdf(2) - 1)); // 2 sigma
   testList.push_back(
      new TestProfileLikelihoodCalculator1(fref, writeRef, verbose, 2 * ROOT::Math::normal_cdf(1) - 1)); // 1 sigma
   testList.push_back(new TestProfileLikelihoodCalculator1(fref, writeRef, verbose, 0.00001)); // boundary case CL -> 0

   // 6-10 TEST PLC CONFINT SIMPLE POISSON : Observed value range is [0,1000]
   testList.push_back(new TestProfileLikelihoodCalculator2(fref, writeRef, verbose, 0)); // boundary Poisson value (0)
   testList.push_back(new TestProfileLikelihoodCalculator2(fref, writeRef, verbose, 1));
   testList.push_back(new TestProfileLikelihoodCalculator2(fref, writeRef, verbose, 5));
   testList.push_back(new TestProfileLikelihoodCalculator2(fref, writeRef, verbose, 100));
   testList.push_back(new TestProfileLikelihoodCalculator2(fref, writeRef, verbose, 800)); // boundary Poisson value

   // 11-13 TEST PLC CONFINT PRODUCT POISSON : Observed value range is [0,30] for x=s+b and [0,80] for y=2*s*1.2^beta
   testList.push_back(new TestProfileLikelihoodCalculator3(fref, writeRef, verbose, 10, 30));
   testList.push_back(new TestProfileLikelihoodCalculator3(fref, writeRef, verbose, 20, 25));
   testList.push_back(
      new TestProfileLikelihoodCalculator3(fref, writeRef, verbose, 15, 20, 2 * ROOT::Math::normal_cdf(2) - 1));

   // 14 TEST PLC HYPOTEST ON/OFF MODEL
   testList.push_back(new TestProfileLikelihoodCalculator4(fref, writeRef, verbose));

   // 15-18 TEST BC CONFINT CENTRAL SIMPLE POISSON : Observed value range is [0,100]
   testList.push_back(new TestBayesianCalculator1(fref, writeRef, verbose, 1));
   testList.push_back(new TestBayesianCalculator1(fref, writeRef, verbose, 3));
   testList.push_back(new TestBayesianCalculator1(fref, writeRef, verbose, 10));
   testList.push_back(new TestBayesianCalculator1(fref, writeRef, verbose, 50));

   // 19 TEST BC CONFINT SHORTEST SIMPLE POISSON
   testList.push_back(new TestBayesianCalculator2(fref, writeRef, verbose));

   // 20-22 TEST BC CONFINT CENTRAL PRODUCT POISSON : Observed value range is [0,30] for x=s+b and [0,80] for
   // y=2*s*1.2^beta
   testList.push_back(new TestBayesianCalculator3(fref, writeRef, verbose, 10, 30));
   testList.push_back(new TestBayesianCalculator3(fref, writeRef, verbose, 20, 25));
   testList.push_back(new TestBayesianCalculator3(fref, writeRef, verbose, 15, 20, 2 * ROOT::Math::normal_cdf(2) - 1));

   // 23-25 TEST MCMCC CONFINT PRODUCT POISSON : Observed value range is [0,30] for x=s+b and [0,80] for y=2*s*1.2^beta
   testList.push_back(new TestMCMCCalculator(fref, writeRef, verbose, 10, 30));
   testList.push_back(new TestMCMCCalculator(fref, writeRef, verbose, 20, 25));
   testList.push_back(new TestMCMCCalculator(fref, writeRef, verbose, 15, 20, 2 * ROOT::Math::normal_cdf(2) - 1));

   // 26 TEST ZBI SIGNIFICANCE
   testList.push_back(new TestZBi(fref, writeRef, verbose));

   // 27-31 TEST PLC VS AC SIGNIFICANCE : Observed value range is [0,300] for on source and [0,1100] for off-source; tau
   // has the range [0.1,5.0]
   testList.push_back(new TestHypoTestCalculator1(fref, writeRef, verbose, 150, 100, 1.0));
   testList.push_back(new TestHypoTestCalculator1(fref, writeRef, verbose, 200, 100, 1.0));
   testList.push_back(new TestHypoTestCalculator1(fref, writeRef, verbose, 105, 100, 1.0));
   testList.push_back(new TestHypoTestCalculator1(fref, writeRef, verbose, 150, 10, 0.1));
   testList.push_back(new TestHypoTestCalculator1(fref, writeRef, verbose, 150, 400, 4.0));

   // 32-36 TEST HTC SIGNIFICANCE
   testList.push_back(new TestHypoTestCalculator2(fref, writeRef, verbose, kAsymptotic));
   testList.push_back(new TestHypoTestCalculator2(fref, writeRef, verbose, kFrequentist, kSimpleLR));
   testList.push_back(new TestHypoTestCalculator2(fref, writeRef, verbose, kFrequentist, kRatioLR));
   testList.push_back(new TestHypoTestCalculator2(fref, writeRef, verbose, kFrequentist, kProfileLROneSidedDiscovery));
   testList.push_back(new TestHypoTestCalculator2(fref, writeRef, verbose, kHybrid, kProfileLROneSidedDiscovery));

   // 37-43 TEST HTI PRODUCT POISSON : Observed value range is [0,30] for x=s+b and [0,80] for y=2*s*1.2^beta
   testList.push_back(new TestHypoTestInverter1(fref, writeRef, verbose, kAsymptotic, kProfileLR, 10, 30));
   testList.push_back(new TestHypoTestInverter1(fref, writeRef, verbose, kAsymptotic, kProfileLR, 20, 25));
   testList.push_back(new TestHypoTestInverter1(fref, writeRef, verbose, kAsymptotic, kProfileLR, 15, 20));
   testList.push_back(new TestHypoTestInverter1(fref, writeRef, verbose, kFrequentist, kProfileLR, 10, 30));
   testList.push_back(new TestHypoTestInverter1(fref, writeRef, verbose, kFrequentist, kProfileLR, 20, 25));
   testList.push_back(new TestHypoTestInverter1(fref, writeRef, verbose, kFrequentist, kProfileLR, 15, 20));
   testList.push_back(new TestHypoTestInverter1(fref, writeRef, verbose, kHybrid, kProfileLR, 10, 30));

   // 44-48 TEST HTI S+B+E POISSON : Observed value range is [0,50] for x = e*s+b
   testList.push_back(new TestHypoTestInverter2(fref, writeRef, verbose, kAsymptotic, kProfileLROneSided, 10, 0.95));
   testList.push_back(new TestHypoTestInverter2(fref, writeRef, verbose, kAsymptotic, kProfileLROneSided, 20));
   //    testList.push_back(new TestHypoTestInverter2(fref, writeRef, verbose, kFrequentist, kSimpleLR, 10));
   //    testList.push_back(new TestHypoTestInverter2(fref, writeRef, verbose, kFrequentist, kSimpleLR, 20));
   testList.push_back(new TestHypoTestInverter2(fref, writeRef, verbose, kFrequentist, kRatioLR, 10, 0.95));
   testList.push_back(new TestHypoTestInverter2(fref, writeRef, verbose, kFrequentist, kProfileLROneSided, 10, 0.95));
   testList.push_back(new TestHypoTestInverter2(fref, writeRef, verbose, kHybrid, kSimpleLR, 10, 0.95));

   TString suiteType = TString::Format(
      " Starting S.T.R.E.S.S. %s",
      allTests ? "full suite" : (oneTest ? TString::Format("test %d", testNumber).Data() : "basic suite"));

   std::cout << "*" << setw(lineWidth - 3) << setfill(' ') << suiteType << " *" << std::endl;
   std::cout << setw(lineWidth) << setfill('*') << "" << std::endl;

   if (doDump) {
      TFile fdbg("stressRooStats_DEBUG.root", "RECREATE");
   }

   gBenchmark->Start("stressRooStats");

   int nFailed = 0;
   {
      int i;
      list<RooUnitTest *>::iterator iter;

      if (oneTest && (testNumber <= 0 || (UInt_t)testNumber > testList.size())) {
         std::cout << "Tests are numbered from 1 to " << testList.size() << std::endl;
      } else {
         for (iter = testList.begin(), i = 1; iter != testList.end(); iter++, i++) {
            if (!oneTest || testNumber == i) {
               if (doDump) {
                  (*iter)->setDebug(true);
               }
               int status = (*iter)->isTestAvailable() ? (*iter)->runTest() : -1;
               StatusPrint(i, (*iter)->GetName(), status, lineWidth);
               if (!status)
                  nFailed++; // do not count the skipped tests
            }
            delete *iter;
         }
      }
   }

   if (dryRun) {
      RooTrace::dump();
   }

   gBenchmark->Stop("stressRooStats");

   // Print table with results
   bool UNIX = strcmp(gSystem->GetName(), "Unix") == 0;
   std::cout << setw(lineWidth) << setfill('*') << "" << std::endl;
   if (UNIX) {
      TString sp = gSystem->GetFromPipe("uname -a");
      std::cout << "* SYS: " << sp << std::endl;
      if (strstr(gSystem->GetBuildNode(), "Darwin")) {
         sp = gSystem->GetFromPipe("sw_vers -productVersion");
         sp += " Mac OS X ";
         std::cout << "* SYS: " << sp << std::endl;
      }
   } else {
      const Char_t *os = gSystem->Getenv("OS");
      if (!os) {
         std::cout << "*  SYS: Windows 95" << std::endl;
      } else {
         std::cout << "*  SYS: " << os << " " << gSystem->Getenv("PROCESSOR_IDENTIFIER") << std::endl;
      }
   }

   std::cout << setw(lineWidth) << setfill('*') << "" << std::endl;
   gBenchmark->Print("stressRooStats");
#ifdef __CINT__
   Double_t reftime = 186.34; // pcbrun4 interpreted
#else
   Double_t reftime = 93.59; // pcbrun4 compiled
#endif
   const Double_t rootmarks = 860 * reftime / gBenchmark->GetCpuTime("stressRooStats");

   std::cout << setw(lineWidth) << setfill('*') << "" << std::endl;
   std::cout << TString::Format("*  ROOTMARKS = %6.1f  *  Root %-8s %d/%d", rootmarks, gROOT->GetVersion(),
                           gROOT->GetVersionDate(), gROOT->GetVersionTime())
        << std::endl;
   std::cout << setw(lineWidth) << setfill('*') << "" << std::endl;

   // NOTE: The function TStopwatch::CpuTime() calls Tstopwatch::Stop(), so you do not need to stop the timer
   // separately.
   std::cout << "Time at the end of job = " << timer.CpuTime() << " seconds" << std::endl;

   if (fref) {
      fref->Close();
      delete fref;
   }

   delete gBenchmark;
   gBenchmark = nullptr;

   // Some of the object are multiple times in the list, let's make sure they
   // are not deleted twice.
   // The addition of memDir to the list of Cleanups is not needed if it already
   // there, for example if memDir is gROOT.
   bool needCleanupAdd = nullptr == gROOT->GetListOfCleanups()->FindObject(memDir->GetList());
   if (needCleanupAdd)
      gROOT->GetListOfCleanups()->Add(memDir->GetList());

   memDir->GetList()->Delete("slow");

   if (needCleanupAdd)
      gROOT->GetListOfCleanups()->Remove(memDir->GetList());

   return nFailed;
}

//_____________________________batch only_____________________
#ifndef __CINT__

int main(int argc, const char *argv[])
{
   bool doWrite = false;
   int verbose = 0;
   bool allTests = false;
   bool oneTest = false;
   int testNumber = 0;
   bool dryRun = false;
   bool doDump = false;
   bool doTreeStore = false;
   auto backend = RooFit::EvalBackend::Legacy();

   // string refFileName = "http://root.cern/files/stressRooStats_v534_ref.root" ;
   string refFileName = "stressRooStats_ref.root";
   string minimizerName = "Minuit";

   // Parse command line arguments
   for (int i = 1; i < argc; i++) {
      string arg = argv[i];

      if (arg == "-b") {
         std::string mode = argv[++i];
         backend = RooFit::EvalBackend(mode);
         std::cout << "stressRooStats: NLL evaluation backend set to " << mode << std::endl;
      } else if (arg == "-f") {
         std::cout << "stressRooStats: using reference file " << argv[i + 1] << std::endl;
         refFileName = argv[++i];
      } else if (arg == "-w") {
         std::cout << "stressRooStats: running in writing mode to update reference file" << std::endl;
         doWrite = true;
      } else if (arg == "-mc") {
         std::cout << "stressRooStats: running in memcheck mode, no regression tests are performed" << std::endl;
         dryRun = true;
      } else if (arg == "-min" || arg == "-minim") {
         std::cout << "stressRooStats: running using minimizer " << argv[i + 1] << std::endl;
         minimizerName = argv[++i];
      } else if (arg == "-ts") {
         std::cout << "stressRooStats: setting tree-based storage for datasets" << std::endl;
         doTreeStore = true;
      } else if (arg == "-v") {
         std::cout << "stressRooStats: running in verbose mode" << std::endl;
         verbose = 1;
      } else if (arg == "-vv") {
         std::cout << "stressRooStats: running in very verbose mode" << std::endl;
         verbose = 2;
      } else if (arg == "-vvv") {
         std::cout << "stressRooStats: running in very very verbose mode" << std::endl;
         verbose = 3;
      } else if (arg == "-a") {
         std::cout << "stressRooStats: deploying full suite of tests" << std::endl;
         allTests = true;
      } else if (arg == "-n") {
         std::cout << "stressRooStats: running single test" << std::endl;
         oneTest = true;
         testNumber = atoi(argv[++i]);
      } else if (arg == "-d") {
         std::cout << "stressRooStats: setting gDebug to " << argv[i + 1] << std::endl;
         gDebug = atoi(argv[++i]);
      } else if (arg == "-c") {
         std::cout << "stressRooStats: dumping comparison file for failed tests " << std::endl;
         doDump = true;
      } else if (arg == "-h" || arg == "--help") {
         std::cout << R"(usage: stressRooStats [ options ]

       -b <mode>   : Perform every fit in the tests with the EvalBackend(<mode>) command argument, where <mode> is a string
       -f <file>   : use given reference file instead of default ("stressRooStats_ref.root")
       -w          : write reference file, instead of reading file and running comparison tests
       -n N        : only run test with sequential number N
       -a          : run full suite of tests (default is basic suite); this overrides the -n single test option
       -c          : dump file stressRooStats_DEBUG.root to which results of both current result and reference for each failed test are written
       -mc         : memory check mode, no regression test are performed. Set this flag when running with valgrind
       -min <name> : minimizer name (default is Minuit, not Minuit2)
       -vs         : use vector-based storage for all datasets (default is tree-based storage)
       -v/-vv      : set verbose mode (show result of each regression test) or very verbose mode (show all roofit output as well)
       -d N        : set ROOT gDebug flag to N
)";
         return 0;
      }
   }

   //    if (doWrite && refFileName.find("http:") == 0) {

   //       // Locate file name part in URL and update refFileName accordingly
   //       char* buf = new char[refFileName.size() + 1];
   //       strcpy(buf, refFileName.c_str());
   //       char *ptr = strrchr(buf, '/');
   //       if (!ptr) ptr = strrchr(buf, ':');
   //       refFileName = ptr + 1;
   //       delete[] buf;

   //       std::cout << "stressRooStats: WARNING running in write mode, but reference file is web file, writing local file
   //       instead: "
   //            << refFileName << std::endl;
   //    }

   // set minimizer
   ROOT::Math::MinimizerOptions::SetDefaultMinimizer(minimizerName.c_str());

   // set default NLL backend
   RooFit::EvalBackend::defaultValue() = backend.value();

   gBenchmark = new TBenchmark();
   return stressRooStats(refFileName.c_str(), doWrite, verbose, allTests, oneTest, testNumber, dryRun, doDump,
                         doTreeStore);
}

////////////////////////////////////////////////////////////////////////////////

int stressRooStats()
{
   bool doWrite = false;
   int verbose = 0;
   bool allTests = false;
   bool oneTest = false;
   int testNumber = 0;
   bool dryRun = false;
   bool doDump = false;
   bool doTreeStore = false;
   string refFileName = "stressRooStats_ref.root";

   // in interpreted mode, the minimizer is hardcoded to Minuit 1
   ROOT::Math::MinimizerOptions::SetDefaultMinimizer("Minuit");

   return stressRooStats(refFileName.c_str(), doWrite, verbose, allTests, oneTest, testNumber, dryRun, doDump,
                         doTreeStore);
}

#endif
