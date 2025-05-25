// @(#)root/roofitcore:$name:  $:$id$
// Authors: Wouter Verkerke  November 2007

#include "RooGlobalFunc.h"
#include "RooMsgService.h"
#include "RooRandom.h"
#include "RooTrace.h"

#include "Math/MinimizerOptions.h"

#include "TWebFile.h"
#include "TSystem.h"
#include "TString.h"
#include "TStopwatch.h"
#include "TROOT.h"
#include "TLine.h"
#include "TFile.h"
#include "TClass.h"
#include "TBenchmark.h"

#include <string>
#include <list>
#include <iostream>
#include <cmath>

using std::string, std::list;
using namespace RooFit;

//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*//
//                                                                           //
// RooFit Examples, Wouter Verkerke                                          //
//                                                                           //
//                                                                           //
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*_*//

int stressRooFit(const char *refFile, bool writeRef, int doVerbose, int oneTest, bool dryRun);

////////////////////////////////////////////////////////////////////////////////
/// Print test program number and its title

void StatusPrint(int id, const TString &title, int status)
{
   const int kMAX = 65;
   Char_t number[4];
   snprintf(number, 4, "%2d", id);
   TString header = TString("Test ") + number + " : " + title;
   const int nch = header.Length();
   for (int i = nch; i < kMAX; i++)
      header += '.';
   std::cout << header << (status > 0 ? "OK" : (status < 0 ? "SKIPPED" : "FAILED")) << std::endl;
}

#include "stressRooFit_tests.h"

////////////////////////////////////////////////////////////////////////////////

int stressRooFit(const char *refFile, bool writeRef, int doVerbose, int oneTest, bool dryRun, bool doDump,
                 bool doTreeStore)
{
   int retVal = 0;
   // Save memory directory location
   RooUnitTest::setMemDir(gDirectory);

   if (doTreeStore) {
      RooAbsData::setDefaultStorageType(RooAbsData::Tree);
   }

   TFile *fref = nullptr;
   if (!dryRun) {
      if (TString(refFile).Contains("http:")) {
         if (writeRef) {
            std::cout << "stressRooFit ERROR: reference file must be local file in writing mode" << std::endl;
            return 1;
         }
         TFile::SetCacheFileDir(".");
         fref = TFile::Open(refFile, "CACHEREAD");
         // std::cout << "using WEB file " << refFile << std::endl;
      } else {
         fref = TFile::Open(refFile, writeRef ? "RECREATE" : "");
         // std::cout << "using file " << refFile << std::endl;
      }
      if (fref->IsZombie()) {
         std::cout << "stressRooFit ERROR: cannot open reference file " << refFile << std::endl;
         return 1;
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

   std::cout << "******************************************************************" << std::endl;
   std::cout << "*  RooFit - S T R E S S suite                                    *" << std::endl;
   std::cout << "******************************************************************" << std::endl;
   std::cout << "******************************************************************" << std::endl;

   TStopwatch timer;
   timer.Start();

   list<RooUnitTest *> testList;
   testList.push_back(new TestBasic101(fref, writeRef, doVerbose));
   testList.push_back(new TestBasic102(fref, writeRef, doVerbose));
   testList.push_back(new TestBasic103(fref, writeRef, doVerbose));
   testList.push_back(new TestBasic105(fref, writeRef, doVerbose));
   testList.push_back(new TestBasic108(fref, writeRef, doVerbose));
   testList.push_back(new TestBasic109(fref, writeRef, doVerbose));
   testList.push_back(new TestBasic110(fref, writeRef, doVerbose));
   testList.push_back(new TestBasic111(fref, writeRef, doVerbose));
   testList.push_back(new TestBasic201(fref, writeRef, doVerbose));
   testList.push_back(new TestBasic202(fref, writeRef, doVerbose));
   testList.push_back(new TestBasic203(fref, writeRef, doVerbose));
   testList.push_back(new TestBasic204(fref, writeRef, doVerbose));
   testList.push_back(new TestBasic205(fref, writeRef, doVerbose));
   testList.push_back(new TestBasic208(fref, writeRef, doVerbose));
   testList.push_back(new TestBasic209(fref, writeRef, doVerbose));
   testList.push_back(new TestBasic301(fref, writeRef, doVerbose));
   testList.push_back(new TestBasic302(fref, writeRef, doVerbose));
   testList.push_back(new TestBasic303(fref, writeRef, doVerbose));
   testList.push_back(new TestBasic304(fref, writeRef, doVerbose));
   testList.push_back(new TestBasic305(fref, writeRef, doVerbose));
   testList.push_back(new TestBasic306(fref, writeRef, doVerbose));
   testList.push_back(new TestBasic307(fref, writeRef, doVerbose));
   testList.push_back(new TestBasic308(fref, writeRef, doVerbose));
   testList.push_back(new TestBasic310(fref, writeRef, doVerbose));
   testList.push_back(new TestBasic311(fref, writeRef, doVerbose));
   testList.push_back(new TestBasic312(fref, writeRef, doVerbose));
   testList.push_back(new TestBasic313(fref, writeRef, doVerbose));
   testList.push_back(new TestBasic315(fref, writeRef, doVerbose));
   testList.push_back(new TestBasic314(fref, writeRef, doVerbose));
   testList.push_back(new TestBasic316(fref, writeRef, doVerbose));
   testList.push_back(new TestBasic402(fref, writeRef, doVerbose));
   testList.push_back(new TestBasic403(fref, writeRef, doVerbose));
   testList.push_back(new TestBasic404(fref, writeRef, doVerbose));
   testList.push_back(new TestBasic405(fref, writeRef, doVerbose));
   testList.push_back(new TestBasic406(fref, writeRef, doVerbose));
   testList.push_back(new TestBasic501(fref, writeRef, doVerbose));
   testList.push_back(new TestBasic599(fref, writeRef, doVerbose));
   testList.push_back(new TestBasic602(fref, writeRef, doVerbose));
   testList.push_back(new TestBasic604(fref, writeRef, doVerbose));
   testList.push_back(new TestBasic605(fref, writeRef, doVerbose));
   testList.push_back(new TestBasic606(fref, writeRef, doVerbose));
   testList.push_back(new TestBasic607(fref, writeRef, doVerbose));
   testList.push_back(new TestBasic609(fref, writeRef, doVerbose));
   testList.push_back(new TestBasic701(fref, writeRef, doVerbose));
   testList.push_back(new TestBasic702(fref, writeRef, doVerbose));
   testList.push_back(new TestBasic703(fref, writeRef, doVerbose));
   testList.push_back(new TestBasic704(fref, writeRef, doVerbose));
   testList.push_back(new TestBasic705(fref, writeRef, doVerbose));
   testList.push_back(new TestBasic706(fref, writeRef, doVerbose));
   testList.push_back(new TestBasic707(fref, writeRef, doVerbose));
   testList.push_back(new TestBasic708(fref, writeRef, doVerbose));
   testList.push_back(new TestBasic801(fref, writeRef, doVerbose));
   testList.push_back(new TestBasic802(fref, writeRef, doVerbose));
   testList.push_back(new TestBasic803(fref, writeRef, doVerbose));
   testList.push_back(new TestBasic804(fref, writeRef, doVerbose));

   std::cout << "*  Starting  S T R E S S  basic suite                            *" << std::endl;
   std::cout << "******************************************************************" << std::endl;

   if (doDump) {
      TFile fdbg("stressRooFit_DEBUG.root", "RECREATE");
   }

   gBenchmark->Start("StressRooFit");

   int i(1);
   for (RooUnitTest *unitTest : testList) {
      if (oneTest < 0 || oneTest == i) {
         if (doDump) {
            unitTest->setDebug(true);
         }
         int status = unitTest->isTestAvailable() ? unitTest->runTest() : -1;
         StatusPrint(i, unitTest->GetName(), status);
         // increment retVal for every failed test
         if (!status)
            ++retVal;
      }
      delete unitTest;
      i++;
   }

   if (dryRun) {
      RooTrace::dump();
   }

   gBenchmark->Stop("StressRooFit");

   // Print table with results
   bool UNIX = strcmp(gSystem->GetName(), "Unix") == 0;
   printf("******************************************************************\n");
   if (UNIX) {
      TString sp = gSystem->GetFromPipe("uname -a");
      sp.Resize(60);
      printf("*  SYS: %s\n", sp.Data());
      if (strstr(gSystem->GetBuildNode(), "Darwin")) {
         sp = gSystem->GetFromPipe("sw_vers -productVersion");
         sp += " Mac OS X ";
         printf("*  SYS: %s\n", sp.Data());
      }
   } else {
      const Char_t *os = gSystem->Getenv("OS");
      if (!os) {
         printf("*  SYS: Windows 95\n");
      } else {
         printf("*  SYS: %s %s \n", os, gSystem->Getenv("PROCESSOR_IDENTIFIER"));
      }
   }

   printf("******************************************************************\n");
   gBenchmark->Print("StressFit");
#ifdef __CINT__
   Double_t reftime = 186.34; // pcbrun4 interpreted
#else
   Double_t reftime = 93.59; // pcbrun4 compiled
#endif
   const Double_t rootmarks = 860 * reftime / gBenchmark->GetCpuTime("StressRooFit");

   printf("******************************************************************\n");
   printf("*  ROOTMARKS =%6.1f   *  Root%-8s  %d/%d\n", rootmarks, gROOT->GetVersion(), gROOT->GetVersionDate(),
          gROOT->GetVersionTime());
   printf("******************************************************************\n");

   printf("Time at the end of job = %f seconds\n", timer.CpuTime());

   if (fref) {
      fref->Close();
      delete fref;
   }

   delete gBenchmark;
   gBenchmark = nullptr;

   return retVal;
}

//_____________________________batch only_____________________
#ifndef __CINT__

int main(int argc, const char *argv[])
{
   bool doWrite = false;
   int doVerbose = 0;
   int oneTest = -1;
   int dryRun = false;
   bool doDump = false;
   bool doTreeStore = false;
   auto backend = RooFit::EvalBackend::Legacy();

   // string refFileName = "http://root.cern.ch/files/stressRooFit_v534_ref.root" ;
   string refFileName = "stressRooFit_ref.root";
   string minimizerName = "Minuit";

   auto verbosityOptionErrorMsg = "Multiple verbosity-related options have been passed to stressRooFit! The options "
                                  "-v, -vv, and -q are mutually exclusive.";

   // Parse command line arguments
   for (int i = 1; i < argc; i++) {
      string arg = argv[i];

      if (arg == "-b") {
         string mode = argv[++i];
         backend = RooFit::EvalBackend(mode);
         std::cout << "stressRooFit: NLL evaluation backend set to " << mode << std::endl;
      } else if (arg == "-f") {
         std::cout << "stressRooFit: using reference file " << argv[i + 1] << std::endl;
         refFileName = argv[++i];
      } else if (arg == "-w") {
         std::cout << "stressRooFit: running in writing mode to updating reference file" << std::endl;
         doWrite = true;
      } else if (arg == "-mc") {
         std::cout << "stressRooFit: running in memcheck mode, no regression tests are performed" << std::endl;
         dryRun = true;
      } else if (arg == "-ts") {
         std::cout << "stressRooFit: setting tree-based storage for datasets" << std::endl;
         doTreeStore = true;
      } else if (arg == "-min" || arg == "-minim") {
         std::cout << "stressRooFit: running using minimizer " << argv[i + 1] << std::endl;
         minimizerName = argv[++i];
      } else if (arg == "-v") {
         std::cout << "stressRooFit: running in verbose mode" << std::endl;
         if (doVerbose != 0)
            throw std::runtime_error(verbosityOptionErrorMsg);
         doVerbose = 1;
      } else if (arg == "-vv") {
         std::cout << "stressRooFit: running in very verbose mode" << std::endl;
         if (doVerbose != 0)
            throw std::runtime_error(verbosityOptionErrorMsg);
         doVerbose = 2;
      } else if (arg == "-q") {
         std::cout << "stressRooFit: running in quiet mode" << std::endl;
         if (doVerbose != 0)
            throw std::runtime_error(verbosityOptionErrorMsg);
         doVerbose = -1;
      } else if (arg == "-n") {
         std::cout << "stressRooFit: running single test " << argv[i + 1] << std::endl;
         oneTest = atoi(argv[++i]);
      } else if (arg == "-d") {
         std::cout << "stressRooFit: setting gDebug to " << argv[i + 1] << std::endl;
         gDebug = atoi(argv[++i]);
      } else if (arg == "-c") {
         std::cout << "stressRooFit: dumping comparison file for failed tests " << std::endl;
         doDump = true;
      }

      if (arg == "-h" || arg == "--help") {
         std::cout << R"(usage: stressRooFit [ options ]

       -b <mode>   : Perform every fit in the tests with the EvalBackend(<mode>) command argument, where <mode> is a string
       -f <file>   : use given reference file instead of default ("stressRooFit_ref.root")
       -w          : write reference file, instead of reading file and running comparison tests

       -n N        : Only run test with sequential number N instead of full suite of tests
       -c          : dump file stressRooFit_DEBUG.root to which results of both current result and reference for each failed test are written
       -mc         : memory check mode, no regression test are performed. Set this flag when running with valgrind
       -min <name> : minimizer name (default is Minuit, not Minuit2)
       -vs         : Use vector-based storage for all datasets (default is tree-based storage)
       -v/-vv      : set verbose mode (show result of each regression test) or very verbose mode (show all roofit output as well)
       -q          : quiet mode where errors are not logged
       -d N        : set ROOT gDebug flag to N
)";
         return 0;
      }
   }

   if (doWrite && refFileName.find("http:") == 0) {

      // Locate file name part in URL and update refFileName accordingly
      char *buf = new char[refFileName.size() + 1];
      strcpy(buf, refFileName.c_str());
      char *ptr = strrchr(buf, '/');
      if (!ptr) {
         ptr = strrchr(buf, ':');
      }
      refFileName = ptr + 1;
      delete[] buf;

      std::cout
         << "stressRooFit: WARNING running in write mode, but reference file is web file, writing local file instead: "
         << refFileName << std::endl;
   }

   // set minimizer
   ROOT::Math::MinimizerOptions::SetDefaultMinimizer(minimizerName.c_str());

   // set default BatchMode backend
   RooFit::EvalBackend::defaultValue() = backend.value();

   gBenchmark = new TBenchmark();
   int retVal = stressRooFit(refFileName.c_str(), doWrite, doVerbose, oneTest, dryRun, doDump, doTreeStore);
   return retVal;
}

////////////////////////////////////////////////////////////////////////////////

int stressRooFit()
{
   bool doWrite = false;
   int doVerbose = 0;
   int oneTest = -1;
   int dryRun = false;
   bool doDump = false;
   bool doTreeStore = false;
   // string refFileName = "http://root.cern.ch/files/stressRooFit_v534_ref.root" ;
   string refFileName = "stressRooFit_ref.root";

   // in interpreted mode, the minimizer is hardcoded to Minuit 1
   ROOT::Math::MinimizerOptions::SetDefaultMinimizer("Minuit");

   return stressRooFit(refFileName.c_str(), doWrite, doVerbose, oneTest, dryRun, doDump, doTreeStore);
}

#endif
