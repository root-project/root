// ************************************************************************* //
// *                                                                       * //
// *                        s t r e s s P r o o f                          * //
// *                                                                       * //
// * This file contains a set of test of PROOF related functionality.      * //
// * The tests can be run as a standalone program or with the interpreter. * //
// * To run as a standalone program:                                       * //
// *                                                                       * //
// *  $ cd $ROOTSYS/test                                                   * //
// *  $ make stressProof                                                   * //
// *  $ ./stressProof [-h] [-n <wrks>] [-v[v[v]]] [-l logfile]             * //
// *                  [-dyn] [-ds] [-t testnum] [-h1 h1src] [master]       * //
// *                                                                       * //
// * Optional arguments:                                                   * //
// *   -h          show help info                                          * //
// *   master      entry point of the cluster where to run the test        * //
// *               in the form '[user@]host.domain[:port]';                * //
// *               default 'localhost:40000'                               * //
// *   -n wrks     number of workers to be started when running on the     * //
// *               local host; default is the nuber of local cores         * //
// *   -d level    verbosity level [1]                                     * //
// *   -l logfile  file where to redirect the processing logs; default is  * //
// *               a temporary file deleted at the end of the test; in     * //
// *               case of success                                         * //
// *   -dyn        run the test in dynamic startup mode                    * //
// *   -ds         force the dataset test if skipped by default            * //
// *   -t testnum  run only test 'testnum' and the tests from which it     * //
// *               depends                                                 * //
// *   -h1 h1src   specify a location for the H1 files;                    * //
// *               use h1src="download" to download them to a temporary    * //
// *               location; by default the files are read directly from   * //
// *               the ROOT http server; however this may give failures if * //
// *               the connection is slow                                  * //
// *   -punzip     use parallel unzipping for data-driven processing       * //
// *                                                                       * //
// * To run interactively:                                                 * //
// * $ root                                                                * //
// * root[] .L stressProof.cxx+                                            * //
// * root[] stressProof(master, wrks, verbose, logfile, dyn, \             * //
// *                                         skipds, testnum, h1src)       * //
// *                                                                       * //
// * The arguments have the same meaning as above except for               * //
// *     verbose [Int_t]   increasing verbosity (0 == minimal)             * //
// *     dyn     [Bool_t]  if kTRUE run in dynamic startup mode            * //
// *     skipds  [Bool_t]  if kTRUE the dataset related tests are skipped  * //
// *                                                                       * //
// *                                                                       * //
// * The successful output looks like this:                                * //
// *                                                                       * //
// *  ******************************************************************   * //
// *  *  Starting  P R O O F - S T R E S S suite                       *   * //
// *  ******************************************************************   * //
// *  *  Log file: /tmp/ProofStress_XrcwBe                                 * //
// *  ******************************************************************   * //
// *   Test  1 : Open a session ................................... OK *   * //
// *   Test  2 : Get session logs ................................. OK *   * //
// *   Test  3 : Simple random number generation .................. OK *   * //
// *   Test  4 : Dataset handling with H1 files ................... OK *   * //
// *   Test  5 : H1: chain processing ............................. OK *   * //
// *   Test  6 : H1: file collection processing ................... OK *   * //
// *   Test  7 : H1: file collection, TPacketizer ................. OK *   * //
// *   Test  8 : H1: by-name processing ........................... OK *   * //
// *   Test  9 : H1: multi dataset processing ..................... OK *   * //
// *   Test 10 : H1: multi dataset and entry list ................. OK *   * //
// *   Test 11 : Package management with 'event' .................. OK *   * //
// *   Test 12 : Package argument passing ......................... OK *   * //
// *   Test 13 : Simple 'event' generation ........................ OK *   * //
// *   Test 14 : Input data propagation ........................... OK *   * //
// *   Test 15 : H1, Simple: async mode :.......................... OK *   * //
// *   Test 16 : Admin functionality .............................. OK *   * //
// *   Test 17 : Dynamic sub-mergers functionality ................ OK *   * //
// *  * All registered tests have been passed  :-)                     *   * //
// *  ******************************************************************   * //
// *                                                                       * //
// * The application redirects the processing logs to a log file which is  * //
// * normally deleted at the end of a successful run; if the test fails    * //
// * the caller is asked if she/he wants to keep the log file; if the      * //
// * specifies a log file path of her/his choice, the log file is never    * //
// * deleted.                                                              * //
// *                                                                       * //
// * SKIPPED means that the test cannot be run.                            * //
// *                                                                       * //
// * New tests can be easily added by providing a function performing the  * //
// * test and a name for the test; see examples below.                     * //
// *                                                                       * //
// * It is also possible to trigger the automatic PROOF valgrind setup by  * //
// * means of the env GETPROOF_VALGRIND.                                   * //
// * E.g. to run the master in valgrind do                                 * //
// *                                                                       * //
// *     $ export GETPROOF_VALGRIND="valgrind=master"                      * //
// * or                                                                    * //
// *     $ export GETPROOF_VALGRIND="valgrind=workers"                     * //
// *                                                                       * //
// * before running stressProof. The syntax is the same as for standard    * //
// * PROOF valgrind runs. See                                              * //
// *   http://root.cern.ch/drupal/content/running-proof-query-valgrind     * //
// *                                                                       * //
// ************************************************************************* //

#include <stdio.h>
#include <stdlib.h>

#include "Getline.h"
#include "TApplication.h"
#include "TChain.h"
#include "TFile.h"
#include "TFileCollection.h"
#include "TFileInfo.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TList.h"
#include "TMacro.h"
#include "TMap.h"
#include "TMath.h"
#include "TNamed.h"
#include "TParameter.h"
#include "TProof.h"
#include "TProofLog.h"
#include "TProofMgr.h"
#include "TQueryResult.h"
#include "TStopwatch.h"
#include "TString.h"
#include "TSystem.h"
#include "TROOT.h"

#include "../tutorials/proof/getProof.C"

static const char *urldef = "proof://localhost:40000";
static TString gtutdir;
static TString gsandbox;
static Int_t gverbose = 1;
static TString glogfile;
static Int_t gpoints = 0;
static Int_t totpoints = 53;
static RedirectHandle_t gRH;
static RedirectHandle_t gRHAdmin;
static Double_t gH1Time = 0;
static Double_t gSimpleTime = 0;
static Int_t gH1Cnt = 0;
static Int_t gSimpleCnt = 0;
static TStopwatch gTimer;
static Bool_t gTimedOut = kFALSE;
static Bool_t gDynamicStartup = kFALSE;
static Bool_t gSkipDataSetTest = kTRUE;
static Bool_t gUseParallelUnzip = kFALSE;
static TString gh1src("http://root.cern.ch/files/h1");
static Bool_t gh1ok = kTRUE;
static const char *gh1file[] = { "dstarmb.root", "dstarp1a.root", "dstarp1b.root", "dstarp2.root" };

// The selectors
static TList gSelectors;
static TString gH1Sel("$ROOTSYS/tutorials/tree/h1analysis.C");
static TString gEventSel("$ROOTSYS/tutorials/proof/ProofEvent.C");
static TString gSimpleSel("$ROOTSYS/tutorials/proof/ProofSimple.C");
static TString gTestsSel("$ROOTSYS/tutorials/proof/ProofTests.C");

void stressProof(const char *url = "proof://localhost:40000",
                 Int_t nwrks = -1, Int_t verbose = 1,
                 const char *logfile = 0, Bool_t dyn = kFALSE,
                 Bool_t skipds = kTRUE, Int_t test = -1,
                 const char *h1pfx = 0);

//_____________________________batch only_____________________
#ifndef __CINT__
int main(int argc,const char *argv[])
{

   // Request for help?
   if (argc > 1 && !strcmp(argv[1],"-h")) {
      printf(" \n");
      printf(" PROOF test suite\n");
      printf(" \n");
      printf(" Usage:\n");
      printf(" \n");
      printf(" $ ./stressProof [-h] [-n <wrks>] [-v[v[v]]] [-l logfile] [-dyn] [-ds] [-t testnum] [-h1 h1src] [-g] [master]\n");
      printf(" \n");
      printf(" Optional arguments:\n");
      printf("   -h            prints this menu\n");
      printf("   master        entry point of the cluster where to run the test\n");
      printf("                 in the form '[user@]host.domain[:port]'; default 'localhost:40000'\n");
      printf("   -n wrks       number of workers to be started when running on the local host;\n");
      printf("                 default is the nuber of local cores\n");
      printf("   -d level      verbosity level [1]\n");
      printf("   -l logfile    file where to redirect the processing logs; must be writable;\n");
      printf("                 default is a temporary file deleted at the end of the test\n");
      printf("                 in case of success\n");
      printf("   -dyn          run the test in dynamicStartup mode\n");
      printf("   -ds           force the dataset test if skipped by default\n");
      printf("   -t testnum    run only test 'testnum' and the tests from which it depends\n");
      printf("   -h1 h1src     specify a location for the H1 files; use h1src=\"download\" to download\n");
      printf("                 to a temporary location; by default the files are read directly from the\n");
      printf("                 ROOT http server; however this may give failures if the connection is slow\n");
      printf("   -punzip       use parallel unzipping for data-driven processing.\n");
      printf("   -g            enable graphics; default is to run in text mode.\n");
      printf(" \n");
      gSystem->Exit(0);
   }

   // Parse options
   const char *url = 0;
   Int_t nWrks = -1;
   Int_t verbose = 1;
   Int_t test = -1;
   Bool_t enablegraphics = kFALSE;
   const char *logfile = 0;
   const char *h1src = 0;
   Int_t i = 1;
   while (i < argc) {
      if (!strcmp(argv[i],"-h")) {
         // Ignore if not first argument
         i++;
      } else if (!strcmp(argv[i],"-n")) {
         if (i+1 == argc || argv[i+1][0] == '-') {
            printf(" -n should be followed by the number of workers: ignoring \n");
            i++;
         } else  {
            nWrks = atoi(argv[i+1]);
            i += 2;
         }
      } else if (!strcmp(argv[i],"-d")) {
         if (i+1 == argc || argv[i+1][0] == '-') {
            printf(" -d should be followed by the debug level: ignoring \n");
            i++;
         } else  {
            verbose = atoi(argv[i+1]);
            i += 2;
         }
      } else if (!strcmp(argv[i],"-l")) {
         if (i+1 == argc || argv[i+1][0] == '-') {
            printf(" -l should be followed by a path: ignoring \n");
            i++;
         } else { 
            logfile = argv[i+1];
            i += 2;
         }
      } else if (!strncmp(argv[i],"-v",2)) {
         // For backward compatibility
         if (!strcmp(argv[i],"-v")) verbose = 1;
         if (!strncmp(argv[i],"-vv",3)) verbose = 2;
         i++;
      } else if (!strncmp(argv[i],"-dyn",4)) {
         gDynamicStartup = kTRUE;
         i++;
      } else if (!strncmp(argv[i],"-ds",3)) {
         gSkipDataSetTest = kFALSE;
         i++;
      } else if (!strncmp(argv[i],"-punzip",7)) {
         gUseParallelUnzip = kTRUE;
         i++;
      } else if (!strcmp(argv[i],"-t")) {
         if (i+1 == argc || argv[i+1][0] == '-') {
            printf(" -t should be followed by a number: ignoring \n");
            i++;
         } else { 
            test = atoi(argv[i+1]);
            i += 2;
         }
      } else if (!strcmp(argv[i],"-h1")) {
         if (i+1 == argc || argv[i+1][0] == '-') {
            printf(" -h1 should be followed by a prefix: ignoring \n");
            i++;
         } else { 
            h1src = argv[i+1];
            i += 2;
         }
      } else if (!strncmp(argv[i],"-g",2)) {
         enablegraphics = kTRUE;
         i++;
      } else {
         url = argv[i];
         i++;
      }
   }
   // Use defaults where required
   if (!url) url = urldef;

   // Enable graphics if required
   if (enablegraphics)
      new TApplication("stressProof", 0, 0);

   stressProof(url, nWrks, verbose, logfile, gDynamicStartup, gSkipDataSetTest, test, h1src);

   gSystem->Exit(0);
}
#endif

//_____________________________________________________________________________
Int_t PutPoint()
{
   // Print one '.' and count it
   printf(".");
   return ++gpoints;
}

//______________________________________________________________________________
void PrintStressProgress(Long64_t total, Long64_t processed, Float_t, Long64_t)
{
   // Print some progress information

   gSystem->RedirectOutput(0, 0, &gRH);

   char pc[2] = { '.', ':'};
   static int lstpc = 1;

   int ns = 0;
   if (processed < total) {
      ns += 6;
      fprintf(stderr, "%c %2.0f %%", pc[(lstpc++)%2],
                                    (total ? ((100.0*processed)/total) : 100.0));
   }
   while (ns--) fprintf(stderr, "\b");

   gSystem->RedirectOutput(glogfile, "a", &gRH);
}

//______________________________________________________________________________
void CleanupSelector(const char *selpath)
{
   // Remove all non source files associated with seletor at path 'selpath'

   if (!selpath) return;

   TString dirpath(gSystem->DirName(selpath));
   if (gSystem->AccessPathName(dirpath)) return;
   TString selname(gSystem->BaseName(selpath));
   selname.ReplaceAll(".C", "_C");
   void *dirp = gSystem->OpenDirectory(dirpath);
   if (!dirp) return;
   TString fn;
   const char *e = 0;
   while ((e = gSystem->GetDirEntry(dirp))) {
      if (!strncmp(e, selname.Data(), selname.Length())) {
         // Cleanup this entry
         fn.Form("%s/%s", dirpath.Data(), e);
         gSystem->Unlink(fn);
      }
   }
}

//_____________________________________________________________________________
void AssertParallelUnzip()
{
   // Set the parallel unzip option

   if (gUseParallelUnzip) {
      gProof->SetParameter("PROOF_UseParallelUnzip", (Int_t)1);
   } else {
      gProof->SetParameter("PROOF_UseParallelUnzip", (Int_t)0);
   }
}

//
// Auxilliary classes for testing
//
typedef Int_t (*ProofTestFun_t)(void *);
class ProofTest : public TNamed {
private:
   Int_t           fSeq;  // Sequential number for the test
   ProofTestFun_t  fFun;  // Function to be executed for the test
   void           *fArgs; // Arguments to be passed to the function
   TString         fDeps;  // Test dependencies, e.g. "1,3"
   TString         fSels;  // Selectors used, e.g. "h1analysis,ProofSimple"
   Int_t           fDepFrom; // Index for looping over deps
   Int_t           fSelFrom; // Index for looping over selectors
   Bool_t          fEnabled; // kTRUE if this test is enabled

public:
   ProofTest(const char *n, Int_t seq, ProofTestFun_t f, void *a = 0, const char *d = "", const char *sel = "")
           : TNamed(n,""), fSeq(seq), fFun(f), fArgs(a),
             fDeps(d), fSels(sel), fDepFrom(0), fSelFrom(0), fEnabled(kTRUE) { }
   virtual ~ProofTest() { }

   void   Disable() { fEnabled = kFALSE; }
   void   Enable() { fEnabled = kTRUE; }
   Bool_t IsEnabled() const { return fEnabled; }

   Int_t  NextDep(Bool_t reset = kFALSE);
   Int_t  NextSel(TString &sel, Bool_t reset = kFALSE);
   Int_t  Num() const { return fSeq; }

   Int_t  Run();
};

//
// Timer to stop asynchronous actions
//
class TTimeOutTimer : public TTimer {
public:
   TTimeOutTimer(Long_t ms);
   Bool_t  Notify();
};

TTimeOutTimer::TTimeOutTimer(Long_t ms)
              : TTimer(ms, kTRUE)
{
   //constructor
   gSystem->AddTimer(this);
}

Bool_t TTimeOutTimer::Notify()
{
   //notifier
   gTimedOut = kTRUE;
   Remove();       // one shot only
   return kTRUE;
}
//------------------------------------------------------------------------------
//_____________________________________________________________________________
Int_t ProofTest::NextDep(Bool_t reset)
{
   // Return index of next dependency or -1 if none (or no more)
   // If reset is kTRUE, reset the internal counter before acting.

   if (reset) fDepFrom = 0;

   TString tkn;
   if (fDeps.Tokenize(tkn, fDepFrom, ",")) {
      if (tkn.IsDigit()) return tkn.Atoi();
   }
   // Not found
   return -1;
}
//_____________________________________________________________________________
Int_t ProofTest::NextSel(TString &sel, Bool_t reset)
{
   // Return index of next dependency or -1 if none (or no more)
   // If reset is kTRUE, reset the internal counter before acting.

   if (reset) fSelFrom = 0;
   if (fSels.Tokenize(sel, fSelFrom, ",")) {
      if (!sel.IsNull()) return 0;
   }
   // Not found
   return -1;
}

//_____________________________________________________________________________
Int_t ProofTest::Run()
{
   // Generic stress steering function; returns 0 on success, -1 on error

   gpoints = 0;
   printf(" Test %2d : %s ", fSeq, GetName());
   PutPoint();
   gSystem->RedirectOutput(glogfile, "a", &gRH);
   Int_t rc = (*fFun)(fArgs);
   gSystem->RedirectOutput(0, 0, &gRH);
   if (rc == 0) {
      Int_t np = totpoints - strlen(GetName()) - strlen(" OK *");
      while (np--) { printf("."); }
      printf(" OK *\n");
   } else if (rc == 1) {
      Int_t np = totpoints - strlen(GetName()) - strlen(" SKIPPED *");
      while (np--) { printf("."); }
      printf(" SKIPPED *\n");
   } else {
      Int_t np = totpoints - strlen(GetName()) - strlen(" FAILED *");
      while (np--) { printf("."); }
      printf(" FAILED *\n");
      gSystem->ShowOutput(&gRH);
   }
   // Done
   return rc;
}

// Test functions
Int_t PT_Open(void *);
Int_t PT_GetLogs(void *);
Int_t PT_Simple(void *smg = 0);
Int_t PT_H1Http(void *);
Int_t PT_H1FileCollection(void *);
Int_t PT_H1DataSet(void *);
Int_t PT_H1MultiDataSet(void *);
Int_t PT_H1MultiDSetEntryList(void *);
Int_t PT_DataSets(void *);
Int_t PT_Packages(void *);
Int_t PT_Event(void *);
Int_t PT_InputData(void *);
Int_t PT_H1SimpleAsync(void *arg);
Int_t PT_AdminFunc(void *arg);
Int_t PT_PackageArguments(void *);

// Arguments structures
typedef struct {            // Open
   const char *url;
   Int_t       nwrks;
} PT_Open_Args_t;

// Packetizer parameters
typedef struct {
   const char *fName;
   Int_t fType;
} PT_Packetizer_t;

static PT_Packetizer_t gStd_Old = { "TPacketizer", 0 };

//_____________________________________________________________________________
void stressProof(const char *url, Int_t nwrks, Int_t verbose, const char *logfile,
                 Bool_t dyn, Bool_t skipds, Int_t test, const char *h1src)
{
   printf("******************************************************************\n");
   printf("*  Starting  P R O O F - S T R E S S  suite                      *\n");
   printf("******************************************************************\n");

   // Set dynamic mode
   gDynamicStartup = (!strcmp(url,"lite://")) ? kFALSE : dyn;

   // Set verbosity
   gverbose = verbose;

   // Notify/warn about the dynamic startup option, if any
   TUrl uu(url), udef(urldef);
   Bool_t extcluster = (strcmp(uu.GetHost(), udef.GetHost()) ||
                       (uu.GetPort() != udef.GetPort())) ? kTRUE : kFALSE;
   if (gDynamicStartup && gverbose > 0) {
      // Check url
      if (extcluster) {
         printf("*   WARNING: request to run a test with per-job scheduling on    *\n");
         printf("*            an external cluster: %s .\n", url);
         printf("*            Make sure the dynamic option is set.                *\n");
         printf("******************************************************************\n");
         gDynamicStartup = kFALSE;
      } else {
         printf("*  Runnning in dynamic mode (per-job scheduling)                 *\n");
         printf("******************************************************************\n");
      }
   }

   // Dataset option
   if (!skipds) {
      gSkipDataSetTest = kFALSE;
   } else {
     gSkipDataSetTest = (!extcluster || !strcmp(url, "lite://")) ? kFALSE : kTRUE;
   }

   // Log file path
   Bool_t usedeflog = kTRUE;
   FILE *flog = 0;
   if (logfile && strlen(logfile) > 0) {
      usedeflog = kFALSE;
      glogfile = logfile;
      if (glogfile.Contains("<tmpdir>"))
         glogfile.ReplaceAll("<tmpdir>", gSystem->TempDirectory());
      if (!gSystem->AccessPathName(glogfile, kFileExists)) {
         if (!gSystem->AccessPathName(glogfile, kWritePermission)) {
            printf(" >>> Cannot write to log file %s - ignore file request\n", logfile);
            usedeflog = kTRUE;
         }
      } else {
         // Create the file
         if (!(flog = fopen(logfile, "w"))) {
            printf(" >>> Cannot create log file %s - ignore file request\n", logfile);
            usedeflog = kTRUE;
         }
      }
   }
   if (usedeflog) {
      glogfile = "ProofStress_";
      if (!(flog = gSystem->TempFileName(glogfile, gSystem->TempDirectory()))) {
         printf(" >>> Cannot create a temporary log file on %s - exit\n", gSystem->TempDirectory());
         return;
      }
      fclose(flog);
      if (gverbose > 0) {
         printf("*  Log file: %s\n", glogfile.Data());
         printf("******************************************************************\n");
      }
   }

   if (gSkipDataSetTest && gverbose > 0) {
      printf("*  Test for dataset handling (#4, #8-10) skipped                **\n");
      printf("******************************************************************\n");
   }
   if (gUseParallelUnzip && gverbose > 0) {
      printf("*  Using parallel unzip where relevant                          **\n");
      printf("******************************************************************\n");
   }
   if (!strcmp(url,"lite://") && gverbose > 0) {
      printf("*  PROOF-Lite session (tests #15 and #16 skipped)               **\n");
      printf("******************************************************************\n");
   }
   if (test > 0 && gverbose > 0) {
      if (test < 18) {
         printf("*  Running only test %2d (and related tests)                     **\n", test);
         printf("******************************************************************\n");
      } else {
         printf("*  Request for unknown test %2d : ignore                         **\n", test);
         printf("******************************************************************\n");
         test = -1;
      }
   }
   if (h1src && strlen(h1src) && gverbose > 0) {
      if (!strcmp(h1src, "download") && extcluster) {
         printf("*  External clusters: ignoring download request of H1 files\n");
         printf("******************************************************************\n");
      } else if (!gh1src.BeginsWith(h1src)) {
         printf("*  Taking H1 files from: %s\n", h1src);
         printf("******************************************************************\n");
         gh1src = h1src;
         gh1ok = kFALSE;
      }
   }
   //
   // Reset dataset settings
   gEnv->SetValue("Proof.DataSetManager","");

   //
   // Register tests
   //
   TList *testList = new TList;
   // Simple open
   PT_Open_Args_t PToa = { url, nwrks };
   testList->Add(new ProofTest("Open a session", 1, &PT_Open, (void *)&PToa));
   // Get logs
   testList->Add(new ProofTest("Get session logs", 2, &PT_GetLogs, (void *)&PToa, "1"));
   // Simple histogram generation
   testList->Add(new ProofTest("Simple random number generation", 3, &PT_Simple, 0, "1", "ProofSimple"));
   // Test of data set handling with the H1 http files
   testList->Add(new ProofTest("Dataset handling with H1 files", 4, &PT_DataSets, 0, "1"));
   // H1 analysis over HTTP (chain)
   testList->Add(new ProofTest("H1: chain processing", 5, &PT_H1Http, 0, "1", "h1analysis"));
   // H1 analysis over HTTP (file collection)
   testList->Add(new ProofTest("H1: file collection processing", 6, &PT_H1FileCollection, 0, "1", "h1analysis"));
   // H1 analysis over HTTP: classic packetizer
   testList->Add(new ProofTest("H1: file collection, TPacketizer", 7, &PT_H1FileCollection, (void *)&gStd_Old, "1", "h1analysis"));
   // H1 analysis over HTTP by dataset name
   testList->Add(new ProofTest("H1: by-name processing", 8, &PT_H1DataSet, 0, "1,4", "h1analysis"));
   // H1 analysis over HTTP by dataset name splitted in two
   testList->Add(new ProofTest("H1: multi dataset processing", 9, &PT_H1MultiDataSet, 0, "1,4", "h1analysis"));
   // H1 analysis over HTTP by dataset name
   testList->Add(new ProofTest("H1: multi dataset and entry list", 10, &PT_H1MultiDSetEntryList, 0, "1,4", "h1analysis"));
   // Test package management with 'event'
   testList->Add(new ProofTest("Package management with 'event'", 11, &PT_Packages, 0, "1"));
   // Test package argument passing
   testList->Add(new ProofTest("Package argument passing", 12, &PT_PackageArguments, 0, "1", "ProofTests"));
   // Simple event analysis
   testList->Add(new ProofTest("Simple 'event' generation", 13, &PT_Event, 0, "1", "ProofEvent"));
   // Test input data propagation (it only works in the static startup mode)
   testList->Add(new ProofTest("Input data propagation", 14, &PT_InputData, 0, "1", "ProofTests"));
   // Test asynchronous running
   testList->Add(new ProofTest("H1, Simple: async mode", 15, &PT_H1SimpleAsync, 0, "1,3,5", "h1analysis,ProofSimple"));
   // Test admin functionality
   testList->Add(new ProofTest("Admin functionality", 16, &PT_AdminFunc, 0, "1"));
   // Test merging via submergers
   Bool_t useMergers = kTRUE;
   testList->Add(new ProofTest("Dynamic sub-mergers functionality", 17, &PT_Simple, (void *)&useMergers, "1", "ProofSimple"));

   // The selectors
   gSystem->ExpandPathName(gH1Sel);
   gSystem->ExpandPathName(gEventSel);
   gSystem->ExpandPathName(gSimpleSel);
   gSystem->ExpandPathName(gTestsSel);
   gSelectors.Add(new TNamed("h1analysis", gH1Sel.Data()));
   gSelectors.Add(new TNamed("ProofEvent", gEventSel.Data()));
   gSelectors.Add(new TNamed("ProofSimple", gSimpleSel.Data()));
   gSelectors.Add(new TNamed("ProofTests", gTestsSel.Data()));
   if (gverbose > 0) printf("*  Cleaning all non-source files associated to:\n");

   // Check what to run
   ProofTest *t = 0, *treq = 0;
   TIter nxt(testList);
   if (test > 0) {
      // Disable first all the tests
      while ((t = (ProofTest *)nxt())) {
         t->Disable();
         if (t->Num() == test) treq = t;
      }
      if (!treq) {
         printf("* Test %2d not found among the registered tests - exiting        **\n", test);
         printf("******************************************************************\n");
         return;
      }
      // Enable the required tests
      Int_t tn = -1;
      while ((tn = treq->NextDep()) > 0) {
         nxt.Reset();
         while ((t = (ProofTest *)nxt())) {
            if (t->Num() == tn) {
               t->Enable();
               break;
            }
         }
      }
      // Reset associated selectors
      TString sel;
      while ((treq->NextSel(sel)) == 0) {
         TNamed *nm = (TNamed *) gSelectors.FindObject(sel.Data());
         if (nm) {
            CleanupSelector(nm->GetTitle());
            if (gverbose > 0)
               printf("*     %s   \t in %s\n", nm->GetName(), gSystem->DirName(nm->GetTitle()));
         }
      }
      // Enable the required test
      treq->Enable();
   } else {
      // Clean all the selectors
      TIter nxs(&gSelectors);
      TNamed *nm = 0;
      while ((nm = (TNamed *)nxs())) {
         CleanupSelector(nm->GetTitle());
         if (gverbose > 0)
            printf("*     %s   \t in %s\n", nm->GetName(), gSystem->DirName(nm->GetTitle()));
      }
   }
   if (gverbose > 0)
      printf("******************************************************************\n");

   // Add the ACLiC option to the selector strings
   gH1Sel += "+";
   gEventSel += "+";
   gSimpleSel += "+";
   gTestsSel += "+";

   //
   // Run the tests
   //
   Bool_t failed = kFALSE;
   nxt.Reset();
   while ((t = (ProofTest *)nxt()))
      if (t->IsEnabled()) {
         if (t->Run() < 0) {
            failed = kTRUE;
            break;
         }
      }

   // Done
   if (failed) {
      Bool_t kept = kTRUE;
      if (usedeflog && !gROOT->IsBatch()) {
         const char *answer = Getline(" Some tests failed: would you like to keep the log file (N,Y)? [Y] ");
         if (answer && (answer[0] == 'N' || answer[0] == 'n')) {
            // Remove log file
            gSystem->Unlink(glogfile);
            kept = kFALSE;
         }
      }
      if (kept)
         printf("* Log file kept at %s\n", glogfile.Data());
   } else {
      printf("* All registered tests have been passed  :-)                     *\n");
      // Remove log file if not passed by the user
      if (usedeflog)
         gSystem->Unlink(glogfile);
   }

   printf("******************************************************************\n");
   if (gProof) {
      gProof->GetStatistics((verbose > 0));
      // Reference time measured on a HP DL580 24 core (4 x Intel(R) Xeon(R) CPU X7460
      // @ 2.132 GHz, 48GB RAM, 1 Gb/s NIC) with 2 workers.
      const double reftime = 21.060;
      double rootmarks = (gProof->GetCpuTime() > 0) ? 1000 * reftime / gProof->GetCpuTime() : -1;
      printf(" ROOTMARKS = %.2f ROOT version: %s\t%s@%d\n", rootmarks, gROOT->GetVersion(),
             gROOT->GetSvnBranch(), gROOT->GetSvnRevision());
      printf("******************************************************************\n");
   }

   // If not PROOF-Lite, stop the daemon used for the test
   if (gProof && !gProof->IsLite() && !extcluster) {
      // Get the manager
      TProofMgr *mgr = gProof->GetManager();
      // Close the instance
      gProof->Close("S");
      delete gProof;
      // Delete the manager
      SafeDelete(mgr);
      // The daemon runs on a port shifted by 1
      if (killXrootdAt(uu.GetPort()+1, "xpdtut") != 0) {
         printf("+++ Warning: test daemon probably still running!\n");
      }
   }
}

//_____________________________________________________________________________
Int_t PT_H1AssertFiles(const char *h1src)
{
   // Make sure that the needed H1 files are available at 'src'
   // If 'src' is "download", the files are download under <tutdir>/h1

   if (!h1src || strlen(h1src) <= 0) {
      printf("\n >>> Test failure: src dir undefined\n");
      return -1;
   }

   // Special case
   if (!strcmp(h1src,"download")) {
      gh1src = TString::Format("%s/h1", gtutdir.Data());
      if (gSystem->AccessPathName(gh1src)) {
         if (gSystem->MakeDirectory(gh1src) != 0) {
            printf("\n >>> Test failure: could not create dir %s\n", gh1src.Data());
            return -1;
         }
      }
      // Copy the files now
      Int_t i = 0;
      for (i = 0; i < 4; i++) {
         TString src = TString::Format("http://root.cern.ch/files/h1/%s", gh1file[i]);
         TString dst = TString::Format("%s/%s", gh1src.Data(), gh1file[i]);
         if (!TFile::Cp(src, dst)) {
            printf("\n >>> Test failure: problems retrieving %s\n", src.Data());
            return -1;
         }
         gSystem->RedirectOutput(0, 0, &gRH);
         printf("%d\b", i);
         gSystem->RedirectOutput(glogfile, "a", &gRH);
      }
      // Done
      gh1ok = kTRUE;
      return 0;
   }

   // Make sure the files exist at 'src'
   Int_t i = 0;
   for (i = 0; i < 4; i++) {
      TString src = TString::Format("%s/%s", h1src, gh1file[i]);
      if (gSystem->AccessPathName(src)) {
         printf("\n >>> Test failure: file %s does not exist\n", src.Data());
         return -1;
      }
      gSystem->RedirectOutput(0, 0, &gRH);
      printf("%d\b", i);
      gSystem->RedirectOutput(glogfile, "a", &gRH);
   }
   gh1src = h1src;

   // Done
   gh1ok = kTRUE;
   return 0;
}

//_____________________________________________________________________________
Int_t PT_CheckSimple(TQueryResult *qr, Long64_t nevt, Int_t nhist)
{
   // Check the result of the ProofSimple analysis

   if (!qr) {
      printf("\n >>> Test failure: query result not found\n");
      return -1;
   }

   // Make sure the number of processed entries is the one expected
   PutPoint();
   if (qr->GetEntries() != nevt) {
      printf("\n >>> Test failure: wrong number of entries processed: %lld (expected %lld)\n",
             qr->GetEntries(), nevt);
      return -1;
   }

   // Make sure the output list is there
   PutPoint();
   TList *out = qr->GetOutputList();
   if (!out) {
      printf("\n >>> Test failure: output list not found\n");
      return -1;
   }

   // Get the histos
   PutPoint();
   TH1F **hist = new TH1F*[nhist];
   for (Int_t i=0; i < nhist; i++) {
      hist[i] = dynamic_cast<TH1F *>(out->FindObject(Form("h%d",i)));
      if (!hist[i]) {
         printf("\n >>> Test failure: 'h%d' histo not found\n", i);
         return -1;
      }
   }

   // Check the mean values
   PutPoint();
   for (Int_t i=0; i < nhist; i++) {
      Double_t ave = hist[i]->GetMean();
      Double_t rms = hist[i]->GetRMS();
      if (TMath::Abs(ave) > 5 * rms / TMath::Sqrt(hist[i]->GetEntries())) {
         printf("\n >>> Test failure: 'h%d' histo: mean > 5 * RMS/Sqrt(N)\n", i);
         return -1;
      }
   }

   // Done
   PutPoint();
   return 0;
}

//_____________________________________________________________________________
Int_t PT_CheckH1(TQueryResult *qr, Int_t irun = 0)
{
   // Check the result of the H1 analysis

   if (!qr) {
      printf("\n >>> Test failure: output list not found\n");
      return -1;
   }

   // Make sure the number of processed entries is the one expected
   PutPoint();
   Long64_t runEntries[2] = {283813, 7525};
   if (qr->GetEntries() != runEntries[irun]) {
      printf("\n >>> Test failure: wrong number of entries processed: %lld (expected 283813)\n", qr->GetEntries());
      return -1;
   }

   // Make sure the output list is there
   PutPoint();
   TList *out = qr->GetOutputList();
   if (!out) {
      printf("\n >>> Test failure: output list not found\n");
      return -1;
   }

   // Check the 'hdmd' histo
   PutPoint();
   TH1F *hdmd = dynamic_cast<TH1F*>(out->FindObject("hdmd"));
   if (!hdmd) {
      printf("\n >>> Test failure: 'hdmd' histo not found\n");
      return -1;
   }
   if ((Int_t)(hdmd->GetEntries()) != 7525) {
      printf("\n >>> Test failure: 'hdmd' histo: wrong number"
             " of entries (%d: expected 7525) \n",(Int_t)(hdmd->GetEntries()));
      return -1;
   }
   if (TMath::Abs((hdmd->GetMean() - 0.15512023) / 0.15512023) > 0.001) {
      printf("\n >>> Test failure: 'hdmd' histo: wrong mean"
             " (%f: expected 0.15512023) \n", hdmd->GetMean());
      return -1;
   }

   PutPoint();
   TH2F *h2 = dynamic_cast<TH2F*>(out->FindObject("h2"));
   if (!h2) {
      printf("\n >>> Test failure: 'h2' histo not found\n");
      return -1;
   }
   if ((Int_t)(h2->GetEntries()) != 7525) {
      printf("\n >>> Test failure: 'h2' histo: wrong number"
             " of entries (%d: expected 7525) \n",(Int_t)(h2->GetEntries()));
      return -1;
   }
   if (TMath::Abs((h2->GetMean() - 0.15245688) / 0.15245688) > 0.001) {
      printf("\n >>> Test failure: 'h2' histo: wrong mean"
             " (%f: expected 0.15245688) \n", h2->GetMean());
      return -1;
   }

   // Done
   PutPoint();
   return 0;
}

//_____________________________________________________________________________
Int_t PT_Open(void *args)
{
   // Test session opening

   // Checking arguments
   PutPoint();
   PT_Open_Args_t *PToa = (PT_Open_Args_t *)args;
   if (!PToa) {
      printf("\n >>> Test failure: invalid arguments: %p\n", args);
      return -1;
   }

   // Temp dir for PROOF tutorials
   PutPoint();
   TString tmpdir(gSystem->TempDirectory()), us;
#if !defined(R__MACOSX) 
   if (!tmpdir.EndsWith(us.Data())) {
      UserGroup_t *ug = gSystem->GetUserInfo(gSystem->GetUid());
      if (ug) {
         us.Form("/%s", ug->fUser.Data());
         tmpdir += us;
         delete ug;
      } else {
         printf("\n >>> Test failure: could not get user info");
         return -1;
      }
   }
   gtutdir.Form("%s/.proof-tutorial", tmpdir.Data());
#else
   gtutdir.Form("%s/.proof", tmpdir.Data());
#endif
   if (gSystem->AccessPathName(gtutdir)) {
      if (gSystem->mkdir(gtutdir, kTRUE) != 0) {
         printf("\n >>> Test failure: could not assert/create the temporary directory"
                " for the tutorial (%s)", gtutdir.Data());
         return -1;
      }
   }

   // String to initialize the dataset manager
   TString dsetmgrstr;
   dsetmgrstr.Form("file dir:%s/datasets opt:-Cq:As:Sb:", gtutdir.Data());
   gEnv->SetValue("Proof.DataSetManager", dsetmgrstr.Data());

   // String to initialize the package dir
   TString packdir;
   packdir.Form("%s/packages", gtutdir.Data());
   gEnv->SetValue("Proof.PackageDir", packdir.Data());

   // Get the PROOF Session
   PutPoint();
   TProof *p = getProof(PToa->url, PToa->nwrks, gtutdir.Data(), "force", gDynamicStartup, kTRUE);
   if (!p || !(p->IsValid())) {
      printf("\n >>> Test failure: could not start the session\n");
      return -1;
   }

   PutPoint();
   if (PToa->nwrks > 0 && p->GetParallel() != PToa->nwrks) {
      printf("\n >>> Test failure: number of workers different from requested\n");
      return -1;
   }

   // Clear Cache
   p->ClearCache();

   // Get some useful info about the cluster (the sandbox dir ...)
   gSystem->RedirectOutput(0, 0, &gRH);
   TString testPrint(TString::Format("%s/testPrint.log", gtutdir.Data()));
   gSystem->RedirectOutput(testPrint, "w", &gRHAdmin);
   gProof->Print();
   gSystem->RedirectOutput(0, 0, &gRHAdmin);
   gSystem->RedirectOutput(glogfile, "a", &gRH);
   TMacro macroPrint(testPrint);
   TObjString *os = macroPrint.GetLineWith("Working directory:");
   if (!os) {
      printf("\n >>> Test failure: problem parsing output from Print()\n");
      return -1;
   }
   Int_t from = strlen("Working directory:") + 1;
   if (!os->GetString().Tokenize(gsandbox, from, " ")) {
      printf("\n >>> Test failure: no sandbox dir found\n");
      return -1;
   }
   gsandbox = gSystem->DirName(gsandbox);
   gsandbox = gSystem->DirName(gsandbox);
   PutPoint();

   // Done
   PutPoint();
   return 0;
}

//_____________________________________________________________________________
Int_t PT_GetLogs(void *args)
{
   // Test log retrieving

   // Checking arguments
   PutPoint();
   PT_Open_Args_t *PToa = (PT_Open_Args_t *)args;
   if (!PToa) {
      printf("\n >>> Test failure: invalid arguments: %p\n", args);
      return -1;
   }

   PutPoint();
   TProofLog *pl = TProof::Mgr(PToa->url)->GetSessionLogs();
   if (!pl) {
      printf("\n >>> Test failure: could not get the logs from last session\n");
      return -1;
   }

   PutPoint();
   if (PToa->nwrks > 0 && pl->GetListOfLogs()->GetSize() != (PToa->nwrks + 1)) {
      printf("\n >>> Test failure: number of logs different from workers of workers + 1\n");
      return -1;
   }

   // Done
   PutPoint();
   return 0;
}

//_____________________________________________________________________________
Int_t PT_Simple(void *submergers)
{
   // Test run for the ProofSimple analysis (see tutorials)

   // Checking arguments
   PutPoint();
   if (!gProof) {
      printf("\n >>> Test failure: no PROOF session found\n");
      return -1;
   }

   // Setup submergers if required
   if (submergers) {
      gProof->SetParameter("PROOF_UseMergers", 0);
   }

   // Define the number of events and histos
   Long64_t nevt = 1000000;
   Int_t nhist = 16;
   // The number of histograms is added as parameter in the input list
   gProof->SetParameter("ProofSimple_NHist", (Long_t)nhist);

   // Clear the list of query results
   if (gProof->GetQueryResults()) gProof->GetQueryResults()->Clear();

   // Process
   PutPoint();
   gProof->SetPrintProgress(&PrintStressProgress);
   gTimer.Start();
   gProof->Process(gSimpleSel.Data(), nevt);
   gTimer.Stop();
   gProof->SetPrintProgress(0);

   // Count
   gSimpleCnt++;
   gSimpleTime += gTimer.RealTime();

   // Remove any setting related to submergers
   gProof->DeleteParameters("PROOF_UseMergers");

   // Check the results
   PutPoint();
   return PT_CheckSimple(gProof->GetQueryResult(), nevt, nhist);
}

//_____________________________________________________________________________
Int_t PT_H1Http(void *)
{
   // Test run for the H1 analysis as a chain reading the data from HTTP

   // Checking arguments
   PutPoint();
   if (!gProof) {
      printf("\n >>> Test failure: no PROOF session found\n");
      return -1;
   }

   // Set/unset the parallel unzip flag
   AssertParallelUnzip();

   // Create the chain
   PutPoint();
   TChain *chain = new TChain("h42");

   // Assert the files, if needed
   if (!gh1ok) {
      if (PT_H1AssertFiles(gh1src.Data()) != 0) {
         gProof->SetPrintProgress(0);
         printf("\n >>> Test failure: could not assert the H1 files\n");
         return -1;
      }
   }
   Int_t i = 0;
   for (i = 0; i < 4; i++) {
      chain->Add(TString::Format("%s/%s", gh1src.Data(), gh1file[i]));
   }

   // Clear the list of query results
   if (gProof->GetQueryResults()) gProof->GetQueryResults()->Clear();

   // Process
   PutPoint();
   chain->SetProof();
   PutPoint();
   gProof->SetPrintProgress(&PrintStressProgress);
   gTimer.Start();
   chain->Process(gH1Sel.Data());
   gTimer.Stop();
   gProof->SetPrintProgress(0);
   gProof->RemoveChain(chain);

   // Count
   gH1Cnt++;
   gH1Time += gTimer.RealTime();

   // Check the results
   PutPoint();
   return PT_CheckH1(gProof->GetQueryResult());
}

//_____________________________________________________________________________
Int_t PT_H1FileCollection(void *arg)
{
   // Test run for the H1 analysis as a file collection reading the data from HTTP

   // Checking arguments
   PutPoint();
   if (!gProof) {
      printf("\n >>> Test failure: no PROOF session found\n");
      return -1;
   }

   // Set/unset the parallel unzip flag
   AssertParallelUnzip();

   // Are we asked to change the packetizer strategy?
   if (arg) {
      PT_Packetizer_t *strategy = (PT_Packetizer_t *)arg;
      if (strcmp(strategy->fName, "TPacketizerAdaptive")) {
         gProof->SetParameter("PROOF_Packetizer", strategy->fName);
      } else {
         if (strategy->fType != 1)
            gProof->SetParameter("PROOF_PacketizerStrategy", strategy->fType);
      }
   }

   // Create the file collection
   PutPoint();
   TFileCollection *fc = new TFileCollection("h42");

   // Assert the files, if needed
   if (!gh1ok) {
      if (PT_H1AssertFiles(gh1src.Data()) != 0) {
         printf("\n >>> Test failure: could not assert the H1 files\n");
         return -1;
      }
   }
   Int_t i = 0;
   for (i = 0; i < 4; i++) {
      fc->Add(new TFileInfo(TString::Format("%s/%s", gh1src.Data(), gh1file[i])));
   }

   // Clear the list of query results
   if (gProof->GetQueryResults()) gProof->GetQueryResults()->Clear();

   // Process
   PutPoint();
   gProof->SetPrintProgress(&PrintStressProgress);
   gTimer.Start();
   gProof->Process(fc, gH1Sel.Data());
   gTimer.Stop();
   gProof->SetPrintProgress(0);

   // Restore settings
   gProof->DeleteParameters("PROOF_Packetizer");
   gProof->DeleteParameters("PROOF_PacketizerStrategy");

   // Count
   gH1Cnt++;
   gH1Time += gTimer.RealTime();

   // Check the results
   PutPoint();
   return PT_CheckH1(gProof->GetQueryResult());
}

//_____________________________________________________________________________
Int_t PT_H1DataSet(void *)
{
   // Test run for the H1 analysis as a named dataset reading the data from HTTP

   // Checking arguments
   if (!gProof) {
      printf("\n >>> Test failure: no PROOF session found\n");
      return -1;
   }
   // Not yet supported for PROOF-Lite
   if (gSkipDataSetTest) {
      return 1;
   }
   PutPoint();

   // Set/unset the parallel unzip flag
   AssertParallelUnzip();

   // Name for the target dataset
   const char *dsname = "h1dset";

   // Clear the list of query results
   if (gProof->GetQueryResults()) gProof->GetQueryResults()->Clear();

   // Process the dataset by name
   PutPoint();
   gProof->SetPrintProgress(&PrintStressProgress);
   gTimer.Start();
   gProof->Process(dsname, gH1Sel.Data());
   gTimer.Stop();
   gProof->SetPrintProgress(0);

   // Count
   gH1Cnt++;
   gH1Time += gTimer.RealTime();

   // Check the results
   PutPoint();
   return PT_CheckH1(gProof->GetQueryResult());
}

//_____________________________________________________________________________
Int_t PT_H1MultiDataSet(void *)
{
   // Test run for the H1 analysis as a named dataset reading the data from HTTP

   // Checking arguments
   if (!gProof) {
      printf("\n >>> Test failure: no PROOF session found\n");
      return -1;
   }
   // Not yet supported for PROOF-Lite
   if (gSkipDataSetTest) {
      return 1;
   }
   PutPoint();

   // Set/unset the parallel unzip flag
   AssertParallelUnzip();

   // Name for the target dataset
   const char *dsname = "h1dseta h1dsetb";

   // Clear the list of query results
   if (gProof->GetQueryResults()) gProof->GetQueryResults()->Clear();

   // Process the dataset by name
   PutPoint();
   gProof->SetPrintProgress(&PrintStressProgress);
   gTimer.Start();
   gProof->Process(dsname, gH1Sel.Data());
   gTimer.Stop();
   gProof->SetPrintProgress(0);

   // Count
   gH1Cnt++;
   gH1Time += gTimer.RealTime();

   // Check the results
   PutPoint();
   return PT_CheckH1(gProof->GetQueryResult());
}

//_____________________________________________________________________________
Int_t PT_H1MultiDSetEntryList(void *)
{
   // Test run using the H1 analysis for the multi-dataset functionality and
   // entry-lists

   // Checking arguments
   if (!gProof) {
      printf("\n >>> Test failure: no PROOF session found\n");
      return -1;
   }
   // Not yet supported for PROOF-Lite
   if (gSkipDataSetTest) {
      return 1;
   }
   PutPoint();

   // Set/unset the parallel unzip flag
   AssertParallelUnzip();

   // Multiple dataset used to create the entry list
   TString dsname("h1dseta|h1dsetb");

   // Clear the list of query results
   if (gProof->GetQueryResults()) gProof->GetQueryResults()->Clear();

   // Entry-list creation run
   PutPoint();
   gProof->SetPrintProgress(&PrintStressProgress);
   gTimer.Start();
   gProof->Process(dsname, gH1Sel.Data(), "fillList=elist.root");
   gTimer.Stop();
   gProof->SetPrintProgress(0);
   // Cleanup entry-list from the input list
   TIter nxi(gProof->GetInputList());
   TObject *o = 0;
   while ((o = nxi())) {
      if (!strncmp(o->GetName(), "elist", 6) || !strcmp(o->GetName(), "fillList")) {
         gProof->GetInputList()->Remove(o);
         delete o;
      }
   }

   // Count
   gH1Cnt++;
   gH1Time += gTimer.RealTime();

   // Run using the entrylist
   dsname = "h1dseta<<elist.root h1dsetb?enl=elist.root";
   PutPoint();
   gProof->SetPrintProgress(&PrintStressProgress);
   gTimer.Start();
   gProof->Process(dsname, gH1Sel.Data());
   gTimer.Stop();
   gProof->SetPrintProgress(0);

   // Unlink the entry list file
   gSystem->Unlink("elist.root");
   // Cleanup entry-list from the input list
   nxi.Reset();
   while ((o = nxi())) {
      if (!strncmp(o->GetName(), "elist", 6)) {
         gProof->GetInputList()->Remove(o);
         delete o;
      }
   }

   // Check the results
   PutPoint();
   return PT_CheckH1(gProof->GetQueryResult(), 1);
}

//_____________________________________________________________________________
Int_t PT_DataSets(void *)
{
   // Test dataset registration, verification, usage, removal.
   // Use H1 analysis files on HTTP as example

   // Checking arguments
   if (!gProof) {
      printf("\n >>> Test failure: no PROOF session found\n");
      return -1;
   }
   // Not yet supported for PROOF-Lite
   if (gSkipDataSetTest) {
      return 1;
   }
   PutPoint();

   // Cleanup the area
   PutPoint();
   TMap *dsm = gProof->GetDataSets();
   if (!dsm) {
      printf("\n >>> Test failure: could not retrieve map of datasets (even empty)!\n");
      return -1;
   }
   if (dsm->GetSize() > 0) {
      // Remove the datasets already registered
      TIter nxd(dsm);
      TObjString *os = 0;
      while ((os = (TObjString *)nxd())) {
         gProof->RemoveDataSet(os->GetName());
      }
      // Check the result
      delete dsm;
      dsm = gProof->GetDataSets();
      if (!dsm || dsm->GetSize() > 0) {
         printf("\n >>> Test failure: could not cleanup the dataset area! (%p)\n", dsm);
         delete dsm;
         return -1;
      }
   }
   delete dsm;

   // Create the file collection
   PutPoint();
   TFileCollection *fc = new TFileCollection();
   TFileCollection *fca = new TFileCollection();
   TFileCollection *fcb = new TFileCollection();

   // Assert the files, if needed
   if (!gh1ok) {
      if (PT_H1AssertFiles(gh1src.Data()) != 0) {
         printf("\n >>> Test failure: could not assert the H1 files\n");
         return -1;
      }
   }
   Int_t i = 0;
   for (i = 0; i < 4; i++) {
      fc->Add(new TFileInfo(TString::Format("%s/%s", gh1src.Data(), gh1file[i])));
      if (i < 2) {
         fca->Add(new TFileInfo(TString::Format("%s/%s", gh1src.Data(), gh1file[i])));
      } else {
         fcb->Add(new TFileInfo(TString::Format("%s/%s", gh1src.Data(), gh1file[i])));
      }
   }
   fc->Update();
   fca->Update();
   fcb->Update();

   // Name for this dataset
   const char *dsname = "h1dset";
   const char *dsnamea = "h1dseta";
   const char *dsnameb = "h1dsetb";

   // Register the dataset
   PutPoint();
   gProof->RegisterDataSet(dsname, fc);
   gProof->RegisterDataSet(dsnamea, fca);
   gProof->RegisterDataSet(dsnameb, fcb);
   // Check the result
   dsm = gProof->GetDataSets();
   if (!dsm || dsm->GetSize() != 3) {
      printf("\n >>> Test failure: could not register '%s,%s,%s' (%p)\n",
             dsname, dsnamea, dsnameb, dsm);
      delete dsm;
      return -1;
   }
   delete dsm;

   // Test removal
   PutPoint();
   gProof->RemoveDataSet(dsname);
   gProof->RemoveDataSet(dsnamea);
   gProof->RemoveDataSet(dsnameb);
   // Check the result
   dsm = gProof->GetDataSets();
   if (!dsm || dsm->GetSize() != 0) {
      printf("\n >>> Test failure: could not cleanup '%s,%s,%s' (%p)\n",
             dsname, dsnamea, dsnameb, dsm);
      delete dsm;
      return -1;
   }
   delete dsm;

   // Re-register the dataset
   PutPoint();
   gProof->RegisterDataSet(dsname, fc);
   gProof->RegisterDataSet(dsnamea, fca);
   gProof->RegisterDataSet(dsnameb, fcb);
   // Check the result
   dsm = gProof->GetDataSets();
   if (!dsm || dsm->GetSize() != 3) {
      printf("\n >>> Test failure: could not re-register '%s,%s,%s' (%p)\n",
             dsname, dsnamea, dsnameb, dsm);
      delete dsm;
      return -1;
   }
   delete dsm;

   // Verify the dataset
   PutPoint();
   if (gProof->VerifyDataSet(dsname) != 0) {
      printf("\n >>> Test failure: could not verify '%s'!\n", dsname);
      return -1;
   }
   if (gProof->VerifyDataSet(dsnamea) != 0) {
      printf("\n >>> Test failure: could not verify '%s'!\n", dsnamea);
      return -1;
   }
   if (gProof->VerifyDataSet(dsnameb) != 0) {
      printf("\n >>> Test failure: could not verify '%s'!\n", dsnameb);
      return -1;
   }
   gProof->ShowDataSets();

   // Remove the file collection
   delete fc;
   delete fca;
   delete fcb;

   // Done
   PutPoint();
   return 0;
}

//_____________________________________________________________________________
Int_t PT_Packages(void *)
{
   // Test package clearing, uploading, enabling, removal.
   // Use event.par as example.

   // Checking arguments
   PutPoint();
   if (!gProof) {
      printf("\n >>> Test failure: no PROOF session found\n");
      return -1;
   }

   // Cleanup the area
   PutPoint();
   TList *packs = gProof->GetListOfPackages();
   if (!packs) {
      printf("\n >>> Test failure: could not retrieve list of packages (even empty)!\n");
      return -1;
   }
   if (packs->GetSize() > 0) {
      // Remove the packages already available
      gProof->ClearPackages();
      // Check the result
      packs = gProof->GetListOfPackages();
      if (!packs || packs->GetSize() > 0) {
         printf("\n >>> Test failure: could not cleanup the package area!\n");
         return -1;
      }
   }

   // Name and location for this package
   const char *pack = "event";
   TString packpath("$ROOTSYS/tutorials/proof/event.par");
   gSystem->ExpandPathName(packpath);

   // Upload the package
   PutPoint();
   gProof->UploadPackage(packpath);
   // Check the result
   packs = gProof->GetListOfPackages();
   if (!packs || packs->GetSize() != 1) {
      printf("\n >>> Test failure: could not upload '%s'!\n", packpath.Data());
      return -1;
   }

   // Test cleanup
   PutPoint();
   gProof->ClearPackage(pack);
   // Check the result
   packs = gProof->GetListOfPackages();
   if (!packs || packs->GetSize() != 0) {
      printf("\n >>> Test failure: could not cleanup '%s'!\n", pack);
      return -1;
   }

   // Re-upload the package
   PutPoint();
   gProof->UploadPackage(packpath);
   // Check the result
   packs = gProof->GetListOfPackages();
   if (!packs || packs->GetSize() != 1) {
      printf("\n >>> Test failure: could not re-upload '%s'!\n", packpath.Data());
      return -1;
   }

   // Enable the package
   PutPoint();
   gProof->EnablePackage(pack);
   // Check the result
   packs = gProof->GetListOfEnabledPackages();
   if (!packs || packs->GetSize() != 1) {
      printf("\n >>> Test failure: could not enable '%s'!\n", pack);
      return -1;
   }

   // Done
   PutPoint();
   return 0;
}

//_____________________________________________________________________________
Int_t PT_Event(void *)
{
   // Test run for the ProofEvent analysis (see tutorials)

   // Checking arguments
   PutPoint();
   if (!gProof) {
      printf("\n >>> Test failure: no PROOF session found\n");
      return -1;
   }

   // Define the number of events
   Long64_t nevt = 100000;

   // Clear the list of query results
   if (gProof->GetQueryResults()) gProof->GetQueryResults()->Clear();

   // Process
   PutPoint();
   gProof->SetPrintProgress(&PrintStressProgress);
   gProof->Process(gEventSel.Data(), nevt);
   gProof->SetPrintProgress(0);

   // Make sure the query result is there
   PutPoint();
   TQueryResult *qr = 0;
   if (!(qr = gProof->GetQueryResult())) {
      printf("\n >>> Test failure: query result not found\n");
      return -1;
   }

   // Make sure the number of processed entries is the one expected
   PutPoint();
   if (qr->GetEntries() != nevt) {
      printf("\n >>> Test failure: wrong number of entries processed: %lld (expected %lld)\n",
             qr->GetEntries(), nevt);
      return -1;
   }

   // Make sure the output list is there
   PutPoint();
   if (!(gProof->GetOutputList())) {
      printf("\n >>> Test failure: output list not found\n");
      return -1;
   }

   // Check the 'histo'
   PutPoint();
   TH1F *histo = dynamic_cast<TH1F*>(gProof->GetOutputList()->FindObject("histo"));
   if (!histo) {
      printf("\n >>> Test failure: 'histo' not found\n");
      return -1;
   }

   // Check the mean values
   Double_t ave = histo->GetMean();
   Double_t rms = histo->GetRMS();
   if (TMath::Abs(ave - 50) > 10 * rms / TMath::Sqrt(histo->GetEntries())) {
      printf("\n >>> Test failure: 'histo': mean > 5 * RMS/Sqrt(N)\n");
      return -1;
   }

   // Done
   PutPoint();
   return 0;
}

//_____________________________________________________________________________
Int_t PT_InputData(void *)
{
   // Test input data functionality

   // Checking arguments
   if (!gProof) {
      printf("\n >>> Test failure: no PROOF session found\n");
      return -1;
   }
   PutPoint();

   // Create the test information to be send via input and retrieved
   TH1F *h1 = new TH1F("h1data","Input data from file",100,-5.,5.);
   h1->FillRandom("gaus", 1000);
   TList *h1list = new TList;
   h1list->SetName("h1list");
   h1list->SetOwner(kTRUE);
   h1list->Add(h1);
   h1list->Add(new TParameter<Double_t>("h1avg", h1->GetMean()));
   h1list->Add(new TParameter<Double_t>("h1rms", h1->GetRMS()));
   TString datafile = glogfile;
   datafile += ("_h1data.root");
   TFile *f = TFile::Open(datafile, "RECREATE");
   if (!f) {
      printf("\n >>> Test failure: could not open file for input data\n");
      return -1;
   }
   f->cd();
   h1list->Write(0, TObject::kSingleKey, 0);
   f->Close();
   gProof->SetInputDataFile(datafile.Data());

   // Histo to be sent from memory
   TH1F *h2 = new TH1F("h2data","Input data from memory",100,-5.,5.);
   h2->FillRandom("gaus", 1000);
   TList *h2list = new TList;
   h2list->SetName("h2list");
   h2list->SetOwner(kTRUE);
   h2list->Add(h2);
   h2list->Add(new TParameter<Double_t>("h2avg", h2->GetMean()));
   h2list->Add(new TParameter<Double_t>("h2rms", h2->GetRMS()));
   gProof->AddInputData(h2list);

   // Normal input parameter
   gProof->AddInput(new TNamed("InputObject", glogfile.Data()));

   // Type of test
   gProof->AddInput(new TNamed("ProofTests_Type", "InputData"));

   // Define the number of events
   Long64_t nevt = 1;

   // Clear the list of query results
   if (gProof->GetQueryResults()) gProof->GetQueryResults()->Clear();

   // Process
   PutPoint();
   gProof->SetPrintProgress(&PrintStressProgress);
   gProof->Process(gTestsSel.Data(), nevt);
   gProof->SetPrintProgress(0);

   // Cleanup
   gSystem->Unlink(datafile.Data());
   gProof->ClearInputData(h1list);
   gProof->ClearInputData(h2list);
   delete h1list;
   delete h2list;

   // Make sure the query result is there
   PutPoint();
   TQueryResult *qr = 0;
   if (!(qr = gProof->GetQueryResult())) {
      printf("\n >>> Test failure: query result not found\n");
      return -1;
   }

   // Make sure the output list is there
   PutPoint();
   if (!(gProof->GetOutputList())) {
      printf("\n >>> Test failure: output list not found\n");
      return -1;
   }

   // Check the 'histo's
   PutPoint();
   TH1I *stat = dynamic_cast<TH1I*>(gProof->GetOutputList()->FindObject("TestStat"));
   if (!stat) {
      printf("\n >>> Test failure: 'TestStat' histo not found\n");
      return -1;
   }

   // Test how many workers got everything successfully
   Int_t nw = (Int_t) stat->GetBinContent(1);
   PutPoint();

   if (TMath::Abs(stat->GetBinContent(2) - nw) > .1) {
      printf("\n >>> Test failure: histo 'h1' not correctly received on all workers (%.0f/%d)\n",
             stat->GetBinContent(2), nw);
      return -1;
   }
   if (TMath::Abs(stat->GetBinContent(3) - nw) > .1) {
      printf("\n >>> Test failure: histo 'h2' not correctly received on all workers (%.0f/%d)\n",
             stat->GetBinContent(3), nw);
      return -1;
   }
   if (TMath::Abs(stat->GetBinContent(4) - nw) > .1) {
      printf("\n >>> Test failure: test input object not correctly received on all workers (%.0f/%d)\n",
             stat->GetBinContent(4), nw);
      return -1;
   }

   // Done
   PutPoint();
   return 0;
}

//_____________________________________________________________________________
Int_t PT_PackageArguments(void *)
{
   // Testing passing arguments to packages

   // Checking arguments
   if (!gProof) {
      printf("\n >>> Test failure: no PROOF session found\n");
      return -1;
   }
   PutPoint();

   // Passing a 'const char *': upload packtest1
   const char *pack1 = "packtest1";
   TString pack1path("$ROOTSYS/tutorials/proof/packtest1.par");
   gSystem->ExpandPathName(pack1path);
   PutPoint();
   gProof->UploadPackage(pack1path);
   // Check the result
   TList *packs = gProof->GetListOfPackages();
   if (!packs || !packs->FindObject(pack1)) {
      printf("\n >>> Test failure: could not upload '%s'!\n", pack1path.Data());
      return -1;
   }
   // Enable the package now passing a 'const char *' argument
   TString arg("ProofTest.ConstChar");
   if (gProof->EnablePackage(pack1, arg) != 0) {
      printf("\n >>> Test failure: could not enable '%s' with argument: '%s'!\n", pack1path.Data(), arg.Data());
      return -1;
   }

   // Type of test
   gProof->SetParameter("ProofTests_Type", "PackTest1");

   // Define the number of events
   Long64_t nevt = 1;

   // Clear the list of query results
   if (gProof->GetQueryResults()) gProof->GetQueryResults()->Clear();

   // Variable to check
   gProof->SetParameter("testenv", arg.Data());

   // Process
   PutPoint();
   gProof->SetPrintProgress(&PrintStressProgress);
   gProof->Process(gTestsSel.Data(), nevt);
   gProof->SetPrintProgress(0);

   // Some cleanup
   gProof->ClearPackage(pack1);
   gProof->DeleteParameters("ProofTests_Type");
   gProof->DeleteParameters("testenv");

   // Make sure the query result is there
   PutPoint();
   TQueryResult *qr = 0;
   if (!(qr = gProof->GetQueryResult())) {
      printf("\n >>> Test failure: query result not found\n");
      return -1;
   }

   // Make sure the output list is there
   PutPoint();
   if (!(gProof->GetOutputList())) {
      printf("\n >>> Test failure: output list not found\n");
      return -1;
   }

   // Check the 'histo's
   PutPoint();
   TH1I *stat = dynamic_cast<TH1I*>(gProof->GetOutputList()->FindObject("TestStat"));
   if (!stat) {
      printf("\n >>> Test failure: 'TestStat' histo not found\n");
      return -1;
   }

   // Test how many workers got everything successfully
   Int_t nw = (Int_t) stat->GetBinContent(1);
   PutPoint();

   if (TMath::Abs(stat->GetBinContent(2) - nw) > .1) {
      printf("\n >>> Test failure: var '%s' not correctly set on all workers (%.0f/%d)\n",
             arg.Data(), stat->GetBinContent(2), nw);
      return -1;
   }

   // Passing a 'TList *': upload packtest2
   const char *pack2 = "packtest2";
   TString pack2path("$ROOTSYS/tutorials/proof/packtest2.par");
   gSystem->ExpandPathName(pack2path);
   PutPoint();
   gProof->UploadPackage(pack2path);
   // Check the result
   packs = gProof->GetListOfPackages();
   if (!packs || !packs->FindObject(pack2)) {
      printf("\n >>> Test failure: could not upload '%s'!\n", pack2path.Data());
      return -1;
   }
   // Create the argument list
   TList *argls = new TList;
   argls->Add(new TNamed("ProofTest.ArgOne", "2."));
   argls->Add(new TNamed("ProofTest.ArgTwo", "3."));
   argls->Add(new TNamed("ProofTest.ArgThree", "4."));
   // Enable the package now passing the 'TList *' argument
   if (gProof->EnablePackage(pack2, argls) != 0) {
      printf("\n >>> Test failure: could not enable '%s' with argument: '%s'!\n", pack2path.Data(), arg.Data());
      return -1;
   }

   // Type of test
   gProof->SetParameter("ProofTests_Type", "PackTest2");

   // Clear the list of query results
   if (gProof->GetQueryResults()) gProof->GetQueryResults()->Clear();

   // Variable to check
   TString envs("ProofTest.ArgOne,ProofTest.ArgTwo,ProofTest.ArgThree");
   gProof->SetParameter("testenv", envs.Data());

   // Process
   PutPoint();
   gProof->SetPrintProgress(&PrintStressProgress);
   gProof->Process(gTestsSel.Data(), nevt);
   gProof->SetPrintProgress(0);

   // Some cleanup
   gProof->ClearPackage(pack2);
   gProof->DeleteParameters("ProofTests_Type");
   gProof->DeleteParameters("testenv");

   // Make sure the query result is there
   PutPoint();
   qr = 0;
   if (!(qr = gProof->GetQueryResult())) {
      printf("\n >>> Test failure: query result not found\n");
      return -1;
   }

   // Make sure the output list is there
   PutPoint();
   if (!(gProof->GetOutputList())) {
      printf("\n >>> Test failure: output list not found\n");
      return -1;
   }

   // Check the 'histo's
   PutPoint();
   if (!(stat = dynamic_cast<TH1I*>(gProof->GetOutputList()->FindObject("TestStat")))) {
      printf("\n >>> Test failure: 'TestStat' histo not found\n");
      return -1;
   }

   // Test how many workers got everything successfully
   nw = (Int_t) stat->GetBinContent(1);
   PutPoint();

   if (TMath::Abs(stat->GetBinContent(2) - nw) > .1) {
      printf("\n >>> Test failure: var 'ProofTest.ArgOne' not correctly set on all workers (%.0f/%d)\n",
             stat->GetBinContent(2), nw);
      return -1;
   }

   if (TMath::Abs(stat->GetBinContent(3) - nw) > .1) {
      printf("\n >>> Test failure: var 'ProofTest.ArgTwo' not correctly set on all workers (%.0f/%d)\n",
             stat->GetBinContent(3), nw);
      return -1;
   }

   if (TMath::Abs(stat->GetBinContent(4) - nw) > .1) {
      printf("\n >>> Test failure: var 'ProofTest.ArgThree' not correctly set on all workers (%.0f/%d)\n",
             stat->GetBinContent(4), nw);
      return -1;
   }

   // Done
   PutPoint();
   return 0;
}

//_____________________________________________________________________________
Int_t PT_H1SimpleAsync(void *arg)
{
   // Test run for the H1 and Simple analysis in asynchronous mode 

   // Checking arguments
   if (!gProof) {
      printf("\n >>> Test failure: no PROOF session found\n");
      return -1;
   }
   // Not yet supported for PROOF-Lite
   if (gProof->IsLite()) {
      return 1;
   }
   PutPoint();

   // Set/unset the parallel unzip flag
   AssertParallelUnzip();

   // Are we asked to change the packetizer strategy?
   if (arg) {
      PT_Packetizer_t *strategy = (PT_Packetizer_t *)arg;
      if (strcmp(strategy->fName, "TPacketizerAdaptive")) {
         gProof->SetParameter("PROOF_Packetizer", strategy->fName);
      } else {
         if (strategy->fType != 1)
            gProof->SetParameter("PROOF_PacketizerStrategy", strategy->fType);
      }
   }

   // Create the file collection
   PutPoint();
   TFileCollection *fc = new TFileCollection("h42");

   // Assert the files, if needed
   if (!gh1ok) {
      if (PT_H1AssertFiles(gh1src.Data()) != 0) {
         printf("\n >>> Test failure: could not assert the H1 files\n");
         return -1;
      }
   }
   Int_t i = 0;
   for (i = 0; i < 4; i++) {
      fc->Add(new TFileInfo(TString::Format("%s/%s", gh1src.Data(), gh1file[i])));
   }

   // Clear the list of query results
   PutPoint();
   if (gProof->GetQueryResults()) {
      gProof->GetQueryResults()->Clear();
      gProof->Remove("cleanupdir");
   }

   // Submit the processing requests
   PutPoint();
   gProof->SetPrintProgress(&PrintStressProgress);
   gProof->Process(fc, gH1Sel.Data(), "ASYN");

   // Define the number of events and histos
   Long64_t nevt = 1000000;
   Int_t nhist = 16;
   // The number of histograms is added as parameter in the input list
   gProof->SetParameter("ProofSimple_NHist", (Long_t)nhist);
   gProof->Process(gSimpleSel.Data(), nevt, "ASYN");

   // Wait a bit as a function of previous runnings
   Double_t dtw = 10;
   if (gH1Cnt > 0 && gSimpleTime > 0) {
      dtw = gH1Time / gH1Cnt + gSimpleTime / gSimpleCnt + 1;
      if (dtw < 10) dtw = 10;
   }
   Int_t tw = (Int_t) (5 * dtw);

   gTimedOut = kFALSE;
   TTimeOutTimer t(tw*1000);

   // Wait for the processing
   while (!gProof->IsIdle() && !gTimedOut)
      gSystem->InnerLoop();

   gProof->SetPrintProgress(0);
   PutPoint();

   // Retrieve the list of available query results
   TList *ql = gProof->GetQueryResults();
   if (ql && ql->GetSize() > 0) {
      ql->Print();
   }

   TString ref;
   TIter nxq(ql, kIterBackward);
   Int_t nd = 2;
   TQueryResult *qr = 0;
   while ((qr = (TQueryResult *)nxq()) && nd > 0) {
      ref.Form("%s:%s", qr->GetTitle(), qr->GetName());
      gProof->Retrieve(ref);
      qr = gProof->GetQueryResult(ref);
      if (qr && qr->GetSelecImp()) {
         if (!strcmp(qr->GetSelecImp()->GetTitle(), "h1analysis")) {
            PutPoint();
            if (PT_CheckH1(qr) != 0) return -1;
            nd--;
         } else if (!strcmp(qr->GetSelecImp()->GetTitle(), "ProofSimple")) {
            PutPoint();
            if (PT_CheckSimple(qr, nevt, nhist) != 0) return -1;
            nd--;
         } else {
            printf("\n >>> Test failure: query with unexpected selector '%s'\n", qr->GetSelecImp()->GetTitle());
            return -1;
         }
      } else {
         printf("\n >>> Test failure: query undefined (%p) or with empty selector macro ('%s')\n", qr, ref.Data());
         return -1;
      }
   }

   // Done
   PutPoint();
   return 0;
}

//_____________________________________________________________________________
Int_t PT_AdminFunc(void *)
{
   // Test run for the admin functionality

   // Checking arguments
   if (!gProof) {
      printf("\n >>> Test failure: no PROOF session found\n");
      return -1;
   }
   // Not yet supported for PROOF-Lite
   if (gProof->IsLite()) {
      return 1;
   }
   PutPoint();
   // Attach to the manager
   TProofMgr *mgr = gProof->GetManager();
   if (!mgr) {
      printf("\n >>> Test failure: no PROOF manager found\n");
      return -1;
   }
   PutPoint();
   // Directory for this test
   TString testDir(TString::Format("%s/stressProof-Admin", gtutdir.Data()));
   if (gSystem->AccessPathName(testDir)) {
      // Create the directory
      if (gSystem->MakeDirectory(testDir)) {
         printf("\n >>> Test failure: cannot create %s\n", testDir.Data());
         return -1;
      }
   }
   // Create a small test file
   TMacro testMacro;
   testMacro.AddLine("// Test macro");
   testMacro.AddLine("#include \"TSystem.h\"");
   testMacro.AddLine("void testMacro(Int_t opt = 1)");
   testMacro.AddLine("{");
   testMacro.AddLine("   if (opt == 1) {");
   testMacro.AddLine("      Printf(\"Pid: \", gSystem->GetPid());");
   testMacro.AddLine("   }");
   testMacro.AddLine("}");
   // Save the file in the temporary area
   TString testFile(TString::Format("%s/testMacro.C", testDir.Data()));
   if (!gSystem->AccessPathName(testFile)) {
      // The file exists: remove it first
      if (gSystem->Unlink(testFile)) {
         printf("\n >>> Test failure: cannot unlink %s\n", testFile.Data());
         return -1;
      }
   }
   testMacro.SaveSource(testFile);
   FileStat_t stloc;
   if (gSystem->GetPathInfo(testFile, stloc) != 0) {
      // The file was not created
      printf("\n >>> Test failure: file %s was not created\n", testFile.Data());
      return -1;
   }
   // Reference checksum
   TMD5 *testMacroMd5 = testMacro.Checksum();
   if (!testMacroMd5) {
      // MD5 sum not calculated
      printf("\n >>> Test failure: could not calculate the md5 sum of the test macro\n");
      return -1;
   }
   PutPoint();

   // Send the file to the sandbox
   if (mgr->PutFile(testFile, "~/", "force") != 0) {
      // The file was not sent over correctly
      printf("\n >>> Test failure: problems sending file to master sandbox\n");
      return -1;
   }
   PutPoint();
   // Retrieve the file
   TString testFile1(TString::Format("%s/testMacro.cxx", testDir.Data()));
   if (mgr->GetFile("~/testMacro.C", testFile1, "force") != 0) {
      // The file was not retrieved correctly
      printf("\n >>> Test failure: problems retrieving file from the master sandbox\n");
      return -1;
   }
   PutPoint();

   // Test 'ls'
   gSystem->RedirectOutput(0, 0, &gRH);
   TString testLs(TString::Format("%s/testLs.log", testDir.Data()));
   gSystem->RedirectOutput(testLs, "w", &gRHAdmin);
   mgr->Ls("~/testMacro.C");
   gSystem->RedirectOutput(0, 0, &gRHAdmin);
   gSystem->RedirectOutput(glogfile, "a", &gRH);
   TMacro macroLs(testLs);
   TString testLsLine = TString::Format("%s/testMacro.C", gsandbox.Data());
   if (testLsLine.Index(".proof-tutorial") != kNPOS)
      // The first part of <tmp> maybe sligthly different
      testLsLine.Remove(0, testLsLine.Index(".proof-tutorial"));
   if (!macroLs.GetLineWith(testLsLine)) {
      printf("\n >>> Test failure: Ls: output not consistent (line: '%s')\n", testLsLine.Data());
      printf(" >>> Log file: '%s'\n", testLs.Data());
      printf("+++ BOF +++\n");
      macroLs.Print();
      printf("+++ EOF +++\n");
      return -1;
   }
   PutPoint();

   // Test 'more'
   gSystem->RedirectOutput(0, 0, &gRH);
   TString testMore(TString::Format("%s/testMore.log", testDir.Data()));
   gSystem->RedirectOutput(testMore, "w", &gRHAdmin);
   mgr->More("~/testMacro.C");
   gSystem->RedirectOutput(0, 0, &gRHAdmin);
   gSystem->RedirectOutput(glogfile, "a", &gRH);
   TMacro macroMore(testMore);
   if (macroMore.GetListOfLines()->GetSize() < 2) {
      printf("\n >>> Test failure: More output not too short: %d lines\n",
                                   macroMore.GetListOfLines()->GetSize());
      return -1;
   }
   macroMore.GetListOfLines()->Remove(macroMore.GetListOfLines()->First());
   macroMore.GetListOfLines()->Remove(macroMore.GetListOfLines()->First());
   TMD5 *testMoreMd5 = macroMore.Checksum();
   if (!testMoreMd5) {
      // MD5 sum not calculated
      printf("\n >>> Test failure: could not calculate the md5 sum of the 'more' result\n");
      return -1;
   }
   if (strcmp(testMacroMd5->AsString(), testMoreMd5->AsString())) {
      printf("\n >>> Test failure: More: result not consistent with reference: {%s, %s}\n",
                                         testMacroMd5->AsString(), testMoreMd5->AsString());
      return -1;
   }
   PutPoint();

   // Test 'stat'
   FileStat_t strem;
   if (mgr->Stat("~/testMacro.C", strem) != 0) {
      // Stat failure
      printf("\n >>> Test failure: could not stat remotely the test file\n");
      return -1;
   }
   if (strem.fSize != stloc.fSize) {
      // Stat failure
      printf("\n >>> Test failure: stat sizes inconsistent: %lld vs %lld (bytes)\n", strem.fSize, stloc.fSize);
      return -1;
   }
   PutPoint();

   // Test 'cp' and 'md5sum;
   if (mgr->Cp("http://root.cern.ch/files/h1/dstarmb.root", "~/") != 0) {
      // Cp failure
      printf("\n >>> Test failure: could not retrieve remotely dstarmb.root from the Root Web server\n");
      return -1;
   }
   TString sum;
   if (mgr->Md5sum("~/dstarmb.root", sum) != 0) {
      // MD5
      printf("\n >>> Test failure: calculating the md5sum of dstarmb.root\n");
      return -1;
   }
   if (sum != "0a60055370e16d954f90fb50c2d1a801") {
      // MD5 wrong
      printf("\n >>> Test failure: wrong value for md5sum of dstarmb.root: %s\n", sum.Data());
      return -1;
   }
   PutPoint();

   // Clean up sums
   SafeDelete(testMoreMd5);
   SafeDelete(testMacroMd5);

   // Done
   PutPoint();
   return 0;

}
