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
// *                  [-dyn] [-ds] [master]                                * //
// *                                                                       * //
// * Optional arguments:                                                   * //
// *   -h          show help info                                          * //
// *   master      entry point of the cluster where to run the test        * //
// *               in the form '[user@]host.domain[:port]';                * //
// *               default 'localhost:11093'                               * //
// *   -n wrks     number of workers to be started when running on the     * //
// *               local host; default is the nuber of local cores         * //
// *   -v[v[v]]    verbosity level (not implemented)                       * //
// *   -l logfile  file where to redirect the processing logs; default is  * //
// *               a temporary file deleted at the end of the test; in     * //
// *               case of success                                         * //
// *   -dyn        run the test in dynamic startup mode                    * //
// *   -ds        force the dataset test if skipped by default             * //
// *                                                                       * //
// * To run interactively:                                                 * //
// * $ root                                                                * //
// * root[] .L stressProof.cxx                                             * //
// * root[] stressProof(master, wrks, verbose, logfile, dyn, skipds)       * //
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
// *   Test  9 : Package management with 'event' .................. OK *   * //
// *   Test 10 : Simple 'event' generation ........................ OK *   * //
// *   Test 11 : Input data propagation ........................... OK *   * //
// *   Test 12 : H1, Simple: async mode :.......................... OK *   * //
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
// ************************************************************************* //

#include <stdio.h>
#include <stdlib.h>

#include "Getline.h"
#include "TChain.h"
#include "TFile.h"
#include "TFileCollection.h"
#include "TFileInfo.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TMap.h"
#include "TMath.h"
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

static const char *urldef = "proof://localhost:11093";
static Int_t gverbose = 0;
static TString glogfile = 0;
static Int_t gpoints = 0;
static Int_t totpoints = 53;
static RedirectHandle_t gRH;
static Double_t gH1Time = 0;
static Double_t gSimpleTime = 0;
static Int_t gH1Cnt = 0;
static Int_t gSimpleCnt = 0;
static TStopwatch gTimer;
static Bool_t gTimedOut = kFALSE;
static Bool_t gDynamicStartup = kFALSE;
static Bool_t gSkipDataSetTest = kTRUE;

void stressProof(const char *url = "proof://localhost:11093",
                 Int_t nwrks = -1, Int_t verbose = 0,
                 const char *logfile = 0, Bool_t dyn = kFALSE, Bool_t skipds = kTRUE);

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
      printf(" $ ./stressProof [-h] [-n <wrks>] [-v[v[v]]] [-l logfile] [-dyn] [-ds] [master]\n");
      printf(" \n");
      printf(" Optional arguments:\n");
      printf("   -h            prints this menu\n");
      printf("   master        entry point of the cluster where to run the test\n");
      printf("                 in the form '[user@]host.domain[:port]'; default 'localhost:11093'\n");
      printf("   -n wrks       number of workers to be started when running on the local host;\n");
      printf("                 default is the nuber of local cores\n");
      printf("   -v[v[v]]      verbosity level (not implemented)\n");
      printf("   -l logfile    file where to redirect the processing logs; must be writable;\n");
      printf("                 default is a temporary file deleted at the end of the test\n");
      printf("                 in case of success\n");
      printf("   -dyn          run the test in dynamicStartup mode\n");
      printf("   -ds           force the dataset test if skipped by default\n");
      printf(" \n");
      gSystem->Exit(0);
   }

   // Parse options
   const char *url = 0;
   Int_t nWrks = -1;
   Int_t verbose = 0;
   const char *logfile = 0;
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
      } else if (!strcmp(argv[i],"-l")) {
         if (i+1 == argc || argv[i+1][0] == '-') {
            printf(" -l should be followed by a path: ignoring \n");
            i++;
         } else { 
            logfile = argv[i+1];
            i += 2;
         }
      } else if (!strncmp(argv[i],"-v",2)) {
         verbose++;
         if (!strncmp(argv[i],"-vv",3)) verbose++;
         if (!strncmp(argv[i],"-vvv",4)) verbose++;
         i++;
      } else if (!strncmp(argv[i],"-dyn",4)) {
         gDynamicStartup = kTRUE;
         i++;
      } else if (!strncmp(argv[i],"-ds",3)) {
         gSkipDataSetTest = kFALSE;
         i++;
      } else {
         url = argv[i];
         i++;
      }
   }
   // Use defaults where required
   if (!url) url = urldef;

   stressProof(url, nWrks, verbose, logfile, gDynamicStartup, gSkipDataSetTest);

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
void PrintStressProgress(Long64_t total, Long64_t processed, Float_t)
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

//
// Auxilliary classes for testing
//
typedef Int_t (*ProofTestFun_t)(void *);
class ProofTest : public TNamed {
private:
   Int_t           fSeq;  // Sequential number for the test
   ProofTestFun_t  fFun;  // Function to be executed for the test
   void           *fArgs; // Arguments to be passed to the function

public:
   ProofTest(const char *n, Int_t seq, ProofTestFun_t f, void *a = 0)
           : TNamed(n,""), fSeq(seq), fFun(f), fArgs(a) { }
   virtual ~ProofTest() { }

   Int_t Run();
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
Int_t PT_Simple(void *);
Int_t PT_H1Http(void *);
Int_t PT_H1FileCollection(void *);
Int_t PT_H1DataSet(void *);
Int_t PT_DataSets(void *);
Int_t PT_Packages(void *);
Int_t PT_Event(void *);
Int_t PT_InputData(void *);
Int_t PT_H1SimpleAsync(void *arg);

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
void stressProof(const char *url, Int_t nwrks,
                 Int_t verbose, const char *logfile, Bool_t dyn, Bool_t skipds)
{
   printf("******************************************************************\n");
   printf("*  Starting  P R O O F - S T R E S S  suite                      *\n");
   printf("******************************************************************\n");

   // Set dynamic mode
   gDynamicStartup = (!strcmp(url,"lite")) ? kFALSE : dyn;

   // Set verbosity
   gverbose = verbose;

   // Notify/warn about the dynamic startup option, if any
   if (gDynamicStartup) {
      // Check url
      TUrl uu(url), udef(urldef);
      if (strcmp(uu.GetHost(), udef.GetHost()) || (uu.GetPort() != udef.GetPort())) {
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
      gSkipDataSetTest = (!strcmp(url, urldef) || !strcmp(url, "lite")) ? kFALSE : kTRUE;
   }

   // Log file path
   Bool_t usedeflog = kTRUE;
   FILE *flog = 0;
   if (logfile && strlen(logfile) > 0) {
      usedeflog = kFALSE;
      glogfile = logfile;
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
      printf("*  Log file: %s\n", glogfile.Data());
      printf("******************************************************************\n");
   }

   if (gSkipDataSetTest) {
      printf("*  Test for dataset handling (#4, #8) skipped                   **\n");
      printf("******************************************************************\n");
   }
   if (!strcmp(url,"lite")) {
      printf("*  PROOF-Lite session (test 12 skipped)                         **\n");
      printf("******************************************************************\n");
   }
   //
   // Register tests
   //
   TList *testList = new TList;
   // Simple open
   PT_Open_Args_t PToa = { url, nwrks };
   testList->Add(new ProofTest("Open a session", 1, &PT_Open, (void *)&PToa));
   // Get logs
   testList->Add(new ProofTest("Get session logs", 2, &PT_GetLogs, (void *)&PToa));
   // Simple histogram generation
   testList->Add(new ProofTest("Simple random number generation", 3, &PT_Simple));
   // Test of data set handling with the H1 http files
   testList->Add(new ProofTest("Dataset handling with H1 files", 4, &PT_DataSets));
   // H1 analysis over HTTP (chain)
   testList->Add(new ProofTest("H1: chain processing", 5, &PT_H1Http));
   // H1 analysis over HTTP (file collection)
   testList->Add(new ProofTest("H1: file collection processing", 6, &PT_H1FileCollection));
   // H1 analysis over HTTP: classic packetizer
   testList->Add(new ProofTest("H1: file collection, TPacketizer", 7, &PT_H1FileCollection, (void *)&gStd_Old));
   // H1 analysis over HTTP by dataset name
   testList->Add(new ProofTest("H1: by-name processing", 8, &PT_H1DataSet));
   // Test of data set handling with the H1 http files
   testList->Add(new ProofTest("Package management with 'event'", 9, &PT_Packages));
   // Simple event analysis
   testList->Add(new ProofTest("Simple 'event' generation", 10, &PT_Event));
   // Test input data propagation (it only works in the static startup mode)
   testList->Add(new ProofTest("Input data propagation", 11, &PT_InputData));
   // Test asynchronous running
   testList->Add(new ProofTest("H1, Simple: async mode", 12, &PT_H1SimpleAsync));

   //
   // Run the tests
   //
   Bool_t failed = kFALSE;
   TIter nxt(testList);
   ProofTest *t = 0;
   while ((t = (ProofTest *)nxt()))
      if (t->Run() < 0) {
         failed = kTRUE;
         break;
      }

   // Done
   if (failed) {
      Bool_t kept = kTRUE;
      if (usedeflog) {
         char *answer = Getline(" Some tests failed: would you like to keep the log file (N,Y)? [Y] ");
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
Int_t PT_CheckH1(TQueryResult *qr)
{
   // Check the result of the H1 analysis

   if (!qr) {
      printf("\n >>> Test failure: output list not found\n");
      return -1;
   }

   // Make sure the number of processed entries is the one expected
   PutPoint();
   if (qr->GetEntries() != 283813) {
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
   TString tutdir, tmpdir(gSystem->TempDirectory()), us;
   UserGroup_t *ug = gSystem->GetUserInfo(gSystem->GetUid());
   if (!ug) {
      printf("\n >>> Test failure: could not get user info");
      return -1;
   }
   us.Form("/%s", ug->fUser.Data());
   if (!tmpdir.EndsWith(us.Data())) tmpdir += us;
   tutdir.Form("%s/.proof-tutorial", tmpdir.Data());
   if (gSystem->AccessPathName(tutdir)) {
      if (gSystem->mkdir(tutdir, kTRUE) != 0) {
         printf("\n >>> Test failure: could not assert/create the temporary directory"
                " for the tutorial (%s)", tutdir.Data());
         return -1;
      }
   }

   // String to initialize the dataset manager
   TString dsetmgrstr;
   dsetmgrstr.Form("file dir:%s/datasets opt:-Cq:As:Sb:", tutdir.Data());
   gEnv->SetValue("Proof.DataSetManager", dsetmgrstr.Data());

   // String to initialize the package dir
   TString packdir;
   packdir.Form("%s/packages", tutdir.Data());
   gEnv->SetValue("Proof.PackageDir", packdir.Data());

   // Get the PROOF Session
   PutPoint();
   TProof *p = getProof(PToa->url, PToa->nwrks, tutdir.Data(), "force", gDynamicStartup);
   if (!p || !(p->IsValid())) {
      printf("\n >>> Test failure: could not start the session\n");
      return -1;
   }

   PutPoint();
   if (PToa->nwrks > 0 && p->GetParallel() != PToa->nwrks) {
      printf("\n >>> Test failure: number of workers different from requested\n");
      return -1;
   }

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
Int_t PT_Simple(void *)
{
   // Test run for the ProofSimple analysis (see tutorials)

   // Checking arguments
   PutPoint();
   if (!gProof) {
      printf("\n >>> Test failure: no PROOF session found\n");
      return -1;
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
   gProof->Process("../tutorials/proof/ProofSimple.C++", nevt);
   gTimer.Stop();
   gProof->SetPrintProgress(0);

   // Count
   gSimpleCnt++;
   gSimpleTime += gTimer.RealTime();

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

   // Create the chain
   PutPoint();
   TChain *chain = new TChain("h42");
   chain->Add("http://root.cern.ch/files/h1/dstarmb.root");
   chain->Add("http://root.cern.ch/files/h1/dstarp1a.root");
   chain->Add("http://root.cern.ch/files/h1/dstarp1b.root");
   chain->Add("http://root.cern.ch/files/h1/dstarp2.root");

   // Clear the list of query results
   if (gProof->GetQueryResults()) gProof->GetQueryResults()->Clear();

   // Process
   PutPoint();
   chain->SetProof();
   PutPoint();
   gProof->SetPrintProgress(&PrintStressProgress);
   gTimer.Start();
   chain->Process("../tutorials/tree/h1analysis.C++");
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
   fc->Add(new TFileInfo("http://root.cern.ch/files/h1/dstarmb.root"));
   fc->Add(new TFileInfo("http://root.cern.ch/files/h1/dstarp1a.root"));
   fc->Add(new TFileInfo("http://root.cern.ch/files/h1/dstarp1b.root"));
   fc->Add(new TFileInfo("http://root.cern.ch/files/h1/dstarp2.root"));

   // Clear the list of query results
   if (gProof->GetQueryResults()) gProof->GetQueryResults()->Clear();

   // Process
   PutPoint();
   gProof->SetPrintProgress(&PrintStressProgress);
   gTimer.Start();
   gProof->Process(fc, "../tutorials/tree/h1analysis.C++");
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

   // Name for the target dataset
   const char *dsname = "h1http";

   // Clear the list of query results
   if (gProof->GetQueryResults()) gProof->GetQueryResults()->Clear();

   // Process the dataset by name
   PutPoint();
   gProof->SetPrintProgress(&PrintStressProgress);
   gTimer.Start();
   gProof->Process(dsname, "../tutorials/tree/h1analysis.C++");
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
   fc->Add(new TFileInfo("http://root.cern.ch/files/h1/dstarmb.root"));
   fc->Add(new TFileInfo("http://root.cern.ch/files/h1/dstarp1a.root"));
   fc->Add(new TFileInfo("http://root.cern.ch/files/h1/dstarp1b.root"));
   fc->Add(new TFileInfo("http://root.cern.ch/files/h1/dstarp2.root"));
   fc->Update();

   // Name for this dataset
   const char *dsname = "h1http";

   // Register the dataset
   PutPoint();
   gProof->RegisterDataSet(dsname, fc);
   // Check the result
   dsm = gProof->GetDataSets();
   if (!dsm || dsm->GetSize() != 1) {
      printf("\n >>> Test failure: could not register '%s' (%p)\n", dsname, dsm);
      delete dsm;
      return -1;
   }
   delete dsm;

   // Test removal
   PutPoint();
   gProof->RemoveDataSet(dsname);
   // Check the result
   dsm = gProof->GetDataSets();
   if (!dsm || dsm->GetSize() != 0) {
      printf("\n >>> Test failure: could not cleanup '%s' (%p)\n", dsname, dsm);
      delete dsm;
      return -1;
   }
   delete dsm;

   // Re-register the dataset
   PutPoint();
   gProof->RegisterDataSet(dsname, fc);
   // Check the result
   dsm = gProof->GetDataSets();
   if (dsm->GetSize() != 1) {
      printf("\n >>> Test failure: could not re-register '%s' (%p)\n", dsname, dsm);
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
   gProof->ShowDataSets();

   // Remove the file collection
   delete fc;

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
   const char *packpath = "../tutorials/proof/event.par";

   // Upload the package
   PutPoint();
   gProof->UploadPackage(packpath);
   // Check the result
   packs = gProof->GetListOfPackages();
   if (!packs || packs->GetSize() != 1) {
      printf("\n >>> Test failure: could not upload '%s'!\n", packpath);
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
      printf("\n >>> Test failure: could not re-upload '%s'!\n", packpath);
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
   gProof->Process("../tutorials/proof/ProofEvent.C++", nevt);
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
   gProof->AddInput(new TNamed("ProofTests_type", "InputData"));

   // Define the number of events
   Long64_t nevt = 1;

   // Clear the list of query results
   if (gProof->GetQueryResults()) gProof->GetQueryResults()->Clear();

   // Process
   PutPoint();
   gProof->SetPrintProgress(&PrintStressProgress);
   gProof->Process("../tutorials/proof/ProofTests.C++", nevt);
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
   fc->Add(new TFileInfo("http://root.cern.ch/files/h1/dstarmb.root"));
   fc->Add(new TFileInfo("http://root.cern.ch/files/h1/dstarp1a.root"));
   fc->Add(new TFileInfo("http://root.cern.ch/files/h1/dstarp1b.root"));
   fc->Add(new TFileInfo("http://root.cern.ch/files/h1/dstarp2.root"));

   // Clear the list of query results
   PutPoint();
   if (gProof->GetQueryResults()) {
      gProof->GetQueryResults()->Clear();
      gProof->Remove("cleanupdir");
   }

   // Submit the processing requests
   PutPoint();
   gProof->SetPrintProgress(&PrintStressProgress);
   gProof->Process(fc, "../tutorials/tree/h1analysis.C++", "ASYN");

   // Define the number of events and histos
   Long64_t nevt = 1000000;
   Int_t nhist = 16;
   // The number of histograms is added as parameter in the input list
   gProof->SetParameter("ProofSimple_NHist", (Long_t)nhist);
   gProof->Process("../tutorials/proof/ProofSimple.C++", nevt, "ASYN");

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


