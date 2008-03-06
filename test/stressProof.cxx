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
// *  $ ./stressProof [-h] [-n <wrks>] [-v[v[v]]] [-l logfile] [master]    * //
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
// *                                                                       * //
// * To run interactively:                                                 * //
// * $ root                                                                * //
// * root[] .L stressProof.cxx                                             * //
// * root[] stressProof(master, wrks, verbose, logfile)                    * //
// *                                                                       * //
// * The arguments have the same meaning as above.                         * //
// *                                                                       * //
// * The successful output looks like this:                                * //
// *                                                                       * //
// *  ******************************************************************   * //
// *  *  Starting  P R O O F - S T R E S S suite                       *   * //
// *  ******************************************************************   * //
// *  *  Log file: /tmp/ProofStress_XrcwBe                                 * //
// *  ******************************************************************   * //
// *   Test  1 : Open ............................................. OK *   * //
// *   Test  2 : GetLogs .......................................... OK *   * //
// *   Test  3 : Simple ........................................... OK *   * //
// *   Test  4 : H1:http .......................................... OK *   * //
// *  * All registered tests have been passed  :-)                     *   * //
// *  ******************************************************************   * //
// *                                                                       * //
// * The application redirects the processing logs to a log file which is  * //
// * normally deleted at the end of a successful run; if the test fails    * //
// * the caller is asked if she/he wants to keep the log file; if the      * //
// * specifies a log file path of her/his choice, the log file is never    * //
// * deleted.
// *                                                                       * //
// * New tests can be easily added by providing a function performing the  * //
// * test and a name for the test; see examples below.                     * //
// *                                                                       * //
// ************************************************************************* //

#include <stdio.h>
#include <stdlib.h>

#include "Getline.h"
#include "TChain.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TMath.h"
#include "TProof.h"
#include "TProofLog.h"
#include "TProofMgr.h"
#include "TString.h"
#include "TSystem.h"
#include "TROOT.h"

#include "../tutorials/proof/getProof.C"

static const char *urldef = "proof://localhost:11093";
static Int_t gverbose = 0;
static TString glogfile = 0;
static Int_t gpoints = 0;
static Int_t totpoints = 53;

void stressProof(const char *url = "proof://localhost:11093",
                 Int_t nwrks = -1, Int_t verbose = 0, const char *logfile = 0);

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
      printf(" $ ./stressProof [-h] [-n <wrks>] [-v[v[v]]] [-l logfile] [master]\n");
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
      } else {
         url = argv[i];
         i++;
      }
   }
   // Use defaults where required
   if (!url) url = urldef;

   stressProof(url, nWrks, verbose, logfile);

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

//
// Auxilliary classes for testing
//
typedef Int_t (*ProofTestFun_t)(void *);
class ProofTest : public TNamed {
private:
   Int_t           fSeq;  // Sequential number for the test
   ProofTestFun_t  fFun;  // Function to be executed for the test
   void           *fArgs; // Arguments to be passed tothe function

public:
   ProofTest(const char *n, Int_t seq, ProofTestFun_t f, void *a = 0)
           : TNamed(n,""), fSeq(seq), fFun(f), fArgs(a) { }
   virtual ~ProofTest() { }

   Int_t Run();
};

//_____________________________________________________________________________
Int_t ProofTest::Run()
{
   // Generic stress steering function; returns 0 on success, -1 on error

   gpoints = 0;
   printf(" Test %2d : %s ", fSeq, GetName());
   PutPoint();
   RedirectHandle_t rh;
   gSystem->RedirectOutput(glogfile, "a", &rh);
   Int_t rc = (*fFun)(fArgs);
   gSystem->RedirectOutput(0, 0, &rh);
   if (rc == 0) {
      Int_t np = totpoints - strlen(GetName()) - strlen(" OK *");
      while (np--) { printf("."); }
      printf(" OK *\n");
   } else {
      Int_t np = totpoints - strlen(GetName()) - strlen(" FAILED *");
      while (np--) { printf("."); }
      printf(" FAILED *\n");
      gSystem->ShowOutput(&rh);
   }
   // Done
   return rc;
}

// Test functions
Int_t PT_Open(void *);
Int_t PT_GetLogs(void *);
Int_t PT_Simple(void *);
Int_t PT_H1Http(void *);

// Arguments structures
typedef struct {            // Open
   const char *url;
   Int_t       nwrks;
} PT_Open_Args_t;

//_____________________________________________________________________________
void stressProof(const char *url, Int_t nwrks, Int_t verbose, const char *logfile)
{
   printf("******************************************************************\n");
   printf("*  Starting  P R O O F - S T R E S S  suite                      *\n");
   printf("******************************************************************\n");

   // Set verbosity
   gverbose = verbose;

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

   //
   // Register tests
   //
   TList *testList = new TList;
   // Simple open
   PT_Open_Args_t PToa = { url, nwrks };
   testList->Add(new ProofTest("Open", 1, &PT_Open, (void *)&PToa));
   // Get logs
   testList->Add(new ProofTest("GetLogs", 2, &PT_GetLogs, (void *)&PToa));
   // Simple histogram generation
   testList->Add(new ProofTest("Simple", 3, &PT_Simple));
   // H1 analysis over HTTP
   testList->Add(new ProofTest("H1:http", 4, &PT_H1Http));

   //
   // Run the tests
   //
   Bool_t failed = kFALSE;
   TIter nxt(testList);
   ProofTest *t = 0;
   while ((t = (ProofTest *)nxt()))
      if (t->Run() != 0) {
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
   TString tutdir = Form("%s/.proof-tutorial", gSystem->TempDirectory());
   if (gSystem->AccessPathName(tutdir)) {
      if (gSystem->mkdir(tutdir, kTRUE) != 0) {
         printf("\n >>> Test failure: could not assert/create the temporary directory"
                " for the tutorial (%s)", tutdir.Data());
         return -1;
      }
   }

   // Get the PROOF Session
   PutPoint();
   TProof *p = getProof(PToa->url, PToa->nwrks, tutdir.Data(), "force");
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
Int_t PT_H1Http(void *)
{
   // Test run for the H1 analysis reading the data from HTTP

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

   // Process
   PutPoint();
   chain->SetProof();
   PutPoint();
   chain->Process("../tutorials/tree/h1analysis.C++");
   gProof->RemoveChain(chain);

   // Make sure the output list is there
   PutPoint();
   if (!(gProof->GetOutputList())) {
      printf("\n >>> Test failure: output list not found\n");
      return -1;
   }

   // Check the 'hdmd' histo
   PutPoint();
   TH1F *hdmd = dynamic_cast<TH1F*>(gProof->GetOutputList()->FindObject("hdmd"));
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
   TH2F *h2 = dynamic_cast<TH2F*>(gProof->GetOutputList()->FindObject("h2"));
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
   // Process
   PutPoint();
   gProof->Process("../tutorials/proof/ProofSimple.C++", nevt);

   // Make sure the output list is there
   PutPoint();
   if (!(gProof->GetOutputList())) {
      printf("\n >>> Test failure: output list not found\n");
      return -1;
   }

   // Get the histos
   PutPoint();
   TH1F **hist = new TH1F*[nhist];
   for (Int_t i=0; i < nhist; i++) {
      hist[i] = dynamic_cast<TH1F *>(gProof->GetOutputList()->FindObject(Form("h%d",i)));
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


