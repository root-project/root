//
// Macro to run examples of analysis on PROOF.
// This macro uses an existing PROOF session or starts one at the indicated URL.
// In the case non existing PROOF session is found and no URL is given, the macro
// tries to start a local PROOF session.
//
// To run the macro:
//
//   root[] .L proof/runProof.C+
//   root[] runProof("<analysis>")
//
//   Currently available analysis are:
//
//   1. "simple"
//
//      root[] runProof("simple")
//
//      This will create a local PROOF session and run an analysis filling 100 histos
//      with 100000 gaussian random numbers, and displaying them in a canvas with 100
//      pads (10x10).
//      The number of entries and the number of histograms can be passed as 'arguments'
//      to 'simple': e.g. to fill 16 histos with 1000000 entries use
//
//      root[] runProof("simple(nevt=1000000,nhist=16)")
//
//   2. "h1"
//
//      root[] runProof("h1")
//
//      This runs the 'famous' H1 analysis from $ROOTSYS/tree/h1analysis.C,.h .
//      The data are read from the HTTP server at root.cern.ch .
//
//   In all cases, to run in non blocking mode the option 'asyn' is available, e.g.
//
//   root[] runProof("h1(asyn)")
//
//   In all cases, to run on a remote Proof cluster, the master URL must be passed as
//   second argument; e.g.
//
//   root[] runProof("simple","master.domain")
//
//   In the case of local running it is possible to specify the number of workers to
//   start as third argument (the default is the number of cores of the machine), e.g.
//
//   root[] runProof("simple",0,4)
//
//   will start 4 workers. Note that the real number of workers is changed only the
//   first time you call runProof into a ROOT session; following calls can reduce the
//   number of active workers, but not increase it. For example, in the same session of
//   the call above starting 4 workers, this
//
//   root[] runProof("simple",0,8)
//
//   will still use 4 workers, while this
//
//   root[] runProof("simple",0,2)
//
//   will disable 2 workers and use the other 2.
//


#include "TChain.h"
#include "TEnv.h"
#include "TProof.h"
#include "TString.h"
#include "TDrawFeedback.h"
#include "TList.h"

#include "getProof.C"

void runProof(const char *what = "simple",
              const char *url = "proof://localhost:11093",
              Int_t nwrks = -1)
{
   gEnv->SetValue("Proof.StatsHist",1);

   // Temp dir for PROOF tutorials
   TString tutdir = Form("%s/.proof-tutorial", gSystem->TempDirectory());
   if (gSystem->AccessPathName(tutdir)) {
      Printf("runProof: creating the temporary directory"
                " for the tutorial (%s) ... ", tutdir.Data());
      if (gSystem->mkdir(tutdir, kTRUE) != 0) {
         Printf("runProof: could not assert / create the temporary directory"
                " for the tutorial (%s)", tutdir.Data());
         return;
      }
   }

   // Get the PROOF Session
   TProof *proof = getProof(url, nwrks, tutdir.Data(), "ask");
   if (!proof) {
      Printf("runProof: could not start/attach a PROOF session");
      return;
   }
   TString proofsessions(Form("%s/sessions",tutdir.Data()));
   // Save tag of the used session
   FILE *fs = fopen(proofsessions.Data(), "a");
   if (!fs) {
      Printf("runProof: could not create files for sessions tags");
   } else {
      fprintf(fs,"session-%s\n", proof->GetSessionTag());
      fclose(fs);
   }
   if (!proof) {
      Printf("runProof: could not start/attach a PROOF session");
      return;
   }

   // Set the number of workers (may only reduce the number of active workers
   // in the session)
   if (nwrks > 0)
      proof->SetParallel(nwrks);

   // Where is the code to run
   char *rootbin = gSystem->Which(gSystem->Getenv("PATH"), "root.exe", kExecutePermission);
   if (!rootbin) {
      Printf("runProof: root.exe not found: please check the environment!");
      return;
   }
   TString rootsys(gSystem->DirName(rootbin));
   rootsys = gSystem->DirName(rootsys);
   TString tutorials(Form("%s/tutorials", rootsys.Data()));
   delete[] rootbin;

   // Create feedback displayer
   TDrawFeedback fb(proof);

   // Number of events per worker
   proof->AddFeedback("PROOF_EventsHist");

   // Have constant progress reporting based on estimated info
   proof->SetParameter("PROOF_RateEstimation", "average");

   // Parse 'what'; it is in the form 'analysis(arg1,arg2,...)'
   TString args(what);
   args.ReplaceAll("("," "); 
   args.ReplaceAll(")"," "); 
   args.ReplaceAll(","," "); 
   Ssiz_t from = 0;
   TString act, tok;
   if (!args.Tokenize(act, from, " ")) {
      // Cannot continue
      Printf("runProof: action not found: check your arguments (%s)", what);
      return;
   }

   // Action
   if (act == "simple") {
      // ProofSimple is an example of non-data driven analysis; it
      // creates and fills with random numbers a given number of histos
      TString aNevt, aNhist, opt;
      while (args.Tokenize(tok, from, " ")) {
         // Number of events
         if (tok.BeginsWith("nevt=")) {
            aNevt = tok;
            aNevt.ReplaceAll("nevt=","");
            if (!aNevt.IsDigit()) {
               Printf("runProof: error parsing the 'nevt=' option (%s) - ignoring", tok.Data());
               aNevt = "";
            }
         }
         // Number of histos
         if (tok.BeginsWith("nhist=")) {
            aNhist = tok;
            aNhist.ReplaceAll("nhist=","");
            if (!aNhist.IsDigit()) {
               Printf("runProof: error parsing the 'nhist=' option (%s) - ignoring", tok.Data());
               aNhist = "";
            }
         }
         // Sync or async ?
         if (tok.BeginsWith("asyn"))
            opt = "ASYN";
      }
      Long64_t nevt = (aNevt.IsNull()) ? 100000 : aNevt.Atoi();
      Int_t nhist = (aNhist.IsNull()) ? 100 : aNhist.Atoi();
      Printf("\nrunProof: running \"simple\" with nhist= %d and nevt= %d\n", nhist, nevt);

      // The number of histograms is added as parameter in the input list
      proof->SetParameter("ProofSimple_NHist", (Long_t)nhist);
      // The selector string
      TString sel = Form("%s/proof/ProofSimple.C+", tutorials.Data());
      //
      // Run it for nevt times
      proof->Process(sel.Data(), nevt, opt);

   } else if (act == "h1") {
      // This is the famous 'h1' example analysis run on Proof reading the
      // data from the ROOT http server.
      TString opt;
      while (args.Tokenize(tok, from, " ")) {
         // Sync or async ?
         if (tok.BeginsWith("asyn"))
            opt = "ASYN";
      }

      // Create the chain
      TChain *chain = new TChain("h42");
      chain->Add("http://root.cern.ch/files/h1/dstarmb.root");
      chain->Add("http://root.cern.ch/files/h1/dstarp1a.root");
      chain->Add("http://root.cern.ch/files/h1/dstarp1b.root");
      chain->Add("http://root.cern.ch/files/h1/dstarp2.root");
      // We run on Proof
      chain->SetProof();
      // The selector
      TString sel = Form("%s/tree/h1analysis.C+", tutorials.Data());
      // Run it for 10000 times
      Printf("\nrunProof: running \"h1\"\n");
      chain->Process(sel.Data(),opt);
   } else {
      // Do not know what to run
      Printf("runProof: unknown tutorial: %s", what);
   }
}
