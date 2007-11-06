// Macro to run the ProofSimple selector.
// This macro uses an existing PROOF session or starts one at the indicated URL.
// In the case non existing PROOF session is found and no URL is given, the macro
// tries to start a local PROOF session.

#include "TChain.h"
#include "TProof.h"
#include "TString.h"

#include "getProof.C"

void runProof(const char *what = "simple",
              const char *url = refloc, Int_t nwrks = -1, const char *wrkdir = "/tmp/proof")
{
   // Get the PROOF Session
   TProof *proof = getProof(url, nwrks, wrkdir);

   // Where is the code to run
   char *rootbin = gSystem->Which(gSystem->Getenv("PATH"), "root.exe", kExecutePermission);
   if (!rootbin) {
      Printf("runProof: root.exe not found: please check the environment!");
      return;
   }
   TString tutorials(Form("%s/tutorials", gSystem->DirName(gSystem->DirName(rootbin))));
   delete[] rootbin;

   // Have constant progress reporting based on estimated info
   //   proof->SetParameter("PROOF_RateEstimation", "average");

   // Action
   if (!strcmp(what, "simple")) {
      // ProofSimple is an example of non-data driven analysis; it
      // creates and fills with random numbers, two histos and an ntuple.
      TString sel = Form("%s/proof/ProofSimple.C+", tutorials.Data());
      // Run it for 10000 times
      proof->Process(sel.Data(), 100000);
   } else if (!strcmp(what, "h1")) {
      // This is the famous 'h1' example analysis run on Proof reading the
      // data from the ROOT http server.

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
      chain->Process(sel.Data());
   }
}
