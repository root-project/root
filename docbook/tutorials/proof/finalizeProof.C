//
// Macro to finalize queries run with the macro tutorials/runProof .
// This macro uses an existing PROOF session or starts one at the indicated URL.
// In the case non existing PROOF session is found and no URL is given, the macro
// tries to start a local PROOF session.
//
// To run the macro:
//
//   root[] .L proof/finalizeProof.C+
//   root[] finalizeProof("<analysis>")
//
//   See runProof.C for the analysis currently available.
//
//   The macro looks for the last completed queries for the chosen analysis and
//   asks which one to finalize. If there is only available, it finalizes it
//   without asking.
//   All queries are considered for this, both those run synchronously and those
//   run asynchronously, e.g. runProof("h1(asyn)").
//

#include "Getline.h"
#include "TChain.h"
#include "TEnv.h"
#include "TProof.h"
#include "TString.h"
#include "TDrawFeedback.h"
#include "TList.h"
#include "TQueryResult.h"
#include "TObjArray.h"

#include "getProof.C"

void finalizeProof(const char *what = "simple",
                   const char *url = "proof://localhost:11093",
                   Int_t nwrks = -1)
{

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

   // Get / Attach-to the PROOF Session
   TProof *proof = getProof(url, nwrks, tutdir.Data(), "");
   if (!proof) {
      Printf("runProof: could not start/attach a PROOF session");
      return;
   }

   // Get the last session run for the tutorial
   TObjArray *qt = new TObjArray();
   TString lasttag;
   TString proofsessions(Form("%s/sessions",tutdir.Data()));
   // Save tag of the used session
   FILE *fs = fopen(proofsessions.Data(), "r");
   if (!fs) {
      Printf("runProof: could not create files for sessions tags");
   } else {
      char line[1024];
      while (fgets(line, sizeof(line), fs)) {
         int l = strlen(line);
         if (l <= 0) continue;
         if (strncmp(line,"session-",strlen("session-"))) continue;
         if (line[l-1] == '\n') line[l-1] = 0;
         lasttag = line;
         qt->Add(new TObjString(lasttag.Data()));
      }
      fclose(fs);
   }

   // Retrieve the list of available query results
   TList *ql = proof->GetListOfQueries("A");
   if (!ql || ql->GetSize() <= 0) {
      Printf("runProof: no queries to be finalized");
      return;
   }
   ql->Print();

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

   TObjArray *qa = new TObjArray();
   TString sel;
   // Action
   if (act == "simple") {
      sel = "ProofSimple";
   } else if (act == "h1") {
      sel = "h1analysis";
   } else if (act == "pythia8") {
      sel = "ProofPythia";
   } else {
      // Do not know what to run
      Printf("runProof: unknown tutorial: %s", what);
   }

   // Get last completed queries for the chosen analysis
   TString ref;
   Int_t nt = qt->GetEntriesFast();
   while (ref.IsNull() && nt--) {
      lasttag = ((TObjString *)(qt->At(nt)))->GetName();
      if (!lasttag.IsNull())
         Printf("runProof: checking session: %s", lasttag.Data());
      TIter nxq(ql);
      TQueryResult *qr = 0;
      while ((qr = (TQueryResult *)nxq())) {
         if (qr->IsDone() && !lasttag.CompareTo(qr->GetTitle()) &&
                           !sel.CompareTo(qr->GetSelecImp()->GetTitle())) {
            TString r = Form("%s:%s",qr->GetTitle(),qr->GetName());
            qa->Add(new TObjString(r.Data()));
         }
      }
      if (qa->GetEntriesFast() > 0) {
         Int_t qn = 0;
         if (qa->GetEntriesFast() > 1) {
            // Query the client which query to finalize
            Printf("finalizeProof: queries completed for analysis '%s'", act.Data());
            for (Int_t k = 0; k < qa->GetEntriesFast(); k++) {
               Printf(" [%d] %s", k, ((TObjString *)(qa->At(k)))->GetName());
            }
            Bool_t ask = kTRUE;
            while (ask) {
               char *answer = Getline("finalizeProof: enter the one you would like to finalize? [0] ");
               if (answer) {
                  if (answer[0] == 'Q' || answer[0] == 'q') {
                     ask = kFALSE;
                     return;
                  }
                  TString sn(answer);
                  sn.Remove(sn.Length()-1);
                  if (sn.IsDigit()) {
                     qn = sn.Atoi();
                     if (qn >= 0 && qn < qa->GetEntriesFast()) {
                        break;
                     } else {
                        Printf("finalizeProof: choice must be in [0,%d] ('Q' to quit)",
                              qa->GetEntriesFast()-1);
                     }
                  } else {
                     if (sn.IsNull()) {
                        qn = 0;
                        break;
                     } else {
                        Printf("finalizeProof: choice must be a number in [0,%d] ('Q' to quit) (%s)",
                              qa->GetEntriesFast()-1,  sn.Data());
                     }
                  }
               }
            }
         }
         ref = ((TObjString *)(qa->At(qn)))->GetName();
      }
   }
   if (!ref.IsNull()) {
      // Retrieve
      proof->Retrieve(ref);
      // Finalize
      proof->Finalize(ref);
   } else {
      Printf("runProof: no queries to be finalized for analysis '%s'", act.Data());
      return;
   }
}
