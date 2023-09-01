// $Id: Run_Simple_Test.C,v 1.5 2005/02/12 02:14:54 rdm Exp $
//
//

// Preparation:
// - configure authentication
// - configure $ROOTSYS/proof/etc/proof.conf (use full domain name)
// - optional: change the data directory below (needs 350 Mb)
// - start xrootd with proof

#include "TSystem.h"
#include "TEnv.h"
#include "TProof.h"
#include "TDSet.h"
#include "TDrawFeedback.h"
#include "TList.h"

#include "make_event_trees.C"
#include "make_tdset.C"

const bool show_feedback = true;
const bool show_stats = true;
const int  files_per_node = 10;
const int  max_files_per_slave = 1;

void Run_Simple_Test(char *host, const char *data_dir = "/var/tmp", Int_t dbg = 0)
{

   if (gSystem->AccessPathName("../libEvent.so")) {
      gSystem->Exec("cd $ROOTSYS/test; make libEvent.so");
      gSystem->Exec("./make_event_par.sh");
      gSystem->Exec("ln -s ../Event.h");
      gSystem->Load("../libEvent.so");
   }

   TProof *p = 0;
   if (dbg > 0) {
      p = TProof::Open(host, "", "", dbg);
   } else {
      p = TProof::Open(host);
   }
   if (!p || !p->IsValid()) {
      cout << "Could not start the PROOF session - exit "<<endl;
      return;
   }
   if (dbg > 0) p->SetLogLevel(dbg);

   p->UploadPackage("event");
   p->EnablePackage("event");

   make_event_trees(data_dir, 100000, files_per_node);

   TDSet *d = make_tdset(data_dir, files_per_node);

   d->Print("a");

   if (show_stats) gEnv->SetValue("Proof.StatsHist", 1);

   TDrawFeedback *fb(0);
   if (show_feedback) {
      if (show_stats) {
         p->AddFeedback("PROOF_ProcTimeHist");
         p->AddFeedback("PROOF_LatencyHist");
         p->AddFeedback("PROOF_EventsHist");
      }

      p->AddFeedback("pt_dist");

      fb = new TDrawFeedback(p);
   }

   // We load the selector before execution to avoid a problem affecting some versions
   // (in particular 5.22/00 and 5.22/00a); with problem-free versions this is equivalent
   // to direct processing via d->Process("EventTree_Proc.C+","")
   p->Load("EventTree_Proc.C+");
   d->Process("EventTree_Proc");

   if (fb) delete fb;
}
