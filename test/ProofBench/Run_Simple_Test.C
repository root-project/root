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

const char data_dir[] = "/var/tmp";
const bool show_feedback = true;
const bool show_stats = true;
const int  files_per_node = 2;
const int  max_files_per_slave = 1;


void Run_Simple_Test(char *host)
{

   gSystem->Exec("cd $ROOTSYS/test; make libEvent.so");
   gSystem->Exec("./make_event_par.sh");
   gSystem->Exec("ln -s ../Event.h");
   gSystem->Load("../libEvent.so");

   TProof::Reset(host);
   TProof::Open(host);
//   gProof->SetLogLevel(2);

   gProof->UploadPackage("event.par");
   gProof->EnablePackage("event");

   make_event_trees(data_dir, 10000, files_per_node);

   TDSet *d = make_tdset(data_dir, max_files_per_slave,
                         files_per_node);

   d->Print("a");

   if (show_stats) gEnv->SetValue("Proof.StatsHist", 1);

   TDrawFeedback *fb(0);
   if (show_feedback) {
      if (show_stats) {
	 gProof->AddFeedback("PROOF_ProcTimeHist");
	 gProof->AddFeedback("PROOF_LatencyHist");
	 gProof->AddFeedback("PROOF_EventsHist");
      }

      gProof->AddFeedback("pt_dist");

      fb = new TDrawFeedback(gProof);
   }

   d->Process("EventTree_Proc.C","");

   if (fb) delete fb;
}
