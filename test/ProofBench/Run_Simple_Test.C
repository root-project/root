// $Id: Run_Simple_Test.C,v 1.1 2004/12/17 23:09:32 brun Exp $
//
//

// Preparation:
// - configure authentication
// - configure $ROOTSYS/proof/etc/proof.conf
// - optional: change the data directory below (needs 350 Mb)
// - start rootd
// - start proofd

#include "make_event_trees.C"
#include "make_tdset.C"

const char data_dir[] = "/var/tmp";
const bool show_feedback = true;
const bool show_stats = true;


void Run_Simple_Test()
{

   gSystem->Exec("cd $ROOTSYS/test; make libEvent.so");
   gSystem->Exec("./make_event_par.sh");
   gSystem->Exec("ln -s ../Event.h");
   gSystem->Load("../libEvent.so");

   gROOT->Proof();

   gProof->UploadPackage("event.par");
   gProof->EnablePackage("event");

   make_event_trees(data_dir, 20000, 2);

   TDSet *d = make_tdset("/data1/tmp",1);

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

   d->Process("EventTree_Proc.C+","");

   if (fb) delete fb;
}
