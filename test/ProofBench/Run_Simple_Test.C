// $Id:$
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
void Run_Simple_Test()
{

   gSystem->Exec("cd $ROOTSYS/test; make libEvent.so");
   gSystem->Exec("./make_event_par.sh");
   gSystem->Exec("ln -s ../Event.h");

   gROOT->Proof();

   make_event_trees(data_dir, 100000, 2);
   TDSet *d = make_tdset("/data1/tmp",1);

   d->Print("a");

   gEnv->SetValue("Proof.StatsHist",1);


   gProof->UploadPackage("event.par");
   gProof->EnablePackage("event");

   gProof->AddFeedback("PROOF_ProcTimeHist");
   gProof->AddFeedback("PROOF_LatencyHist");
   gProof->AddFeedback("PROOF_EventsHist");

   TDrawFeedback fb(gProof);


   d->Process("EventTree_Proc.C","");
}
