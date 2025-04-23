void exec_tmpi()
{
   Int_t N_collectors = 2;
   Int_t sync_rate = 10;
   Int_t events_per_rank = 50;

   Int_t jetm = 25;
   Int_t trackm = 60;
   Int_t hitam = 200;
   Int_t hitbm = 100;

   TMPIFile *newfile = new TMPIFile("exec_tmpifile.root", "RECREATE", N_collectors);
   gRandom->SetSeed(gRandom->GetSeed() + newfile->GetMPIGlobalRank());

   if (newfile->IsCollector()) {
      newfile->RunCollector();
   } else {
      TTree *tree = new TTree("test_tmpi", "Event example with Jets");
      tree->SetAutoFlush(sync_rate);

      JetEvent *event = new JetEvent;

      tree->Branch("event", "JetEvent", &event, 8000, 2);

      for (int i = 0; i < events_per_rank; i++) {
         event->Build(jetm, trackm, hitam, hitbm);
         tree->Fill();

         if ((i + 1) % sync_rate == 0) {
            newfile->Sync();
         }
      }

      if (events_per_rank % sync_rate != 0) {
         newfile->Sync();
      }
   }

   newfile->Close();

   if (newfile->GetMPILocalRank() == 0) {
      TString filename = newfile->GetMPIFilename();
      TFile file(filename.Data());
      if (file.IsOpen()) {
         TTree *tree = (TTree *)file.Get("test_tmpi");

         Info("Rank", "[%d] [%d] file should have %d events and has %lld", newfile->GetMPIColor(),
              newfile->GetMPILocalRank(), (newfile->GetMPILocalSize() - 1) * events_per_rank, tree->GetEntries());
      }
   }
}

int execTMPIFile()
{
   exec_tmpi();

   return 0;
}
