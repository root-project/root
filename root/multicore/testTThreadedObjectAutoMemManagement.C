int testTThreadedObjectAutoMemManagement() {
   ROOT::EnableThreadSafety();
   const auto filename = "testTThreadedObjectAutoMemManagement.root";
   const auto nentries = 5000;
   {
      TFile f(filename,"RECREATE");
      ROOT::TThreadedObject<TH1F> h("h","h",64,-2,2);
      auto fillh = [&h](){
         h->FillRandom("gaus",nentries);
      };
      auto t1 = std::thread(fillh);
      auto t2 = std::thread(fillh);
      fillh();
      t1.join();
      t2.join();
      auto mh = h.Merge();
      mh->Write();
   }
   // Check content
   TFile f(filename);
   TH1F* h;
   f.GetObject("h",h);
   if (!h) {
      std::cerr << "Cannot find merged histo on disk!\n";
      return 1;
   }

   if (nentries*3 != h->GetEntries()) {
      std::cerr << "Wrong number of entries: " << h->GetEntries() << "!\n";
      return 1;
   }
   return 0;
}
