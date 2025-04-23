int execCacheRange() {


   std::vector<int> vlarge;
   for (int i = 0; i < 100; ++i)
      vlarge.push_back(3 * i);

   auto f2 = new TMemFile("largeStandaloneBasket.root", "RECREATE", "", 0);
   auto t2 = new TTree("largeStandaloneBasket", "");
   t2->Branch("vlarge", &vlarge);

   t2->SetAutoFlush(10);

   for (int i = 0; i < 3000; ++i) {
      t2->Fill();
   }
   f2->Write();

   t2->AddBranchToCache("vlarge");
   t2->StopCacheLearningPhase();

   t2->SetCacheEntryRange(1000,2000);

   t2->GetEntry(1000);
   t2->GetEntry(10);
   t2->GetEntry(1010);
   t2->GetEntry(2020);
   t2->GetEntry(20);
   t2->GetEntry(1020);
   t2->GetEntry(30);

   auto c2 = f2->GetCacheRead(t2);
   if (!c2) {
      Error("execCacheRange", "No TTreeCache found");
      return 1;
   }
   auto notCached = c2->GetNoCacheReadCalls();
   if (notCached != 4) {
      Error("execCacheRange", "Incorrect number of read not cached %d vs %d expected",
            notCached, 4);
      return 2;
   }
   return 0;
}
