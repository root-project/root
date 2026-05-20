// Testing what happens when the TTreeCache is too small
// in different scenarii: well clustered, badly clustered.

int largeStandaloneBasket()
{
   Printf("largeStandaloneBasket: Running test");

   std::vector<int> v1;
   for (int i = 0; i < 100; ++i)
      v1.push_back(i);

   std::vector<int> v2;
   for (int i = 0; i < 100; ++i)
      v2.push_back(2 * i);

   std::vector<int> vlarge;
   for (int i = 0; i < 100; ++i)
      vlarge.push_back(3 * i);

   auto f2 = new TMemFile("largeStandaloneBasket.root", "RECREATE", "", 0);
   auto t2 = new TTree("largeStandaloneBasket", "");
   t2->Branch("v1", &v1);
   t2->Branch("v2a", &v2);
   t2->Branch("v2b", &v2);
   t2->Branch("v2c", &v2);
   t2->SetAutoFlush(50);
   for (int i = 0; i < 3000; ++i) {
      t2->Fill();
      if (i && ((i + 1) % 25) == 0) {
         // Force more than once basket per cluster.
         t2->FlushBaskets();
      }
   }
   auto b2 = t2->Branch("vlarge", &vlarge, 1 * 45000);
   for (int i = 0; i < 3000; ++i) {
      b2->Fill();
      if (i && ((i + 1) % 1500) == 0) {
         b2->FlushBaskets();
      }
   }
   // t2->GetBranch("v1")->Print("basketsInfo");
   // t2->GetBranch("vlarge")->Print("basketsInfo");
   f2->Write();

   t2->Print();

   t2->SetCacheSize(45000);

   if (1)
      t2->AddBranchToCache("vlarge");
   else
      t2->SetBranchStatus("vlarge", false);
   t2->AddBranchToCache("v1");
   t2->AddBranchToCache("v2*");
   t2->StopCacheLearningPhase();

   auto ps2 = new TTreePerfStats("Perf Stats", t2);

   for (auto e = 0ll; e < t2->GetEntries(); ++e) {
      t2->GetEntry(e);
   }

   int result = 0;
   auto c2 = f2->GetCacheRead(t2);
   if (c2) {
      c2->Print();
      auto notCached = c2->GetNoCacheReadCalls();
      if (notCached)
         Error("largeStandaloneBasket", "For %s we got some uncached read: %d\n", t2->GetName(), notCached);
      result += notCached;
   }
   auto duplicates(ps2->GetDuplicateBasketCache());
   if (!duplicates.empty()) {
      Error("largeStandaloneBasket", "For %s: %d branches have duplicate basket reads", t2->GetName(),
            (int)duplicates.size());
   }
   ps2->Print("basket");
   ps2->SaveAs("t2.root");

   fprintf(stdout, "Cache size 2 for t2: %lld zipbytes=%lld\n", t2->GetCacheSize(), t2->GetZipBytes());

   return result;
}

int largeWithSkip(bool skip = true, bool split = true, bool wrongOrder = true, int expectedNotCached = 0)
{
   TString config;
   config.Form("skip=%d split=%d wrongOrder=%d", skip, split, wrongOrder);
   Printf("largeWithSkip: Running test: %s", config.Data());

   int result = 0;

   std::vector<int> v1;
   for (int i = 0; i < 100; ++i)
      v1.push_back(i);

   std::vector<int> v2;
   for (int i = 0; i < 100; ++i)
      v2.push_back(2 * i);

   std::vector<int> vlarge;
   for (int i = 0; i < 1000; ++i)
      vlarge.push_back(3 * i); // and only 2 other branches.

   auto f2 = new TMemFile("f2.root", "RECREATE", "", 0);
   auto t2 = new TTree("t2", "");
   t2->Branch("v1", &v1);
   t2->Branch("v2", &v2);
   // t2->Branch("v2", &v2);
   // t2->Branch("v2", &v2);
   t2->SetAutoFlush(50);
   for (int i = 0; i < 3000; ++i) {
      t2->Fill();
      if (split && i && ((i + 1) % 25) == 0) {
         // Force more than once basket per cluster.
         t2->FlushBaskets();
      }
   }
   auto b2 = t2->Branch("vlarge", &vlarge); // , 1*45000);
   for (int i = 0; i < 3000; ++i) {
      b2->Fill();
      if (i && ((i + 1) % 1500) == 0) {
         b2->FlushBaskets();
      }
   }
   // t2->GetBranch("v1")->Print("basketsInfo");
   // t2->GetBranch("vlarge")->Print("basketsInfo");
   f2->Write();

   t2->Print();
   fprintf(stdout, "Cache size 1 for t2: %lld\n", t2->GetCacheSize());

   // t2->SetCacheSize(600000 / 6);
   if (split)
      t2->SetCacheSize(45000);
   else
      t2->SetCacheSize(2 * 45000);

   if (wrongOrder) {
      if (1)
         t2->AddBranchToCache("vlarge");
      else
         t2->SetBranchStatus("vlarge", false);
      t2->AddBranchToCache("v2");
      t2->AddBranchToCache("v1");
   } else {
      t2->AddBranchToCache("v1");
      t2->AddBranchToCache("v2");
      if (1)
         t2->AddBranchToCache("vlarge");
      else
         t2->SetBranchStatus("vlarge", false);
   }
   t2->StopCacheLearningPhase();

   auto ps2 = new TTreePerfStats("Perf Stats", t2);

   // When looping over all the entries, the TTreeCache (v6.12) algorithm
   // will manage to load one basket per branch when requesting entry 0 and
   // end with a 'next' entry of 25 (since the TTree was constructed to have
   // basket of size 25 even-though the cluster size is set to 50)
   // On the second pass, when requesting entry 25, the range is set to 0
   // through 50, however since the basket for entry [0,25[ are still in memory
   // their are skipped and this second pass can fit the 2nd set of baskets.
   //
   // So a worth scenario is if the first requested entry is 25.  In that case,
   // the range is set to [0,50[ and the first baskets [0,25[ are put in the
   // cache and none of the necessary baskets [25,50[ are there.  Resulting in
   // uncached read of the only data used.

   const auto incr = skip ? 50ll : 1ll;
   for (auto e = skip ? 25ll : 0; e < t2->GetEntries(); e += incr) {
      // for (auto e = 0ll; e < t2->GetEntries(); ++e) {
      t2->GetEntry(e);
   }
   auto c2 = f2->GetCacheRead(t2);
   if (c2) {
      c2->Print();
      auto notCached = c2->GetNoCacheReadCalls();
      if (notCached > expectedNotCached)
         Error("largeWithSkip", "For %s we got some uncached read: %d\n", config.Data(), notCached);
      result += notCached;

      auto bytesReadCached = c2->GetBytesRead();
      auto bytesReadNotCached = c2->GetNoCacheBytesRead();
      auto bytesRead = bytesReadCached + bytesReadNotCached;
      auto expectedBytesRead = t2->GetTotBytes();
      if (skip && split && expectedNotCached > 0)
         expectedBytesRead = expectedBytesRead / 2;
      auto minimalBytesRead = expectedBytesRead;
      if (skip) {
         minimalBytesRead = 0;
         for (auto bo : *t2->GetListOfBranches()) {
            TBranch *b = (TBranch *)bo;
            Int_t nb = b->GetMaxBaskets();
            Int_t *lbaskets = b->GetBasketBytes();
            Long64_t *entries = b->GetBasketEntry();
            for (auto e = skip ? 25ll : 0; e < t2->GetEntries(); e += incr) {
               for (int j = 0; j < nb; ++j) {
                  if (entries[j] <= e && e <= ((j < (nb - 1)) ? (entries[j + 1]) : b->GetEntries())) {
                     minimalBytesRead += lbaskets[j];
                     break;
                  }
               }
            }
         }
//         Info("largeWithSkip", "Read %lld vs a minimal of %lld (+%lld)", bytesRead, minimalBytesRead,
//              bytesRead - minimalBytesRead);
      }
      if (bytesRead > expectedBytesRead) {
         Error("largeWithSkip", "For %s we read too much %lld instead of only %lld (expectedNotCached=%d)",
               config.Data(), bytesRead, expectedBytesRead, expectedNotCached);
         result += 100000;
      }
   }
   auto duplicates(ps2->GetDuplicateBasketCache());
   if (!duplicates.empty()) {
      Error("largeWithSkip", "For %s: %d branches have duplicate basket reads", config.Data(), (int)duplicates.size());
   }
   ps2->Print("basket");
   ps2->SaveAs("t1.root");

   t2->Print();
   fprintf(stdout, "Cache size 2 for t2: %lld zipbytes=%lld\n", t2->GetCacheSize(), t2->GetZipBytes());

   if (result <= expectedNotCached)
      return 0;
   else
      return result - expectedNotCached;
}

int tooSmall(bool skip = false, bool split = false, bool wrongOrder = false, bool haslarge = false,
             int expectedNotCached = 0)
{
   TString config;
   config.Form("skip=%d split=%d wrongOrder=%d haslarge=%d", skip, split, wrongOrder, haslarge);
   Printf("tooSmall: Running test: %s", config.Data());

   int result = 0;

   std::vector<int> v1;
   for (int i = 0; i < 100; ++i)
      v1.push_back(i);

   std::vector<int> v2;
   for (int i = 0; i < 100; ++i)
      v2.push_back(2 * i);

   std::vector<int> vlarge;
   for (int i = 0; i < 1000; ++i)
      vlarge.push_back(3 * i);

   auto f2 = new TMemFile("f2.root", "RECREATE", "", 0);
   auto t2 = new TTree("t2", "");
   t2->Branch("v1a", &v1);
   t2->Branch("v1b", &v1);
   t2->Branch("v1c", &v1);
   t2->Branch("v1d", &v1);
   t2->Branch("v1e", &v1);
   t2->Branch("v2a", &v2);
   t2->Branch("v2b", &v2);
   t2->Branch("v2c", &v2);
   t2->Branch("v2d", &v2);
   t2->Branch("v2e", &v2);

   t2->SetAutoFlush(50);

   for (int i = 0; i < 3000; ++i) {
      t2->Fill();
      if (split && i && ((i + 1) % 25) == 0) {
         // Force more than once basket per cluster.
         t2->FlushBaskets();
      }
   }
   if (haslarge) {
      auto b2 = t2->Branch("vlarge", &vlarge); // , 1*45000);
      for (int i = 0; i < 3000; ++i) {
         b2->BackFill();
         //      if ( i && ((i+1)%1500) == 0) {
         //         b2->FlushBaskets();
         //      }
      }
   }
   // t2->GetBranch("v1")->Print("basketsInfo");
   // t2->GetBranch("vlarge")->Print("basketsInfo");
   f2->Write();

   t2->Print();
   fprintf(stdout, "Cache size 1 for t2: %lld\n", t2->GetCacheSize());

   // t2->SetCacheSize(600000 / 6);
   if (split)
      t2->SetCacheSize(45000 / 2);
   // else
   //   t2->SetCacheSize(45000);

   if (wrongOrder) {
      if (haslarge)
         t2->AddBranchToCache("vlarge");
      // else
      //   t2->SetBranchStatus("vlarge", false);
      t2->AddBranchToCache("v2*");
      t2->AddBranchToCache("v1*");
   } else {
      t2->AddBranchToCache("v1*");
      t2->AddBranchToCache("v2*");
      if (haslarge)
         t2->AddBranchToCache("vlarge");
      // else
      //   t2->SetBranchStatus("vlarge", false);
   }
   t2->StopCacheLearningPhase();

   auto ps2 = new TTreePerfStats("Perf Stats", t2);

   // When looping over all the entries, the TTreeCache (v6.12) algorithm
   // will manage to load one basket per branch when requesting entry 0 and
   // end with a 'next' entry of 25 (since the TTree was constructed to have
   // basket of size 25 even-though the cluster size is set to 50)
   // On the second pass, when requesting entry 25, the range is set to 0
   // through 50, however since the basket for entry [0,25[ are still in memory
   // their are skipped and this second pass can fit the 2nd set of baskets.
   //
   // So a worth scenario is if the first requested entry is 25.  In that case,
   // the range is set to [0,50[ and the first baskets [0,25[ are put in the
   // cache and none of the necessary baskets [25,50[ are there.  Resulting in
   // uncached read of the only data used.

   const auto incr = skip ? 50ll : 1ll;
   for (auto e = skip ? 25ll : 0; e < t2->GetEntries(); e += incr) {
      // for (auto e = 0ll; e < t2->GetEntries(); ++e) {
      t2->GetEntry(e);
   }
   auto c2 = f2->GetCacheRead(t2);
   if (c2) {
      c2->Print();
      auto notCached = c2->GetNoCacheReadCalls();
      if (notCached > expectedNotCached)
         Error("tooSmall", "For %s we got some uncached read: %d\n", config.Data(), notCached);
      result += notCached;

      auto bytesReadCached = c2->GetBytesRead();
      auto bytesReadNotCached = c2->GetNoCacheBytesRead();
      auto bytesRead = bytesReadCached + bytesReadNotCached;
      auto expectedBytesRead = t2->GetTotBytes();
      if (skip && split)
         expectedBytesRead = expectedBytesRead / 2;
      auto minimalBytesRead = expectedBytesRead;
      if (skip) {
         minimalBytesRead = 0;
         auto lbr = t2->GetBranch("vlarge");
         for (auto bo : *t2->GetListOfBranches()) {
            TBranch *b = (TBranch *)bo;
            Int_t nb = b->GetMaxBaskets();
            Int_t *lbaskets = b->GetBasketBytes();
            Long64_t *entries = b->GetBasketEntry();
            for (auto e = skip ? 25ll : 0; e < t2->GetEntries(); e += incr) {
               for (int j = 0; j < nb; ++j) {
                  if (entries[j] <= e && e <= ((j < (nb - 1)) ? (entries[j + 1] - 1) : b->GetEntries() - 1)) {
                     minimalBytesRead += lbaskets[j];
                     break;
                  }
                  if (b == lbr && (entries[j] + 1) == ((j < (nb - 1)) ? (entries[j + 1]) : b->GetEntries())) {
                     // baskets from vlarge containing one entry are cached
                     // even-though they are not used (but we can't know since
                     // the TTreeCache was not told before hand which entries
                     // would be read.  This adds 60 'cached' transactions since
                     // they are also not 'next to' the other baskets because of the
                     // backfilling.
                     minimalBytesRead += lbaskets[j];
                     break;
                  }
               }
            }
         }
      }
      if (bytesRead > expectedBytesRead || bytesRead < minimalBytesRead) {
         Error("tooSmall",
               "For %s we read too much %lld instead of only %lld or minimally (%lld) (expectedNotCached=%d)",
               config.Data(), bytesRead, expectedBytesRead, minimalBytesRead, expectedNotCached);
         result += 100000;
      }
   }
   auto duplicates(ps2->GetDuplicateBasketCache());
   if (!duplicates.empty()) {
      Error("tooSmall", "For %s: %d branches have duplicate basket reads", config.Data(), (int)duplicates.size());
   }
   ps2->Print("basket");
   ps2->SaveAs("t1.root");

   t2->Print();
   fprintf(stdout, "Cache size 2 for t2: %lld zipbytes=%lld\n", t2->GetCacheSize(), t2->GetZipBytes());

   if (result <= expectedNotCached)
      return 0;
   else
      return result - expectedNotCached;
}

int regular(bool skip = false, bool wrongOrder = false, bool uselarge = true, int expectedNotCached = 0)
{

   TString config;
   config.Form("skip=%d wrongOrder=%d uselarge=%d", skip, wrongOrder, uselarge);
   Printf("regular: Running test: %s", config.Data());
   bool split = false;
   int result = 0;

   std::vector<int> v1;
   for (int i = 0; i < 100; ++i)
      v1.push_back(i);

   std::vector<int> v2;
   for (int i = 0; i < 100; ++i)
      v2.push_back(2 * i);

   std::vector<int> vlarge;
   for (int i = 0; i < 1000; ++i)
      vlarge.push_back(3 * i);
   // for (int i = 0; i < 100; ++i) vlarge.push_back(3*i);

   auto f1 = new TMemFile("f1.root", "RECREATE", "", 0);
   auto t1 = new TTree("t1", "");
   t1->Branch("v1", &v1);
   t1->Branch("v2", &v2);
   t1->SetAutoFlush(50);
   for (int i = 0; i < 3000; ++i) {
      t1->Fill();
   }
   auto b1 = t1->Branch("vlarge", &vlarge);
   for (int i = 0; i < 3000; ++i) {
      b1->BackFill();
   }
   f1->Write();

   t1->Print();
   fprintf(stdout, "Cache size 1 for t1: %lld\n", t1->GetCacheSize());

   // t1->SetCacheSize(600000 / 6);

   if (wrongOrder) {
      if (uselarge)
         t1->AddBranchToCache("vlarge");
      else
         t1->SetBranchStatus("vlarge", false);
      t1->AddBranchToCache("v2");
      t1->AddBranchToCache("v1");
   } else {
      t1->AddBranchToCache("v1");
      t1->AddBranchToCache("v2");
      if (uselarge)
         t1->AddBranchToCache("vlarge");
      else
         t1->SetBranchStatus("vlarge", false);
   }
   t1->StopCacheLearningPhase();

   auto ps1 = new TTreePerfStats("Perf Stats", t1);

   const auto incr = skip ? 50ll : 1ll;
   for (auto e = skip ? 25ll : 0; e < t1->GetEntries(); e += incr) {
      // for (auto e = 0ll; e < t2->GetEntries(); ++e) {
      t1->GetEntry(e);
   }

   auto c1 = f1->GetCacheRead(t1);
   if (c1) {
      c1->Print();

      auto notCached = c1->GetNoCacheReadCalls();
      if (notCached > expectedNotCached)
         Error("tooSmall", "For %s we got some uncached read: %d\n", config.Data(), notCached);
      result += notCached;

      auto bytesReadCached = c1->GetBytesRead();
      auto bytesReadNotCached = c1->GetNoCacheBytesRead();
      auto bytesRead = bytesReadCached + bytesReadNotCached;
      auto expectedBytesRead = t1->GetTotBytes();
      if (skip && split)
         expectedBytesRead = expectedBytesRead / 2;
      auto minimalBytesRead = expectedBytesRead;

      if (skip || !uselarge) {
         minimalBytesRead = 0;
         auto lbr = t1->GetBranch("vlarge");
         for (auto bo : *t1->GetListOfBranches()) {
            TBranch *b = (TBranch *)bo;
            if (!uselarge && b == lbr) {
               continue;
            }
            Int_t nb = b->GetMaxBaskets();
            Int_t *lbaskets = b->GetBasketBytes();
            Long64_t *entries = b->GetBasketEntry();
            std::vector<bool> seen(nb);
            for (auto e = skip ? 25ll : 0; e < t1->GetEntries(); e += incr) {
               for (int j = 0; j < nb; ++j) {
                  if (entries[j] <= e && e <= ((j < (nb - 1)) ? (entries[j + 1] - 1) : b->GetEntries() - 1) &&
                      !seen[j]) {
                     seen[j] = true;
                     minimalBytesRead += lbaskets[j];
                     break;
                  }
               }
            }
         }
         if (!uselarge)
            expectedBytesRead = minimalBytesRead;
      }
      if (bytesRead > expectedBytesRead || bytesRead < minimalBytesRead) {
         Error("regular",
               "For %s we read too much %lld instead of only %lld or minimally (%lld) (expectedNotCached=%d)",
               config.Data(), bytesRead, expectedBytesRead, minimalBytesRead, expectedNotCached);
         result += 100000;
      }
   }
   auto duplicates(ps1->GetDuplicateBasketCache());
   if (!duplicates.empty()) {
      Error("regular", "For %s: %d branches have duplicate basket reads", config.Data(), (int)duplicates.size());
   }
   ps1->Print("basket");
   ps1->SaveAs("t1.root");

   t1->Print();
   fprintf(stdout, "Cache size 2 for t1: %lld\n", t1->GetCacheSize());

   return result;
}

int assertTooSmall()
{

   int result = 0;

   result += ::regular(true, true, true);
   result += ::regular(true, true, false);
   result += ::regular(true, false, true);
   result += ::regular(true, false, false);

   result += ::regular(false, true, true);
   result += ::regular(false, true, false);
   result += ::regular(false, false, true);
   result += ::regular(false, false, false);

   result += largeStandaloneBasket();

   result += largeWithSkip(true, true, true);
   result += largeWithSkip(true, true, false);
   result += largeWithSkip(true, false, true);
   result += largeWithSkip(true, false, false);

   result += largeWithSkip(false, true, true, 0 * 120);
   result += largeWithSkip(false, true, false, 0 * 30);
   result += largeWithSkip(false, false, true);
   result += largeWithSkip(false, false, false);

   // return result;

   result += tooSmall(false, false, false, false);
   result += tooSmall(true, false, false, false);
   result += tooSmall(false, true, false, false);
   result += tooSmall(true, true, false, false);

   // Wrong order
   result += tooSmall(false, false, true, false, 0);
   result += tooSmall(true, false, true, false, 0);
   result += tooSmall(false, true, true, false, 240);
   result += tooSmall(true, true, true, false, 120);

   // With large branch

   result += tooSmall(false, false, false, true);
   result += tooSmall(true, false, false, true);
   result += tooSmall(false, true, false, true, 420);
   result += tooSmall(true, true, false, true, 60);

   // Wrong order
   result += tooSmall(false, false, true, true, 0);
   result += tooSmall(true, false, true, true, 0);
   result += tooSmall(false, true, true, true, 660);
   result += tooSmall(true, true, true, true, 180);

   return result;

   return 0;
}
