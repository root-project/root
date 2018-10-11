/// \file
/// \ingroup tutorial_multicore
/// \notebook
/// Parallel fill of a histogram.
/// This tutorial shows how a histogram can be filled in parallel
/// with a multithreaded approach. The difference with the multiprocess case,
/// see mp201, is that here we cannot count on the copy-on-write mechanism, but
/// we rather need to protect the histogram resource with a TThreadedObject
/// class. The result of the filling is monitored with the *SnapshotMerge*
/// method. This method is not thread safe: in presence of ROOT histograms, the
/// system will not crash but the result is not uniquely defined.
///
/// \macro_image
/// \macro_code
///
/// \date January 2016
/// \author Danilo Piparo

const UInt_t poolSize = 4U;

Int_t mt201_parallelHistoFill()
{
   ROOT::EnableThreadSafety();

   // The concrete histogram instances are created in each thread
   // lazily, i.e. only if a method is invoked.
   ROOT::TThreadedObject<TH1F> ts_h("myHist", "Filled in parallel", 128, -8, 8);

   // The function used to fill the histograms in each thread.
   auto fillRandomHisto = [&](int seed = 0) {
      TRandom3 rndm(seed);
      // IMPORTANT!
      // It is important to realise that a copy on the stack of the object we
      // would like to perform operations on is the most efficient way of
      // accessing it, in particular in presence of a tight loop like the one
      // below where any overhead put on top of the Fill function call would
      // have an impact.
      auto histogram = ts_h.Get();
      for (auto i : ROOT::TSeqI(1000000)) {
         histogram->Fill(rndm.Gaus(0, 1));
      }
   };

   // The seeds for the random number generators.
   auto seeds = ROOT::TSeqI(1, poolSize + 1);

   std::vector<std::thread> pool;

   // A monitoring thread. This is here only to illustrate the functionality of
   // the SnapshotMerge method.
   // It allows "to spy" the multithreaded calculation without the need
   // of interrupting it.
   auto monitor = [&]() {
      for (auto i : ROOT::TSeqI(5)) {
         std::this_thread::sleep_for(std::chrono::duration<double, std::nano>(500));
         auto h = ts_h.SnapshotMerge();
         std::cout << "Entries for the snapshot " << h->GetEntries() << std::endl;
      }
   };
   pool.emplace_back(monitor);

   // The threads filling the histograms
   for (auto seed : ROOT::TSeqI(seeds)) {
      pool.emplace_back(fillRandomHisto, seed);
   }

   // Wait for the threads to finish
   for (auto &&t : pool)
      t.join();

   // Merge the final result
   auto sumRandomHisto = ts_h.Merge();

   std::cout << "Entries for the total sum " << sumRandomHisto->GetEntries() << std::endl;

   auto c = new TCanvas();
   sumRandomHisto->DrawClone();
   return 0;
}
