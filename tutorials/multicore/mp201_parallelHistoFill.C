/// \file
/// \ingroup tutorial_multicore
/// Parallel fill of a histogram
/// This tutorial shows how a histogram can be filled in parallel
/// with a multiprocess approach.
///
/// \macro_image
/// \macro_code
/// \author Danilo Piparo

// Measure time in a scope
class TimerRAII {
   TStopwatch fTimer;
   std::string fMeta;
public:
   TimerRAII(const char *meta): fMeta(meta)
   {
      fTimer.Start();
   }
   ~TimerRAII()
   {
      fTimer.Stop();
      std::cout << fMeta << " - real time elapsed " << fTimer.RealTime() << "s" << std::endl;
   }
};

Int_t mp201_parallelHistoFill(UInt_t poolSize = 4)
{
   TH1::AddDirectory(false);
   TProcPool pool(poolSize);
   auto fillRandomHisto = [](int seed = 0) {
      TRandom3 rndm(seed);
      auto h = new TH1F("myHist", "Filled in parallel", 128, -8, 8);
      for (auto i : ROOT::TSeqI(1000000)) {
         h->Fill(rndm.Gaus(0,1));
      }
      return h;
   };

   TimerRAII timer("Filling Histogram in parallel and drawing it.");
   auto seeds = ROOT::TSeqI(23);
   auto sumRandomHisto = pool.MapReduce(fillRandomHisto, seeds, PoolUtils::ReduceObjects);

   auto c = new TCanvas();
   sumRandomHisto->Draw();
   return 0;
}
