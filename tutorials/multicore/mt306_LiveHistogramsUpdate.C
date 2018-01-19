/// \file
/// \ingroup tutorial_multicore
/// Show how to fill histograms in parallel and display concurrently updated
/// canvases.
///
/// Inspired by code of Victor Perevovchikov
///
/// \macro_code
///
/// \author Danilo Piparo

std::vector<std::thread> threads;
const Long64_t nFills = 250000;
const int updateInterval = 500;
std::atomic<bool> isFinished(false);

void doWork(unsigned nr, TH1F *histo, TCanvas *can)
{
   float px, py, pz;
   TRandom3 rnd(nr + 1);
   for (auto i : ROOT::TSeqI(nFills)) {
      rnd.Rannor(px, py);
      pz = px * px + py * py;
      histo->Fill(px);
      if (i && 0 == (i % updateInterval)) {
         if (i == updateInterval) {
            // This is the lock guard which makes this section "critical".
            // Everything which happens within this scope is sequential.
            R__LOCKGUARD(gROOTMutex);
            can->cd();
            histo->Draw("histE");
         }
         if (threads[nr].joinable()) {
            can->Modified();
         }
         gSystem->Sleep(10); // Give time to the user to see the histogram being updated!
      }
   }
}

void JoinThread(unsigned index)
{
   auto &t = threads[index];
   if (t.joinable()) {
      t.join();
   }
}

// This function joins the threads which needs it and signals that the
// processing is finished. It could have been expressed as a lambda inside
// the main macro function
void Join()
{
   for (auto i : {0, 1, 2, 3}) {
      JoinThread(i);
   }
   isFinished = kTRUE;
}

void mt306_LiveHistogramsUpdate()
{
   ROOT::EnableThreadSafety();

   // Initialise the histograms
   std::vector<TH1F> histograms{{"hpx_0", "This is the px distribution", 100, -4, 4},
                                {"hpx_1", "This is the px distribution", 100, -4, 4},
                                {"hpx_2", "This is the px distribution", 100, -4, 4},
                                {"hpx_3", "This is the px distribution", 100, -4, 4}};

   // Create the canvases
   // The shared ptr is necessary since TCanvas is not copy constructible
   std::vector<std::shared_ptr<TCanvas>> canvases;
   canvases.emplace_back(new TCanvas("c0", "Dynamic Filling Example", 100, 20, 800, 600));
   canvases.emplace_back(new TCanvas("c1", "Dynamic Filling Example", 510, 20, 800, 600));
   canvases.emplace_back(new TCanvas("c2", "Dynamic Filling Example", 100, 350, 800, 600));
   canvases.emplace_back(new TCanvas("c3", "Dynamic Filling Example", 510, 350, 800, 600));

   // Connect to the Closed() signal to kill the thread when a canvas is closed
   // This is done with a string which links to the Close function passing an argument
   // i.e. closed(Int_t=i) where i is an integer from 0 to 3
   canvases[0]->Connect("Closed()", 0, 0, "JoinThread(unsigned=0)");
   canvases[1]->Connect("Closed()", 0, 0, "JoinThread(unsigned=1)");
   canvases[2]->Connect("Closed()", 0, 0, "JoinThread(unsigned=2)");
   canvases[3]->Connect("Closed()", 0, 0, "JoinThread(unsigned=3)");

   gSystem->ProcessEvents();

   // Fire up the worker threads
   for (auto i : {0, 1, 2, 3})
      threads.emplace_back(doWork, i, &histograms[i], canvases[i].get());

   // This is a special thread, which is dedicated to join the other four
   std::thread joiner(Join);

   while (!isFinished) {
      for (auto i : {0, 1, 2, 3}) {
         if (threads[i].joinable() && canvases[i]->IsModified()) {
            canvases[i]->Update();
         }
      }
      gSystem->Sleep(100); // just slows down the animation
   }

   joiner.join();
}
