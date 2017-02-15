#include<Mpi.h>
#include <cassert>
using namespace ROOT::Mpi;


void timer()
{
   TEnvironment env;          //environment to start communication system
   TMpiTimer timer(COMM_WORLD);

   timer.Start();

   assert(timer.IsGlobal() == kFALSE);
   timer.Sleep(1000 * COMM_WORLD.GetRank() + 1000);
   timer.Print();
   auto hist = timer.GetElapsedHist(0);

   if (COMM_WORLD.GetRank() == 0) {
      TCanvas *c = new TCanvas("times");
      hist->Draw();
      c->SaveAs("times.png");
   }
}

