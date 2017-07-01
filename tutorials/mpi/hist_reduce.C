// Omar.Zapata@cern.ch http://oproject.org/ROOTMpi  2017
// Example to generated random numbers to fill a TH1F histogram in every process
// and merging the result through a custom reduce operation
// run it with: rootmpi -np 3 hist_reduce.C   where 3 is the number of processes
#include <Mpi.h>
#include <TH1F.h>

using namespace ROOT::Mpi;

TOp<TH1F> HSUM() // histogram sum(custom operation for reduce)
{
   // returning an  ROOT::Mpi::Op<TH1F>(arg) object where "arg" is a lambda
   // function with histograms sum
   return TOp<TH1F>([](const TH1F &a, const TH1F &b) {
      TH1F c(a);
      c.Add(&b);
      return c;
   });
}

void hist_reduce(Int_t points = 100000)
{
   TEnvironment env;

   auto root = 0;
   auto rank = COMM_WORLD.GetRank();

   if (COMM_WORLD.GetSize() == 1) return; // need at least 2 process

   auto form1 = new TFormula("form1", "abs(sin(x)/x)");
   auto sqroot = new TF1("sqroot", "x*gaus(0) + [3]*form1", 0, 10);
   sqroot->SetParameters(10, 4, 1, 20);

   TH1F h1f("h1f", "Test random numbers", 200, 0, 10);
   h1f.SetFillColor(rank);
   h1f.FillRandom("sqroot", points);

   TH1F result;

   COMM_WORLD.Reduce(h1f, result, HSUM, root);

   if (rank == root) {
      TCanvas *c1 = new TCanvas("c1", "The FillRandom example", 200, 10, 700, 900);
      c1->SetFillColor(18);
      result.Draw();
      c1->SaveAs("hist.png");
      delete c1;
   }

   delete form1;
   delete sqroot;
}
