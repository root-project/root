#include "ROOT/TDataFrame.hxx"
#include "TFile.h"
#include "TTree.h"

void FillTree(const char* filename, const char* treeName) {
   TFile f(filename, "RECREATE");
   TTree t(treeName, treeName);
   int b;
   t.Branch("b", &b);
   for(b = 0; b < 100; ++b)
      t.Fill();
   t.Write();
   f.Close();
}

int main() {
   // TODO check exception is thrown when calling Range after EnableImplicitMT
   auto fileName = "test_ranges.root";
   auto treeName = "rangeTree";
   FillTree(fileName, treeName);

   ROOT::Experimental::TDataFrame d(treeName, fileName, {"b"});

   // all Range signatures. Event-loop is run once
   auto c1 = d.Range(0).Count();            // 100
   auto c2 = d.Range(10).Count();           // 10
   auto m  = d.Range(5, 50).Max();          // 49
   auto t  = d.Range(5, 10, 3).Take<int>(); // {5,8}
   std::cout << *c1 << "\n" << *c2 << "\n" << *m << "\n" << "{";
   for(auto i : t)
      std::cout << i << ",";
   std::cout << "\b}\n";


   // ranges hanging from other nodes
   auto fromARange  = d.Range(10, 50).Range(10, 20).Min();                      // 20
   auto fromAFilter = d.Filter([](int b) { return b > 95; }).Range(10).Count(); // 4

   auto fromAColumn =
      d.Filter([](int) { return true; }).Define("dummy", [](int) { return 42; }).Range(10).Count(); // 10

   std::cout << *fromARange << "\n" << *fromAFilter << "\n" << *fromAColumn << "\n";

   // branching and early stopping TODO how do I check that the event-loop is actually interrupted after 20 iterations?
   unsigned int count = 0;
   auto b1 = d.Range(10).Count();

   auto b2 = d.Define("counter",
                         [&count](int) {
                            ++count;
                            return 42;
                         })
                .Range(20)
                .Take<int>("counter");
   *b1;
   std::cout << count << "\n"; // 20

   // branching, no early stopping
   auto f = d.Filter([](int b) { return b % 2 == 0; });
   auto b3 = f.Range(2).Count();
   auto b4 = f.Count();
   std::cout << *b3 << "\n" << *b4 << "\n";

   return 0;
}
