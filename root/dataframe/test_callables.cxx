#include "ROOT/RDataFrame.hxx"
#include "TFile.h"
#include "TTree.h"
#include "TError.h"
#include <atomic>

// Test support to different callable types:
// free functions, functor classes, lambdas and std::functions are used
// as filters, temporary branch expressions and foreach expressions.
// Usage of these callables is sprinkled with std::moves.

void FillTree(const char* filename, const char* treeName) {
   TFile f(filename, "RECREATE");
   TTree t(treeName, treeName);
   double b;
   t.Branch("b", &b);
   for(int i = 0; i < 100000; ++i) {
      b = i / 100000.;
      t.Fill();
   }
   t.Write();
   f.Close();
}

bool freeFilter(double) {
   return true;
}

int freeBranch() {
   return 42;
}

std::atomic_int freeCounter(0);
void freeForeach(double) {
   ++freeCounter;
}

class FunctorFilter {
public:
   bool operator()(double) {
      return true;
   }
};

class FunctorBranch {
public:
   int operator()() {
      return 42;
   }
};

class FunctorForeach {
   static std::atomic_int fCounter;
public:
   void operator()(double) {
      ++fCounter;
   }
   int GetCounter() const {
      return fCounter;
   }
};
std::atomic_int FunctorForeach::fCounter(0);

int main() {
   auto fileName = "test_callables.root";
   auto treeName = "callablesTree";
   FillTree(fileName, treeName);
   
   TFile f("test_callables.root");
   ROOT::RDataFrame d(treeName, fileName, {"b"});

   // free function
   d.Filter(freeFilter)
    .Define("a", freeBranch, {})
    .Foreach(freeForeach); // this in turn calls ForeachSlot

   // functor
   FunctorFilter ff;
   FunctorForeach fff;
   d.Filter(ff)
    .Define("c", FunctorBranch())
    .ForeachSlot(std::move(fff));

   // lambda
   auto filt = [](double) { return true; };
   auto branch = []() { return 42; };
   std::atomic_int counter(0);
   auto foreach = [&counter](double) { ++counter; };
   d.Filter(filt)
    .Define("d", std::move(branch), {})
    .Foreach(foreach);

   // std::function
   std::function<bool(double)> stdfilt = filt;
   std::function<int(void)> stdbranch = branch;
   std::function<void(double)> stdforeach = foreach;
   d.Filter(std::move(stdfilt))
    .Define("e", std::move(stdbranch), {})
    .Foreach(foreach);

   if (freeCounter == fff.GetCounter() && freeCounter*2 == counter
       && freeCounter == static_cast<TTree*>(f.Get(treeName))->GetEntries())
      Info("test_callables", "alright");
   else
      Error("test_callables",
            "not alright: %i %i %i",
            static_cast<int>(freeCounter),
            static_cast<int>(fff.GetCounter()),
            static_cast<int>(counter));

   return 0;
}
