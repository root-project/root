// Minimal reproducer: TChain::Add(TChain*) does not update the cached entry
// count.  After merging chains this way, GetEntries() / GetEntriesFast() return
// a wrong (too small) value, even though every file and every entry is present
// in the merged chain (a LoadTree walk and CopyTree both see the full sample).
//
// Run:   root -l -b -q chain_add_getentries.C
//
// Observed with ROOT 6.40.00.

#include "TChain.h"
#include "TFile.h"
#include "TTree.h"
#include "TRandom3.h"
#include "TSystem.h"
#include "TROOT.h"

namespace {

   // Create nfiles files under dir, each with a TTree "tree" holding a random
   // number of entries.  Returns the total number of entries written.
   Long64_t make_sample(const std::string& dir, int nfiles, TRandom3& rng)
   {
      gSystem->mkdir(dir.c_str(), kTRUE);
      Long64_t total = 0;
      for (int f = 0; f < nfiles; ++f) {
         const std::string fname = dir + "/f_" + std::to_string(f) + ".root";
         TFile out(fname.c_str(), "RECREATE");
         TTree t("tree", "tree");
         Double_t x;
         t.Branch("x", &x);
         const Long64_t n = 50 + rng.Integer(200);   // 50..249 entries per file
         for (Long64_t i = 0; i < n; ++i) { x = rng.Gaus(); t.Fill(); }
         t.Write();
         out.Close();
         total += n;
      }
      return total;
   }

   TChain* make_chain(const std::string& dir, int nfiles)
   {
      TChain* c = new TChain("tree");
      for (int f = 0; f < nfiles; ++f)
         c->Add((dir + "/f_" + std::to_string(f) + ".root").c_str());
      return c;
   }

} // namespace

int chain_add_getentries()
{
   printf("ROOT version: %s\n\n", gROOT->GetVersion());

   TRandom3 rng(42);
   const std::string base = "chain_add_repro_files";
   gSystem->Exec(("rm -rf " + base).c_str());

   // Three independent samples with different file counts.
   const int nA = 4, nB = 7, nC = 5;
   const Long64_t tA = make_sample(base + "/A", nA, rng);
   const Long64_t tB = make_sample(base + "/B", nB, rng);
   const Long64_t tC = make_sample(base + "/C", nC, rng);
   const Long64_t truth = tA + tB + tC;
   printf("entries written:   A=%lld  B=%lld  C=%lld   →  true total = %lld\n",
          tA, tB, tC, truth);

   // One chain per sample; query each count (e.g. to print per-sample yields).
   TChain* a = make_chain(base + "/A", nA);
   TChain* b = make_chain(base + "/B", nB);
   TChain* c = make_chain(base + "/C", nC);
   printf("per-chain GetEntries(): a=%lld  b=%lld  c=%lld   (sum=%lld)\n",
          a->GetEntries(), b->GetEntries(), c->GetEntries(),
          a->GetEntries() + b->GetEntries() + c->GetEntries());

   // Merge by adding TChains into a TChain.
   a->Add(b);
   a->Add(c);

   bool result = (a->GetEntries() == truth);

   printf("\nafter  a->Add(b); a->Add(c):\n");
   printf("  GetEntries()     = %lld   ← %s (expected %lld)\n",
          a->GetEntries(), result ? "Correct" : "WRONG", truth);
   printf("  GetEntriesFast() = %lld   ← %s\n", a->GetEntriesFast(), (a->GetEntriesFast() == truth) ? "Correct" : "WRONG");
   printf("  files in chain   = %d   (all present)\n",
          a->GetListOfFiles()->GetEntries());

   // The entries really are all there: an explicit LoadTree walk reaches them.
   Long64_t nav = 0;
   while (a->LoadTree(nav) >= 0) ++nav;
   printf("  reachable via LoadTree loop = %lld   ← correct total\n", nav);

   // Workaround: add the files into a single chain instead of chaining chains.
   TChain* good = new TChain("tree");
   for (int f = 0; f < nA; ++f) good->Add((base + "/A/f_" + std::to_string(f) + ".root").c_str());
   for (int f = 0; f < nB; ++f) good->Add((base + "/B/f_" + std::to_string(f) + ".root").c_str());
   for (int f = 0; f < nC; ++f) good->Add((base + "/C/f_" + std::to_string(f) + ".root").c_str());
   printf("\nsingle chain (Add per file): GetEntries() = %lld   ← %s\n",
          good->GetEntries(), (good->GetEntries() == truth) ? "Correct" : "WRONG");

   return result ? 0 : 1;
}

