// Reads back the file written by writeGrowingPairVector.C without loading its
// dictionary, so the vector<pair<string,double>> goes through
// TEmulatedCollectionProxy.
//
// growingPairVector.h is deliberately NOT included: if cling knew the class,
// the collection would get a real proxy and the emulated path -- the one this
// test is about -- would never run. The expectations below mirror the header
// on purpose; keep the two in sync.

#include "TError.h"
#include "TFile.h"
#include "TTree.h"

int readGrowingPairVector()
{
   const int nEntries = 12;

   auto file = TFile::Open("growingPairVector.root");
   if (!file || file->IsZombie()) {
      Error("readGrowingPairVector", "could not open growingPairVector.root");
      return 1;
   }

   TTree *tree = nullptr;
   file->GetObject("tree", tree);
   if (!tree) {
      Error("readGrowingPairVector", "could not find the tree");
      return 1;
   }

   if (tree->GetEntries() != nEntries) {
      Error("readGrowingPairVector", "expected %d entries, found %lld", nEntries, tree->GetEntries());
      return 1;
   }

   // The collection grows on every entry, so the emulated buffer has to
   // reallocate repeatedly. Before the fix for #20882 this aborted with an
   // invalid free: the relocated std::string kept pointing into the old buffer.
   for (Long64_t entry = 0; entry < tree->GetEntries(); ++entry)
      tree->GetEntry(entry);

   // Check the values too, so that silent corruption fails the test rather than
   // only an outright crash.
   Long64_t expectedCount = 0;
   double expectedSum = 0.;
   for (int entry = 0; entry < nEntries; ++entry) {
      for (int i = 0; i < 8 * (entry + 1); ++i) {
         ++expectedCount;
         expectedSum += entry * 1000.0 + i;
      }
   }

   const Long64_t count = tree->Draw("obj.fData.second", "", "goff");
   if (count != expectedCount) {
      Error("readGrowingPairVector", "expected %lld elements, read %lld", expectedCount, count);
      return 1;
   }

   double sum = 0.;
   const double *values = tree->GetV1();
   for (Long64_t i = 0; i < count; ++i)
      sum += values[i];

   if (sum != expectedSum) {
      Error("readGrowingPairVector", "expected the values to sum to %f, got %f", expectedSum, sum);
      return 1;
   }

   return 0;
}
