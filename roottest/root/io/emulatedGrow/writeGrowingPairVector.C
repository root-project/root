#include "growingPairVector.h"

#include "TFile.h"
#include "TTree.h"

// Writes a tree whose only branch holds a class with a growing
// vector<pair<string,double>>. Compiled with ACLiC, so the dictionary exists
// here; the reader deliberately runs without it.
int writeGrowingPairVector()
{
   TFile f("growingPairVector.root", "RECREATE");
   TTree tree("tree", "growing emulated collections");

   GrowingPairVector obj;
   GrowingPairVector *pobj = &obj;
   // Unsplit, so the collection is read back through the collection proxy.
   tree.Branch("obj", &pobj, 32000, 0);

   for (int entry = 0; entry < kNEntries; ++entry) {
      obj.Fill(entry);
      tree.Fill();
   }

   tree.Write();
   f.Close();
   return 0;
}
