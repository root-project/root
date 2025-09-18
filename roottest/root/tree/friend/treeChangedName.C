#include "TFile.h"
#include "TTree.h"

// https://its.cern.ch/jira/browse/ROOT-7743
void treeChangedName()
{
   float x, y;
  
   TFile f1("ftest7743.root", "RECREATE");
   TTree tree1("tree1", "a test tree");
   tree1.Branch("x", &x, "x/F");
   TTree tree2("tree2", "another test tree");
   tree2.Branch("y", &y, "y/F");

   for(int i = 0; i < 1000; i++) {
      x = i * 0.001;
      y = 1 - x;

      tree1.Fill();
      tree2.Fill();
   }
   tree1.AddFriend(&tree2);

   tree1.Write("tree1");
   tree2.Write("treeW");

   f1.Close();
}
