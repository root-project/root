#include "TTree.h"
#include "TFile.h"

TTree *fillTree(int index) 
{  
   TString name = Form("tree%d",index);
   TTree *tree = new TTree(name,name);
   int var[500];// = 33;
   for(int i = 0; i<200+index*100; ++i) {
      TString lname =  Form("%s_i_%d",tree->GetName(),i);
      var[i] = 123+i-index/98;
      tree->Branch(lname,&var[i],lname,512);
   }
   for(int j = 0; j<1000/*index*/; ++j) {
      tree->Fill();
   }
   tree->ResetBranchAddresses();
   return tree;
}

void writefile() {
   TFile *output = new TFile("multi.root","RECREATE");
   fillTree(1);
   fillTree(2);
   fillTree(3);
   output->Write();
   delete output;
}

void readtree(TTree* tree) {
   for(int i = 0; i<tree->GetEntries(); ++i) {
      tree->GetEntry(i);
   }
}

void readfile() {
   TFile input("multi.root");

   TTree *tree1; input.GetObject("tree1",tree1);
   tree1->SetCacheSize(300*1024);

   TTree *tree2; input.GetObject("tree2",tree2);
   tree2->SetCacheSize(400*2048);

   TTree *tree3; input.GetObject("tree3",tree3);
   tree3->SetCacheSize(500*4096);

   readtree(tree1);
   readtree(tree2);
   readtree(tree3);

   tree1->PrintCacheStats();
   tree2->PrintCacheStats();
   tree3->PrintCacheStats();

}

int runmultiTree() {
   // Fill out the code of the actual test


   writefile();
   readfile();
   return 0;
}
