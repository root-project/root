#include "TTree.h"
#include "TFile.h"
#include "TFileCacheRead.h"
#include "TTreeCache.h"

void print(TFileCacheRead *fcache)
{
   TTreeCache *tcache = dynamic_cast<TTreeCache*>(fcache);
   fcache->Print();
   printf("File Reading.......................: %d transactions\n",fcache->GetFile()->GetReadCalls());
}

void print(TTree *tree)
{
   print(tree->GetCurrentFile()->GetCacheRead(tree));
}

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

void readfileOldInterface()
{
   TFile input("multi.root");

   TTree *tree1; input.GetObject("tree1",tree1);
   tree1->SetCacheSize(300*1024);

   TFileCacheRead *cache1 = input.GetCacheRead();
   input.SetCacheRead(0);

   TTree *tree2; input.GetObject("tree2",tree2);
   tree2->SetCacheSize(400*2048);

   TFileCacheRead *cache2 = input.GetCacheRead();
   input.SetCacheRead(0);

   TTree *tree3; input.GetObject("tree3",tree3);
   tree3->SetCacheSize(500*4096);

   TFileCacheRead *cache3 = input.GetCacheRead();
   input.SetCacheRead(0);

   input.SetCacheRead(cache1);
   readtree(tree1);
   input.SetCacheRead(cache2);
   readtree(tree2);
   input.SetCacheRead(cache3);
   readtree(tree3);

   print(cache1);
   print(cache2);
   print(cache3);

   delete cache1;
   delete cache2;
   delete cache3;
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

   print(tree1);
   print(tree2);
   print(tree3);
}

int runmultiTree() {
   writefile();

   // Disable prefilling.
   gEnv->SetValue("TTreeCache.Prefill",0);

   readfileOldInterface();
   readfile();

   // Retest with prefilling on.
   gEnv->SetValue("TTreeCache.Prefill",1);

   readfileOldInterface();
   readfile();

   return 0;
}
