#include "TFile.h"
#include "TChain.h"
#include "TVirtualIndex.h"

struct MiniEvent {
   int run;
   int event;
   float value;
   
   void Set(int r, int e) {
      run = r;
      event = e;
      value = 1000*r + e;
   }
   
};

void writeFile(const char *prefix, int index, int size)
{
   TString filename = Form("%s-%d.root",prefix,index);
   TFile f(filename,"RECREATE");
   TTree *tree = new TTree("tree","with index");
   MiniEvent *e = new MiniEvent;
   tree->Branch("event.",&e);

   for(int i = 0; i<size; ++i) {
      if (index%2) {
         e->Set(4-index,i);
      } else {
         e->Set(4-index,size-i);
      }
      tree->Fill();
   }
   tree->BuildIndex("run","event");
   f.Write();
}

void readFile(const char *prefix, int index = -1) 
{
   TString filename = Form("%s-%d.root",prefix,index);
   if (index == -1) {
      filename = Form("%s_merge.root",prefix);
   }
   TFile f(filename,"READ");
   TTree *tree; f.GetObject("tree",tree);
   if (tree) {
      tree->SetScanField(0);
      tree->Scan("*");
      tree->GetTreeIndex()->Print("all");
   }
}

   
void mergeFiles(const char *prefix, Option_t *opt = "") 
{
   TChain c("tree");
   c.Add(Form("%s-*",prefix));
   
   TString filename = Form("%s%s_merge.root",prefix,opt);

   c.Merge(filename,opt);

}

void runindex(int size = 10) {
   for(int i=0; i<4; i++) {
      writeFile("index",i,size);
      readFile("index",i);
   }
   mergeFiles("index");
   readFile("index");

   mergeFiles("index","_fast");
   readFile("index_fast");
}
