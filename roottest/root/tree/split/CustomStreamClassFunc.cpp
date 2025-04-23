#include "CustomStreamClass.h"
#include "TTree.h"
#include "TFile.h"
#include "TBranchElement.h"

constexpr const char *filename = "customClassTree.root";

void WriteTree()
{
   TFile f(filename,"recreate");
   
   TTree* t = new TTree("tree","tree");
   
   MyClass* m = new MyClass;
   
   t->Branch("myclass", "MyClass",(void*)&m, 32000, 0);
   
   for(int i=0;i<5;++i) {
      m->SetX(i);
      t->Fill();
   }
   
   m->Write("MyObject");
   f.Write();
}

int ReadTree()
{
   TFile f(filename);

   if (f.IsZombie()) return -1;

   TTree* t = (TTree*)f.Get("tree");
   // t->SetBranchStyle(0);

   if (!t) return -2;

   MyClass* m = (MyClass*)f.Get("MyObject");

   if (!m) return -3;

   t->SetBranchAddress("myclass", &m);
   
   for(int i=0;i<5;++i) {
      t->GetEntry(i);
      m->PrintX();
   }
   return 0;
}

