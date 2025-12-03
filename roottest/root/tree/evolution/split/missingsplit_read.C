#include "TFile.h"
#include "TTree.h"

class Content {
public:
   Content() : a(0),b(0) {}
   void Set(int i) { a = i; b = 2*i; }
   int a;
   int b;
};

class MyContainer {
public:
   MyContainer() : one() {}
   void Set(int i) { one.Set(i); }
   Content one;
   // Content two;
   void Print() {
      fprintf(stdout,"one.a : %d\n",one.a);
      fprintf(stdout,"one.b : %d\n",one.b);
   }
};

void missingsplit_read(const char *filename = "missingsplit.root") 
{
   TFile *f = new TFile(filename,"READ");
   TTree *tree; f->GetObject("T",tree);
   MyContainer *cont = 0;
   tree->SetBranchAddress("cont.",&cont);
   tree->GetEntry(0);
   cont->Print();
   tree->ResetBranchAddresses();
}
