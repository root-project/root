#include "def.h"

#include "TFile.h"
#include "TTree.h"


#include <iostream>
namespace std {} using namespace std;

int def() {

   MyClass *out = new MyClass(99);
   TClonesArray *cla = new TClonesArray("Wrapper");
   new ( (*cla)[0] ) Wrapper(1);
   new ( (*cla)[1] ) Wrapper(2);

   TFile *file = new TFile("test.root","RECREATE");
   cla->Write("array");
   
   TTree *tree = new TTree("T","T");
   tree->Branch("obj","MyClass",&out);
   tree->Branch("arr","TClonesArray",&cla);
   tree->Fill();
   tree->Fill();
   file->Write();
   file->Close();

   Wrapper *w = (Wrapper*)cla->At(0);
   cout << "myvar is : " << w->chunk->myvar << "\n";

   return 0;

};
