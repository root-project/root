#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstdlib>
#include "TApplication.h"
#include "TFile.h"
#include "TTree.h"

#include "A.h"
#include "B.h"

int write();
int exec();

int main(int argc,char** argv) {

  TApplication theApp("App", &argc, argv);
  gApplication->Init();
  return write();

}
int write() { 
  TFile* tfo = new TFile("test.root","RECREATE");
  TTree* tree = new TTree("tree","test");

  B*   b = new B;
  tree->Branch("B.","B",&b);
  TClonesArray &ar = *(b->fA);

  for(int i=0;i<3;i++) {
    ar.Clear();
    for (int k=0; k<2;k++) {
      new(ar[k]) A();
      //A*   a = new A();
      ((A*)ar[k])->val=1;
      ((A*)ar[k])->tv=TVector3(1,2,3)*(i+1.);

      b->fVecA.push_back( *((A*)ar[k]) );
    }
    tree->Fill();
  }
  tree->Write();
  tfo->Close();

  return 0;
}
