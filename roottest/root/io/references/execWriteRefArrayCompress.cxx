#include "RefArrayCompress.hh"
#include <iostream>
#include "TTree.h"
#include "TFile.h"
using namespace std;

int execWriteRefArrayCompress() 
{
  Top* atop = new Top();
  TFile afile("refArrayComp.root", "recreate");
  TTree tree("tree", "tree");
 
  tree.BranchRef();
  tree.Branch("top", atop);

  for (size_t i=0;i<10;i++) {
    ObjA* a = static_cast<ObjA*>(atop->fObjAArray->New(i));
    a->fObjAVal = i*100; 
    atop->fObjAs.AddLast(a);
  }
  tree.Fill();

  atop->Clear();
  for (size_t i=0;i<5;i++) {
    ObjA* a = static_cast<ObjA*>(atop->fObjAArray->New(i));
    a->fObjAVal = i*100; 
    atop->fObjAs.AddLast(a);
  }
  tree.Fill();

  tree.Write();
  afile.Close();
  
   
  return 0;
}
