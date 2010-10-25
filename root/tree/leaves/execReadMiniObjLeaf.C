#include "TFile.h"
#include "TTree.h"
#include "TBranch.h"
#include "TLeaf.h"
#include "TLeafElement.h"
#include "TBranchElement.h"
#include <vector>

#include <iostream>

void execReadMiniObjLeaf()
{
  TFile f("miniObjTree.root");
  TTree* t=(TTree*)f.Get("miniObjTree");
   
   
  TLeaf* l_v=t->FindLeaf("v");
  TLeaf* l_x=t->FindLeaf("x");
  if(!l_v || ! l_x){
    cout << "No leaf" << endl;
    exit(1);
  }
   l_v->GetBranch()->GetEntry(0); // Make sure the address are set
   std::vector<double> *v = 0;
   v = (std::vector<double> *)l_v->GetBranch()->GetAddress();

  for(int i=0; i<2 /* t->GetEntries() */; ++i){
//    cout << l_v->GetBranch()->GetEntry(i) << " bytes read from l_v" << endl;
//    cout << l_x->GetBranch()->GetEntry(i) << " bytes read from l_x" << endl;
    for(int j=0; j<10; ++j) cout << "l_v value " << j << "=" << l_v->GetValue(j) << endl;
   // for(int j=0; j<10; ++j) cout << "l_v value " << j << "=" << v->at(j) << endl;
    cout << "l_x value=" << l_x->GetValue() << endl;
  }
}
