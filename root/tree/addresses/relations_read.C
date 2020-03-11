#include "relations.h"
#include "TFile.h"
#include "TTree.h"
#include "TBranch.h"
#include <iostream>
using namespace std;

void Read(bool /*read*/=true) {

   TFile *f    = new TFile("RootRelations.root");
   TTree *tree = (TTree*)f->Get("T");
 
   tree->Scan("m_direct.m_entries.first:m_direct.m_tentries.i:m_direct.m_ptentries.i");

   Relation1D<int,float>* obj = new Relation1D<int,float>();

   TBranch *branch  = tree->GetBranch("B");

   branch->SetAddress(&obj);

   Int_t nevent = Int_t(tree->GetEntries());
   if (gDebug>0) std::cout << "Address:" << &obj->m_direct.m_entries << std::endl;
   printf("relations' read\n");
   for (Int_t i=0;i<nevent;i++) {
      Int_t val = tree->GetEntry(i);
      if (gDebug>0) printf("%d %p\n", val, obj); 
      printf("values read for entry #%d: %d, %f, %d, %f, %d, %f\n", i,
                obj->m_direct.m_entries[0].first,
                obj->m_direct.m_entries[0].second,
                ((DataTObject*)obj->m_direct.m_tentries.At(0))->i,
                ((DataTObject*)obj->m_direct.m_tentries.At(0))->j,
                ((DataTObject*)obj->m_direct.m_ptentries->At(0))->i,
                ((DataTObject*)obj->m_direct.m_ptentries->At(0))->j
                ); 
   }
   tree->Scan("m_direct.m_entries.first:m_direct.m_tentries.i:m_direct.m_ptentries.i");
   delete f;

}
