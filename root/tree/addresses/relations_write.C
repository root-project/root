#include "relations.h"
#include "TFile.h"
#include "TTree.h"
#include "TROOT.h"

void Write(bool write=false) 
{
   TFile *f    = new TFile("RootRelations.root", "RECREATE", "Root RootRelations test");
   TTree *tree = new TTree("T","An example of a ROOT tree");
   Relation1D<int,float>* obj = new Relation1D<int,float>();

   if (gDebug>0) {
      fprintf(stderr,"Relation1D<int,float> is %p\n",obj);
      fprintf(stderr,"DataObject is %p\n",(DataObject*)obj);
      fprintf(stderr,"Relation<int,float> is %p\n",(Relation<int,float>*)obj);
      fprintf(stderr,"m_direct is %p\n",&(obj->m_direct));
      fprintf(stderr,"m_direct.m_entries is %p\n",&(obj->m_direct.m_entries));
   }

   TBranch *b = tree->Branch("B", "Relation1D<int,float>", &obj);
   //   tree->Print();

   
   if (write) {
      
      printf("relations' write\n");
      for(int i=0; i < 10; ++i) {
         obj->m_direct.m_entries.push_back(std::pair<int,float>(10*i,i));
         printf("%d: %d\n",i,tree->Fill());

         if (gDebug>0) {
            printf("%d %p\n", tree->GetEvent(i), obj); 
         }
         printf("%d: %d, %f\n", i,
                obj->m_direct.m_entries[0].first,
                obj->m_direct.m_entries[0].second); 
         obj->m_direct.m_entries.clear();
      }
      b->Write();
      tree->Write();
      f->Write();
   }
   delete f;
};
