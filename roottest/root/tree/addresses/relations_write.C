#include "relations.h"
#include "TFile.h"
#include "TTree.h"
#include "TBranch.h"
#include "TROOT.h"

void Write(bool write=false) 
{
   TFile *f    = new TFile("RootRelations.root", "RECREATE", "Root RootRelations test",0);
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

   
   if (write) {
      
      printf("relations' write\n");
      for(int i=0; i < 10; ++i) {

         obj->m_direct.m_entries.push_back(std::pair<int,float>(10*i,i));
         DataTObject *dobj = new ( obj->m_direct.m_tentries[0] ) DataTObject(i*22,i*22/3.0);
         DataTObject *dobj2 = new ( (*(obj->m_direct.m_ptentries))[0] ) DataTObject(i*44,i*44/3.0);
	 if (!dobj || !dobj2) return;

         printf("byte written for entry   #%d: %d\n",i,tree->Fill());

         if (gDebug>0) {
            printf("byte re-read for the same entry: %d %p\n", tree->GetEvent(i), obj); 
         }
	 if (i<0) {
           fprintf(stderr,"the pointer are %p and %p\n",
                   dobj,obj->m_direct.m_tentries.At(0));
         }
         printf("values written for entry #%d: %d, %f, %d, %f, %d, %f\n", i,
                obj->m_direct.m_entries[0].first,
                obj->m_direct.m_entries[0].second,
                ((DataTObject*)obj->m_direct.m_tentries.At(0))->i,
                ((DataTObject*)obj->m_direct.m_tentries.At(0))->j,
                ((DataTObject*)obj->m_direct.m_ptentries->At(0))->i,
                ((DataTObject*)obj->m_direct.m_ptentries->At(0))->j
                ); 

         obj->m_direct.m_entries.clear();
         obj->m_direct.m_tentries.Clear();
      }
      b->Write();
      tree->Write();
      tree->Print();
      f->Write();
   }
   delete f;
};
