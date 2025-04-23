#include "TTree.h"

void output(TTree *tree, char *name) 
{  
   tree->Scan("name");
   for(int i=0; i<tree->GetEntries(); ++i) {
      tree->GetEntry(i);
      fprintf(stdout,"the %d name is %s\n",i,name);
   }
}

int runleaflist(int save=false) 
{
   TTree *tree = new TTree("T","T");
   char name[20];
   strcpy(name,"test1");
   tree->Branch("name",name,"name/C");
   tree->Fill();
   strcpy(name,"");
   tree->Fill();
   strcpy(name,"test2");
   tree->Fill();
   strcpy(name,"");
   tree->Fill();
   output(tree,name);

   if (save) {
      TFile *file = new TFile("badleafc.root","RECREATE");
      tree->Write();
      file->Write();
      delete file;
   } else {
      TFile *file = new TFile("badleafc.root","READ");
      TTree *old; file->GetObject("T",old);
      old->SetBranchAddress("name",name);
      output(old,name);
      delete file;
   }
   return 0;
}
