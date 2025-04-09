#include "TFile.h"
#include "TTree.h"

void createsel(const char *filename = "full-mcfile.root")
{
   TFile *f = TFile::Open(filename,"READ");
   TTree *t; f->GetObject("CollectionTree",t);
   if (t) t->MakeProxy("fullmcSel","fullmc.cxx","","");
}

void usesel(const char *filename = "full-mcfile.root")
{
   TFile *f = TFile::Open(filename,"READ");
   TTree *t; f->GetObject("CollectionTree",t); 
   if (t) t->Process("fullmcSel.h+","goff");
}

int runfullmc(int mode = 1) 
{
   if (mode==2) {
     createsel();
     usesel();
   } else if (mode==4) {
     createsel();
   } else if (mode==5) {
     usesel();
   }
   return 0;
}



