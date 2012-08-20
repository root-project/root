#include "TFile.h"
#include "TTree.h"

void createsel(const char *filename = "vec.root")
{
   TFile *f = TFile::Open(filename,"READ");
   TTree *t; f->GetObject("t",t);
   t->MakeProxy("vectorintEmuSel","dude.cxx","","");
}

void usesel(const char *filename = "vec.root")
{
   TFile *f = TFile::Open(filename,"READ");
   TTree *t; f->GetObject("t",t); 
   t->Process("vectorintEmuSel.h+","goff");
}

int runvectorintEmu(int mode = 1) 
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



