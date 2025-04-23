#include <vector>
#include "TFile.h"
#include "TTree.h"
#include "TROOT.h"
#include "TClonesArray.h"

void createsel(const char *filename = "btagobjs.root")
{
   TFile *f = TFile::Open(filename,"READ");
   TTree *t; f->GetObject("btagging",t);
   t->MakeProxy("btagobjsSel","btagobjsScript.cxx","","");
}

void usesel(const char *filename = "btagobjs.root")
{
   TFile *f = TFile::Open(filename,"READ");
   TTree *t; f->GetObject("btagging",t); 
   t->Process("btagobjsSel.h+","goff");
}

int runbtagobjs(int mode = 1) 
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



