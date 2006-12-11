   //example of script showing how to navigate in a ROOT file
   //with sub-directories and read the objects in each sub-directory.
   //This example uses the file produced by the tutorial importCode.C
   //Author: Rene Brun
      
#include "TFile.h"
#include "TKey.h"
#include "TMacro.h"
   
Int_t nlines = 0;
Int_t nfiles = 0;
Int_t ndirs = 0;
Int_t nh = 0;
Int_t nc = 0;
Int_t nC = 0;
Int_t npy = 0;
void readdir(TDirectory *dir) {
   ndirs++;
   TDirectory *dirsav = gDirectory;
   TIter next(dir->GetListOfKeys());
   TKey *key;
   while ((key = (TKey*)next())) {
      if (key->IsFolder()) {
         dir->cd(key->GetName());
         TDirectory *subdir = gDirectory;
         readdir(subdir);
         dirsav->cd();
         continue;
      }
      TMacro *macro = (TMacro*)key->ReadObj();
      nfiles++;
      nlines += macro->GetListOfLines()->GetEntries();
      if (strstr(key->GetName(),".h"))   nh++;
      if (strstr(key->GetName(),".c"))   nc++;
      if (strstr(key->GetName(),".C"))   nC++;
      if (strstr(key->GetName(),".py"))  npy++;
      delete macro;
   }
}
         
      
void readCode() {
   TFile *f = new TFile("code.root");
   if (f->IsZombie()) {
      printf("File code.root does not exist. Run tutorial importCode.C first\n");
      return;
   }
   printf("Reading file ==> code.root\n");
   printf("File size in bytes       = %lld\n",f->GetEND());
   printf("File compression factor  = %g\n",f->GetCompressionFactor());
   
   readdir(f);
   
   printf("Number of sub-dirs       = %d\n",ndirs);
   printf("Number of macro files    = %d\n",nfiles);
   printf("Number of lines in mac   = %d\n",nlines);
   printf("Number of cxx,c,cc files = %d\n",nc);
   printf("Number of C files        = %d\n",nC);
   printf("Number of Python files   = %d\n",npy);
}
