#include "TFile.h"

int execTestMultiMerge()
{
   TFile *file = TFile::Open("mfile1-4.root");
   file->ls();
   file->cd("hist");
   gDirectory->ls();
   gDirectory->Get("Gaus")->Print();
   file->cd("named");
   gDirectory->ls();
   file->Get("MyList")->Print();

   delete file;

   file = TFile::Open("mzfile1-4.root");
   file->ls();
   file->cd("hist");
   gDirectory->ls();
   gDirectory->Get("Gaus")->Print();
   file->cd("named");
   gDirectory->ls();
   file->Get("MyList")->Print();
   if (file->GetCompressionSettings() != 206) {
      Error("execTestMultiMerge","Compression level of %s should have been %d but is %d\n",file->GetName(), 206, file->GetCompressionSettings() );
      return 1;
   }


   return 0;
}
