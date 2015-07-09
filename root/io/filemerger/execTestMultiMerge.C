#include "TFile.h"

int testMergedFile(const char *filename, Int_t compSetting, Long64_t fileSize)
{
   TFile *file = TFile::Open(filename);
   if (file == nullptr || file->IsZombie()) {
      Error("testSimpleFile", "Could not open %s.",filename);
      return 1;
   }
   file->ls();
   file->cd("hist");
   gDirectory->ls();
   gDirectory->Get("Gaus")->Print();
   file->cd("named");
   gDirectory->ls();
   file->Get("MyList")->Print();

   if (file->GetCompressionSettings() != compSetting) {
      Error("execTestMultiMerge","Compression level of %s should have been %d but is %d\n",file->GetName(), 206, file->GetCompressionSettings() );
      return 3;
   }
   if (file->GetSize() != fileSize) {
      Error("execTestMultiMerge","Disk size of %s should have been %lld but is %lld\n",file->GetName(), fileSize, file->GetSize() );
      return 4;
   }

   delete file;

   return 0;
}

int testSimpleFile(const char *filename, Long64_t entries, Int_t compSetting, Long64_t fileSize, UInt_t tolerance = 0)
{
   fprintf(stdout,"Checking %s\n",filename);
   TFile *file = TFile::Open(filename);
   if (file == nullptr || file->IsZombie()) {
      Error("testSimpleFile", "Could not open %s.",filename);
      return 1;
   }
   //file->ls();
   if (file->GetCompressionSettings() != compSetting) {
      Error("testSimpleFile","Compression level of %s should have been %d but is %d\n",file->GetName(), 206, file->GetCompressionSettings() );
      return 3;
   }
   if (abs(file->GetSize()-fileSize) > tolerance) {
      Error("testSimpleFile","Disk size of %s should have been %lld but is %lld (tolerance %u bytes)\n",file->GetName(), fileSize, file->GetSize(), tolerance);
      return 4;
   }

   TTree *ntuple;
   file->GetObject("ntuple",ntuple);
   if (ntuple == 0) {
      Error("testSimpleFile", "Could not retrieve ntuple from %s.",file->GetName());
      return 2;
   }
   if (ntuple->GetEntries() != entries) {
      Error("testSimpleFile","Number of entries in ntuple in %s should have been %lld but is %lld\n",file->GetName(), entries, ntuple->GetEntries());
      return 4;
   }
   delete file;

   return 0;
}


int execTestMultiMerge()
{
   Int_t result = 0;
   if (!result) result = testMergedFile("mfile1-4.root",1,4922);
   if (!result) result = testMergedFile("mzfile1-4.root",206,4977);

   if (!result) result = testSimpleFile("hsimple.root",25000,1,414411,1);
   if (!result) result = testSimpleFile("hsimple9.root",25000,9,432010);
   if (!result) result = testSimpleFile("hsimple9x2.root",2*25000,9,851108);
   if (!result) result = testSimpleFile("hsimple209.root",25000,209,393974);
   if (!result) result = testSimpleFile("hsimpleK.root",5*25000,209,1917248);
   if (!result) result = testSimpleFile("hsimpleK202.root",5*25000,202,1938626,16);
   if (!result) result = testSimpleFile("hsimpleF.root",5*25000,9,2108423,1);
   return result;
}
