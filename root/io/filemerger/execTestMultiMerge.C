#include "TFile.h"

extern "C" uint32_t lzma_version_number(void);

constexpr bool kIs32bits = sizeof(long) == 4;

int testMergedFile(const char *filename, Int_t compSetting, Long64_t fileSize, UInt_t tolerance = 0)
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

   if (abs(file->GetSize()-fileSize) > tolerance) {
      Error("execTestMultiMerge","Disk size of %s should have been %lld but is %lld (tolerance %u bytes)\n",file->GetName(), fileSize, file->GetSize(), tolerance );
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
      return 100;
   }
   if (abs(file->GetSize()-fileSize) > tolerance) {
      Error("testSimpleFile","Disk size of %s should have been %lld but is %lld (tolerance %u bytes)\n",file->GetName(), fileSize, file->GetSize(), tolerance);
      return 1000;
   }

   TTree *ntuple;
   file->GetObject("ntuple",ntuple);
   if (ntuple == 0) {
      Error("testSimpleFile", "Could not retrieve ntuple from %s.",file->GetName());
      return 10;
   }
   if (ntuple->GetEntries() != entries) {
      Error("testSimpleFile","Number of entries in ntuple in %s should have been %lld but is %lld\n",file->GetName(), entries, ntuple->GetEntries());
      return 10000;
   }
   delete file;

   return 0;
}

int execTestMultiMerge()
{
   Int_t result = 0;
   int hsimpleFTolerance = 6; // 5 for non fst builds
   result += testMergedFile("mzfile1-4.root",206,4992, kIs32bits ? 2 : 0);
   result += testMergedFile("mlz4file1-4.root",406,5013, kIs32bits ? 2 : 0);
   result += testMergedFile("mzlibfile1-4.root",106,4917, kIs32bits ? 2 : 0);
   result += testSimpleFile("hsimple.root",25000,1,414415, kIs32bits ? 10 : 8);
   result += testSimpleFile("hsimple9.root",25000,9,432029,3);
   result += testSimpleFile("hsimple109.root",25000,109,432038,3);
   result += testSimpleFile("hsimple9x2.root",2*25000,9,851123,9);
   result += testSimpleFile("hsimple109x2.root",2*25000,109,851134,9);
   result += testSimpleFile("hsimple209.root",25000,209,394077,8);
   result += testSimpleFile("hsimple409.root",25000,409,516289,8);
   result += testSimpleFile("hsimpleK.root",6*25000,209,2298976,16);
   if (lzma_version_number() < 50020010) {
      // lzma v5.2.0 produced larger files ...
      // but even older version (eg v5.0.0) produced smaller files ...
      result += testSimpleFile("hsimpleK202.root",6*25000,202,1938720,700);
   } else {
      result += testSimpleFile("hsimpleK202.root",12*25000,202,4631252,16);
   }
   result += testSimpleFile("hsimpleK409.root",24*25000,409,12046474,16);
   result += testSimpleFile("hsimpleF.root",6*25000,9,2527731,hsimpleFTolerance);
   return result;
}
