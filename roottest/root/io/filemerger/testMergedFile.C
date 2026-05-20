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
      Error("execTestMultiMerge","Compression level of %s should have been %d but is %d\n",file->GetName(), compSetting, file->GetCompressionSettings());
      return 3;
   }

   if (abs(file->GetSize()-fileSize) > tolerance) {
      Error("execTestMultiMerge","Disk size of %s should have been %lld but is %lld (tolerance %u bytes)\n",file->GetName(), fileSize, file->GetSize(), tolerance);
      return 4;
   }

   delete file;

   return 0;
}
