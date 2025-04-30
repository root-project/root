#ifdef R__HAS_DEFAULT_LZ4
constexpr int expectedcomplevel = 404;
#elif defined R__HAS_DEFAULT_ZLIB
constexpr int expectedcomplevel = 101;
#endif

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
      Error("testSimpleFile","Compression level of %s should have been %d but is %d\n",file->GetName(), expectedcomplevel, file->GetCompressionSettings() );
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
