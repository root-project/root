int testSimpleFile(const char *filename, long entries, int compSetting, long fileSize, unsigned tolerance = 0)
{
   fprintf(stdout,"Checking %s\n",filename);
   auto file = TFile::Open(filename);
   if (!file || file->IsZombie()) {
      Error("testSimpleFile", "Could not open %s.",filename);
      return 1;
   }

   if (file->GetCompressionSettings() != compSetting) {
      Error("testSimpleFile","Compression level of %s should have been %d but is %d\n",file->GetName(), compSetting, (int) file->GetCompressionSettings() );
      return 100;
   }

   if (abs(file->GetSize() - fileSize) > tolerance) {
      Error("testSimpleFile","Disk size of %s should have been %ld but is %ld (tolerance %u bytes)\n",file->GetName(), fileSize, (long) file->GetSize(), tolerance);
      return 1000;
   }

   TTree *ntuple = nullptr;
   file->GetObject("ntuple",ntuple);
   if (!ntuple) {
      Error("testSimpleFile", "Could not retrieve ntuple from %s.",file->GetName());
      return 10;
   }

   if (ntuple->GetEntries() != entries) {
      Error("testSimpleFile","Number of entries in ntuple in %s should have been %ld but is %ld\n",file->GetName(), entries, (long) ntuple->GetEntries());
      return 10000;
   }

   delete file;

   return 0;
}
