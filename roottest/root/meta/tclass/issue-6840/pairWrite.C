int ForceWriteInfo(TFile *file, const char *classname)
{

   auto cl = TClass::GetClass(classname);
   if (!cl) {
      Error("pairWrite", "Can not get the %s TClass", classname);
      return 1;
   }

   auto info = cl->GetStreamerInfo();
   if (!info) {
      Error("pairWrite", "Can not get the %s StreamerInfo", classname);
      return 2;
   }

   info->ForceWriteInfo(file, true); // Register the StreamerInfo and its dependent.

   return 0;
}


int pairWrite(const char *filename = "pair.root")
{
   if (0 != gSystem->Load("libPairs")) {
      Error("pairWrite", "Can not load libPairs");
      return 1;
   }

   auto file = TFile::Open(filename, "RECREATE");

   auto result = ForceWriteInfo(file, "std::pair<SameAsShort, SameAsShort>");
   if (result)
      return 1+result;

   result = ForceWriteInfo(file, "std::pair<short, SameAsShort>");
   if (result)
      return 3+result;

   file->Write();
   delete file;

   return 0;
}
