void SkipLibrary(const char *libskip)
{
   TEnv* mapfile = gInterpreter->GetMapfile();
   if (!mapfile || !mapfile->GetTable()) return 0;

   for(auto rec : TRangeStaticCast<TEnvRec>(mapfile->GetTable())) {
      if (strstr(rec->GetValue(),libskip)) {
         mapfile->SetValue(rec->GetName(),"");
      }
   }
}


int execStatusBitsCheck(bool verbosity = false)
{
   SkipLibrary("libGviz");

   gSystem->LoadAllLibraries();
   printf("First verify verbose output in the case of TStreamerElement\n");
   ROOT::Detail::TStatusBitsChecker::Check("TStreamerElement",true);
   printf("Second verify all classes for any overlaps\n");
   return ! ROOT::Detail::TStatusBitsChecker::CheckAllClasses(verbosity);
}
