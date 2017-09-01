int execStatusBitsCheck(bool verbosity = false)
{
   gSystem->LoadAllLibraries();
   printf("First verify verbose output in the case of TStreamerElement\n");
   ROOT::Detail::TStatusBitsChecker::Check("TStreamerElement",true);
   printf("Second verify all classes for any overlaps\n");
   return ! ROOT::Detail::TStatusBitsChecker::CheckAllClasses(verbosity);
}
