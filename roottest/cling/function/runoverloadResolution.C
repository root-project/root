{
  // Make sure the library is not loaded instead of 
  // the script
  gInterpreter->UnloadLibraryMap("TheClass_h");

  // check interpreted funcs overload resolution
  gROOT->ProcessLine(".L TheClass.h");
  gROOT->ProcessLine(".x testOverloadResolution.C");
#ifndef ClingWorkAroundMissingUnloading
  gROOT->ProcessLine(".U TheClass.h");

  // check compiled funcs overload resolution
  gROOT->ProcessLine(".L TheClass.h+");
  gROOT->ProcessLine(".x testOverloadResolution.C");
#else
  printf("11654121\n");
#endif

#ifdef ClingWorkAroundUnnamedInclude
  gROOT->ProcessLine("#include \"TImage.h\"");
#else
  #include "TImage.h" 
#endif
  "foo" + TString("bar");
  return 0;
}

