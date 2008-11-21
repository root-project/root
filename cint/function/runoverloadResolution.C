{
  // Make sure the library is not loaded instead of 
  // the script
  gInterpreter->UnloadLibraryMap("TheClass_h");

  // check interpreted funcs overload resolution
  gROOT->ProcessLine(".L TheClass.h");
  gROOT->ProcessLine(".x testOverloadResolution.C");
  gROOT->ProcessLine(".U TheClass.h");

  // check compiled funcs overload resolution
  gROOT->ProcessLine(".L TheClass.h+");
  gROOT->ProcessLine(".x testOverloadResolution.C");
  return 0;
}

