{
   gErrorIgnoreLevel = kWarning;
   gInterpreter->AddIncludePath("nest/subdir1");
   gROOT->ProcessLine(".include nest/subdir2");
   gROOT->ProcessLine(".L script.C++");
}

