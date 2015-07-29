{
   gErrorIgnoreLevel = kWarning;
   const char *path = gSystem->DirName(gInterpreter->GetCurrentMacroName());

   if (strcmp(path,gSystem->pwd()) == 0) {
      //cout << "Local branch\n";
      gInterpreter->AddIncludePath("nest/subdir1");
      gROOT->ProcessLine(".include nest/subdir2");
   } else {
      //cout << "Remote branch\n";
      gInterpreter->AddIncludePath(TString::Format("%s/nest/subdir1",path));
      gROOT->ProcessLine(TString::Format(".include %s/nest/subdir2",path));
   }
   gROOT->ProcessLine(".L script.C++");
}

