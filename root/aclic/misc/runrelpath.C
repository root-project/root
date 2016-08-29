{
   gErrorIgnoreLevel = kWarning;
   TString path = gSystem->DirName(gInterpreter->GetCurrentMacroName());
   if (path.EndsWith("/.")) {
      path.Remove(path.Length()-2);
   }

   if (strcmp(path,gSystem->pwd()) == 0) {
      // cout << "Local branch\n";
      gInterpreter->AddIncludePath("nest/subdir1");
      gROOT->ProcessLine(".include nest/subdir2");
   } else {
      // cout << "Remote branch\n";
      gInterpreter->AddIncludePath(TString::Format("%s/nest/subdir1",path.Data()));
      gROOT->ProcessLine(TString::Format(".include %s/nest/subdir2",path.Data()));
   }
   if (gSystem->AccessPathName("temp/subdir3", kFileExists)
       && gSystem->mkdir("temp/subdir3", kTRUE) ) {
      Error("","Could not create directory temp/subdir3");
      exit(1);
   }
   TMacro header("");
   header.AddLine("#ifndef NESTED_TEMP_HEADER_H\n"
                 "#define NESTED_TEMP_HEADER_H\n"
                 "int defined_in_temp_subdir3() { return 0; }\n"
                 "#endif\n");
   header.SaveSource("temp/subdir3/tempHeader.h");

   gSystem->AddIncludePath("-Itemp/subdir3");
   gROOT->ProcessLine(".L script.C+");
}

