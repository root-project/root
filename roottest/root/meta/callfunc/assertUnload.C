// Check that unloading keeps TMethodCall alive.

int assertUnload() {
   gInterpreter->Declare("int myFunc() { return 42; }");

   TString path = gSystem->DirName(gInterpreter->GetCurrentMacroName());
   gInterpreter->AddIncludePath(path);

   TInterpreter::EErrorCode status;
   gInterpreter->ProcessLine(".x assertUnloadHelper.C",&status);
   if (status != TInterpreter::EErrorCode::kNoError) {
      std::cerr << "Loading of assertUnloadHelper.C failed\n";
      return 1;
   }

   TMethodCall mc(nullptr, "myFunc", "");
   Longptr_t result = 0;

   mc.Execute(result);
   if (result != 42) {
      std::cerr << "myFunc() returned not 42 but " << result << '\n';
      return 1;
   }
   gInterpreter->ProcessLine(".U assertUnloadHelper.C",&status);
   if (status != TInterpreter::EErrorCode::kNoError) {
      std::cerr << "Unloading of assertUnloadHelper.C failed\n";
      return 1;
   }

   mc.Execute(result);
   if (result != 42) {
      std::cerr << "Second myFunc() returned not 42 but " << result << '\n';
      return 1;
   }

   return 0;
}
