// Check that unloading keeps TMethodCall alive.

int assertUnload() {
   gInterpreter->Declare("int myFunc() { return 42; }");
   gInterpreter->ProcessLine(".x assertUnloadHelper.C");
   TMethodCall mc(nullptr, "myFunc", "");
   long result = 0;

   mc.Execute(result);
   if (result != 42) {
      std::cerr << "myFunc() returned not 42 but " << result << '\n';
      return 1;
   }
   gInterpreter->ProcessLine(".U assertUnloadHelper.C");

   mc.Execute(result);
   if (result != 42) {
      std::cerr << "Second myFunc() returned not 42 but " << result << '\n';
      return 1;
   }

   return 0;
}
