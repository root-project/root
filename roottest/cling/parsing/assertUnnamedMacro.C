// Check that anUnnamedMacro.C is parsed as an unnamed macro
// ROOT-8253
int assertUnnamedMacro() {
  gROOT->Macro("anUnnamedMacro.C");
  gROOT->ProcessLine("if (FindThis != 42) exit(1);");
  return 0;
}
