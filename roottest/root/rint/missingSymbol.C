int missingSymbol() {
   gInterpreter->ProcessLine("extern int MyMissingSymbol();");
   gInterpreter->ProcessLine("MyMissingSymbol();");
   return 0;
}
