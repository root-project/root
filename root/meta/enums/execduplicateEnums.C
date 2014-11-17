void execduplicateEnums(){

   gSystem->Load("libduplicateEnums_dictrflx");

   gInterpreter->ProcessLine("namespace edm{}");
   gInterpreter->ProcessLine("TClass::GetClass(\"edm\")->GetListOfEnums(true);");
   gInterpreter->ProcessLine("#include \"duplicateEnums.h\"");
   gInterpreter->ProcessLine("TClass::GetClass(\"edm\")->GetListOfEnums(true);");

   std::cout << "Number of enums: " << TClass::GetClass("edm")->GetListOfEnums(true)->GetSize() << std::endl;

}
