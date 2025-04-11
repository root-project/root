void execduplicateEnums(){

   // This avoids double injection in the interpreter of the enum definition
   // which is not swallowed by ROOT when modules are active - and rightfully so
   // as this is how C++ works.
   gInterpreter->ProcessLine("#define __duplicateEnums__");

   gSystem->Load("libduplicateEnums_dictrflx");

   gInterpreter->ProcessLine("namespace edm{}");
   gInterpreter->ProcessLine("TClass::GetClass(\"edm\")->GetListOfEnums(true);");
   gInterpreter->ProcessLine("#undef __duplicateEnums__");

   // We undefine the guard to actually allow the injection in cling of the
   // enum definition
   gInterpreter->ProcessLine("#include \"duplicateEnums.h\"");
   gInterpreter->ProcessLine("TClass::GetClass(\"edm\")->GetListOfEnums(true);");

   std::cout << "Number of enums: " << TClass::GetClass("edm")->GetListOfEnums(true)->GetSize() << std::endl;

}
