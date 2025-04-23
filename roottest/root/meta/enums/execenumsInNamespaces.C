void checkEnum(){
   TClass* ns = TClass::GetClass("myNamespace");
   if (!ns) {
      std::cerr << "Namespace not found!!!\n";
      return ;
   }

//    ns->GetListOfEnums();

   std::cout << "Enum myNamespace::A " << (TEnum::GetEnum("myNamespace::A", TEnum::kAutoload) ? " " : "not " ) << "found!\n";
   std::cout << "Enum myNamespace::B " << (TEnum::GetEnum("myNamespace::B", TEnum::kAutoload) ? " " : "not " ) << "found!\n";



}

void execenumsInNamespaces(){

   std::cout << "Creating empty Namespace\n";
   gInterpreter->ProcessLine("namespace myNamespace{};");
   checkEnum();

   // This avoids double injection in the interpreter of the enum definition
   // which is not swallowed by ROOT when modules are active - and rightfully so
   // as this is how C++ works.
   gInterpreter->ProcessLine("#define __mynamespace__");

   std::cout << "Loading library\n";
   gSystem->Load("libenumsInNamespaces_dictrflx");
   checkEnum();

   std::cout << "Adding enum in the interpreter\n";
   gInterpreter->ProcessLine("namespace myNamespace{enum B{kOne};};");
   checkEnum();

   // Now check that the enum read from protoclasses does not appear twice after being re-defined in the interpreter
   auto enList = TClass::GetClass("myNamespace2")->GetListOfEnums();
   gInterpreter->ProcessLine("namespace myNamespace2{enum C{kOne};}");
   enList = TClass::GetClass("myNamespace2")->GetListOfEnums();
   int nEnums =enList->GetSize();
   if (nEnums != 1 ){
     std::cerr << "One enum expected, but " << nEnums << " found!\n";
     enList->Dump();
   }
}
