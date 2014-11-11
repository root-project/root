void checkEnum(){
   ns = TClass::GetClass("myNamespace");
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

   std::cout << "Loading library\n";
   gSystem->Load("libenumsInNamespaces_dictrflx");
   checkEnum();

   std::cout << "Adding enum in the interpreter\n";
   gInterpreter->ProcessLine("namespace myNamespace{enum B{kOne};};");
   checkEnum();

}
