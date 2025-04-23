void execQualifiedNames(){

   gInterpreter->ProcessLine("enum en{kNull};");
   gInterpreter->ProcessLine("namespace a{ namespace b{ enum en{kNull};}}");
   for (auto const & enName : {"en","a::b::en","myns::enpclass","enpclass"}){
      auto ien = TEnum::GetEnum(enName);
      if (!ien){
         std::cout << "Error! Could not find " << enName << std::endl;
         continue;
      }
      std::cout << "Name/Title: "  << ien->GetName() << "/" << ien->GetTitle() << std::endl;
      std::cout << "QualName: " << ien->GetQualifiedName() << std::endl;
   }

}
