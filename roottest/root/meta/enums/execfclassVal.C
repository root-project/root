void execfclassVal(){
   gInterpreter->ProcessLine("enum en{kNone};");
   gInterpreter->ProcessLine("namespace a{enum en{kNone};}");
   gInterpreter->ProcessLine("class A{public: \nenum en{kNone};};");

   TClass* ptr = (TClass*) 0x9999999999999999;

   for (auto const & enName : {"en","a::en","A::en","pclass","a::pclass","B::pclass"}){
      auto en = TEnum::GetEnum(enName);
      std::cout << enName << ": fClass is " << (ptr == en->GetClass() ? "" : "NOT " ) << " kObjectAllocMemValue" << std::endl;
   }
}
