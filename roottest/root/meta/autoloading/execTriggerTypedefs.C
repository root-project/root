// This test checks that enums trigger autoload in case the root typesystem is queried.
// It is of course not possible to have this as a tcling callback as it is impossible
// to trigger an incomplete type error with enums: they cannot be forward declared.

void execTriggerTypedefs(){

   auto en = TEnum::GetEnum("trigger::TriggerObjectType");
   auto constants = en->GetConstants();

   if (!en){
      std::cerr << "No enumerator found!\n";
      return ;
   }

   std::cout << "Enum: " << en->GetName() << std::endl;

   for (auto constantAsObj : *constants){
      auto constant = (TEnumConstant*) constantAsObj;
      std::cout << " - Constant " << constant->GetName() << " has value " << constant->GetValue() << std::endl;
   }

}
