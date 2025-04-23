void execAttributesCheck() {

auto c = TClass::GetClass("TestDictProperties");
for (auto dm: *c->GetListOfDataMembers()){
   auto dmAM = static_cast<TDataMember*>(dm)->GetAttributeMap();
   for (auto const key : {"mapping","persistency"}){
      if (dmAM->HasKey(key)) std::cout << "Found property \"" << key << "\" and its value is " << dmAM->GetPropertyAsString(key)  << "\n"; 
   }
}

}
