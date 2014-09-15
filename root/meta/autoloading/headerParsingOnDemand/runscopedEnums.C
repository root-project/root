void runscopedEnums()
{
   auto className = "cond::IOVSequence";
   auto enumName = "cond::IOVSequence::ScopeType";

   if(!TEnum::GetEnum(enumName,TEnum::kNone))
      std::cout << "OK: enumerator " << enumName << "not found\n";

   gDebug=1;
   if(TClass::GetClass(enumName))
      std::cout << "Error: no TClass should be found!\n";

   if(!gROOT->GetListOfClasses()->FindObject(className))
      std::cout << "OK: class " << className << " should not be in the list of classes\n";

   if(gClassTable->GetProto(className))
      std::cout << "OK: proto class " << className << " should be in the list of proto classes as a side effect of loading\n";

   if(TEnum::GetEnum(enumName,TEnum::kNone))
      std::cout << "OK: enumerator " << enumName << "found\n";
}
