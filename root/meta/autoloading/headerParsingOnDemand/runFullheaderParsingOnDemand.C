void runFullheaderParsingOnDemand(){
   gDebug=1;

   TClass::GetClass("myClass0<E>")->GetClassInfo();
   TClass::GetClass("myClass1<E>")->GetClassInfo();
   TClass::GetClass("myClass1<int>")->GetClassInfo();
   TClass::GetClass("myClass2<E>")->GetClassInfo();
   TClass::GetClass("myClass2<float>")->GetClassInfo();
   TClass::GetClass("myClass3<E>")->GetClassInfo();
   TClass::GetClass("myClass4<E>")->GetClassInfo();
   TClass::GetClass("myClass5<E>")->GetClassInfo();
   TClass::GetClass("myClass6<E>")->GetClassInfo();
   TClass::GetClass("myClass7<E>")->GetClassInfo();
   TClass::GetClass("myClass8<E>")->GetClassInfo();
   TClass::GetClass("myClass8<std::vector<E>>")->GetClassInfo();
   TClass::GetClass("myClass8<std::vector<std::queue<E>>>")->GetClassInfo();
   TClass::GetClass("myClass9<E>")->GetClassInfo();

   std::cout << "\n";
}
