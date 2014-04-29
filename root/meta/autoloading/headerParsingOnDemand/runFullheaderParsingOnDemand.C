void runFullheaderParsingOnDemand(){
   gDebug=1;

   TClass::GetClass("myClass0<E>");
   TClass::GetClass("myClass1<E>");
   TClass::GetClass("myClass1<int>");
   TClass::GetClass("myClass2<E>");
   TClass::GetClass("myClass2<float>");
   TClass::GetClass("myClass3<E>");
   TClass::GetClass("myClass4<E>");
   TClass::GetClass("myClass5<E>");
   TClass::GetClass("myClass6<E>");
   TClass::GetClass("myClass7<E>");
   TClass::GetClass("myClass8<E>");

   std::cout << "\n";
}
