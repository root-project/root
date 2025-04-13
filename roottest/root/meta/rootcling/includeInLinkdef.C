int includeInLinkdef() {
   auto c = TClass::GetClass("classForLinkdef");
   if (!c) return 1;
   classForLinkdef test;
   return 0;
}
