
int checkMissingDictionaries () {

   auto theClass = TClass::GetClass("ECont");
   THashTable ht;
   theClass->GetMissingDictionaries(ht);
   return nullptr != ht.FindObject("vector<Elem>") ? 0: 1;
}
