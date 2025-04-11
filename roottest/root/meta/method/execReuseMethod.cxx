int execReuseMethod() {
   auto cl = TClass::GetClass("vector<TObject*>");
   if (cl->GetState() != TClass::kInterpreted) {
      Error("reuseMethod", "Found a dictionary for %s, state: %d", cl->GetName(), (int)cl->GetState());
      return 1;
   }
   auto m = cl->GetMethodWithPrototype("size","");
   if (m != cl->GetListOfMethods(false)->At(0)) {
      cl->GetListOfMethods(false)->ls();
      Error("reuseMethod", "Test invariant violated (method searched for is not the only one there.  We looked for 'size'");
      return 2;
   }
   gInterpreter->GenerateDictionary("vector<TObject*>");
   cl = TClass::GetClass("vector<TObject*>");
   if (cl->GetState() != TClass::kHasTClassInit) {
      Error("reuseMethod", "Dictionary generation failed for %s, state: %d", cl->GetName(), (int)cl->GetState());
      return 3;
   }
   if (m != cl->GetListOfMethods(false)->At(0)) {
      cl->GetListOfMethods(false)->ls();
      Error("reuseMethod", "The reuse is not working properly, TFunction for 'size' is no longer there or has moved.");
      return 4;
   }
   return 0;
}

