{
   gSystem->Load("./libdatamemberload");
   auto c = TClass::GetClass("Bottom");
   c = TClass::GetClass("Transient");
   if (c->GetState() != TClass::kForwardDeclared) {
      Error("datamemberload.C", "Unexpected state for TClass for %s expected kForwardDeclared(1) and got %d\n",
            c->GetName(), c->GetState());
      return 1;
   }
   auto l = (TListOfDataMembers *)c->GetListOfDataMembers(false);
   if (l->GetEntries()) {
      Error("datamemberload.C",
            "Unexpected content for the list of data member for class %s, it should have empty but had %d elements",
            c->GetName(), l->GetEntries());
      l->ls();
      return 2;
   }
   gROOT->ProcessLine("#include \"datamemberload.h\"");
   if (c->GetState() != TClass::kInterpreted) {
      Error("datamemberload.C", "Unexpected state for TClass for %s expected kInterpreted(3) and got %d\n",
            c->GetName(), c->GetState());
      return 3;
   }
   l = (TListOfDataMembers *)c->GetListOfDataMembers(false);
   if (l->GetEntries() != 1) {
      Error("datamemberload.C",
            "Unexpected content for the list of data member for class %s, it should have had one element but had %d "
            "elements",
            c->GetName(), l->GetEntries());
      l->ls();
      return 4;
   }
   l->Load();
   if (l->GetEntries() != 1) {
      Error("datamemberload.C",
            "Unexpected content for the list of data member for class %s, it should have had one element but had %d "
            "elements",
            c->GetName(), l->GetEntries());
      l->ls();
      return 5;
   }
   return 0;
}
