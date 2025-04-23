void runReflexCintexOrder() {
   gSystem->Setenv("LINES","-1");
   gSystem->Load("libReflex");
   gSystem->Load("libReflexDict");
   gSystem->Load("libCintex");
   ROOT::Cintex::Cintex::Enable();
   TClass* cl = TClass::GetClass("Reflex::Scope");
   TMethod* m = (TMethod*) cl->GetListOfMethods()->FindObject("AddMemberTemplate");
   TMethodArg* arg = (TMethodArg*) m->GetListOfMethodArgs()->First();
   cout << "AddMemberTemplate() arg0: " << arg->GetFullTypeName() << endl;
}
