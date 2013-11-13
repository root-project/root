void runCintexReflexOrder() {
   gSystem->Setenv("LINES","-1");
   gSystem->Load("libCintex");
   gSystem->Load("libReflexDict");
   ROOT::Cintex::Cintex::Enable();
   TClass* cl = TClass::GetClass("Reflex::Scope");
   TMethod* m = (TMethod*) cl->GetListOfMethods()->FindObject("AddMemberTemplate");
   TMethodArg* arg = (TMethodArg*) m->GetListOfMethodArgs()->First();
   cout << "AddMemberTemplate() arg0: " << arg->GetFullTypeName() << endl;
}
