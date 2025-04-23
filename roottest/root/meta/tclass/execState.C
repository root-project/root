void writeState() {
  gROOT->ProcessLine(".L libState.cxx+");
  gROOT->ProcessLine("writeFile();");
} 

int execState() {
   int result = 0;

   gInterpreter->Declare("class Event;");
   TClass *c = TClass::GetClass("Event"); 
   if (TClass::kForwardDeclared != c->GetState()) {
      Error("execState","State is %d instead of %d (TClass::kForwardDeclared)",
            (int)c->GetState(),(int)TClass::kForwardDeclared);
      ++result;
   }
   TFile *f = TFile::Open("tc_state.root");
   c = TClass::GetClass("Event");
   if (TClass::kEmulated != c->GetState()) {
      Error("execState","State is %d instead of %d (TClass::kEmulated)",
            (int)c->GetState(),(int)TClass::kEmulated);
      ++result;
   }

   gROOT->ProcessLine(".L libState.cxx+");
   c = TClass::GetClass("Event");
   if (TClass::kHasTClassInit != c->GetState()) {
      Error("execState","State is %d instead of %d (TClass::kHasTClassInit)",
            (int)c->GetState(),(int)TClass::kHasTClassInit);
      ++result;
   }
   return result;
}
