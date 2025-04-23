{
#if defined(ClingWorkAroundIncorrectTearDownOrder)
   if (1) {
#endif
      
   TFile f("Event.root");
   gSystem->Load("./libEvent_2"); 
   
#ifdef ClingWorkAroundMissingDynamicScope
   TTree *T; f.GetObject("T",T);
#endif
   
   T->Show(5); //ok
   T->Scan("fTemperature"); //ok
   // gSystem->Load("libTreePlayer");
   auto tf = new TTreeFormula("tf","fTemperature",T);
      
#if defined(ClingWorkAroundIncorrectTearDownOrder)
   }
#endif
}
