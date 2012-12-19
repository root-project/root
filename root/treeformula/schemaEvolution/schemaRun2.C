{
#if  defined(ClingWorkAroundIncorrectTearDownOrder)
   if (1) {
#endif
      
   TFile f("Event.root");
   gSystem->Load("./libEvent"); 

#ifdef ClingWorkAroundMissingDynamicScope
   TTree *T; f.GetObject("T",T);
#endif
      
   T->Show(5); //ok
   Long64_t n = T->Scan("fTemperature"); //ok

#ifdef ClingWorkAroundBrokenUnnamedReturn
   gApplication->Terminate(n!=0);
#else    
   return (n!=0);
#endif
#if defined(ClingWorkAroundIncorrectTearDownOrder)
   }
#ifndef ClingWorkAroundBrokenUnnamedReturn
   return 1;
#endif
#endif
}
