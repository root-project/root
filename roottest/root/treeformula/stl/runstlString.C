{
#if defined(ClingWorkAroundIncorrectTearDownOrder)
   if (1) {
#endif

   TFile f("EDM.root");

#ifdef ClingWorkAroundMissingDynamicScope
   TTree *Events; f.GetObject("Events",Events);
#endif

   Events->SetScanField(0);
   Long64_t res = Events->Scan("CaloDataFrame0.obj.mycell.myBase");

#ifdef ClingWorkAroundBrokenUnnamedReturn
      gApplication->Terminate(res==0);
#else    
      return (res==0);
#endif
#if defined(ClingWorkAroundIncorrectTearDownOrder)
   }
#ifndef ClingWorkAroundBrokenUnnamedReturn
   return 1;
#endif
#endif
}
