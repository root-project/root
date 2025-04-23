{
// Fill out the code of the actual test
   gROOT->ProcessLine(".L loadcode.C+");
#ifdef ClingWorkAroundMissingDynamicScope
   gROOT->ProcessLine("jetwrite();");
#else
   jetwrite();
#endif
   TFile *_file0 = TFile::Open("jetclass.root");

#ifdef ClingWorkAroundMissingDynamicScope
   TTree *t; _file0->GetObject("t",t);
#endif

   new TCanvas;
   t->Draw("jet[].v_x[jet.nbest]","jet.nbest>=0");
   new TCanvas;
   t->Draw("jet[].v_x[jet.nbest+(jet.nbest==-1)]","");
      
#ifdef ClingWorkAroundBrokenUnnamedReturn
   gApplication->Terminate(0);
#else    
      return (0);
#endif
}
