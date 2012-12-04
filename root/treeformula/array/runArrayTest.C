{
   if (!gSystem->CompileMacro("Data.cxx","k")) gApplication->Terminate(1);
#ifdef ClingWorkAroundUnnamedIncorrectInitOrder
   TFile *f = 0;
   f = new TFile("myTree.root");
#else
   TFile *f = new TFile("myTree.root");
#endif
#ifdef ClingWorkAroundMissingDynamicScope
   TTree *tr; f->GetObject("tr",tr);
#endif
   cerr << "We expect an error about the dimension size:\n";
#ifdef ClingWorkAroundUnnamedIncorrectInitOrder
   Long64_t n1 = 0;
   n1 = tr->Draw("ns[40].adc[2]","");
#else
   Long64_t n1 = tr->Draw("ns[40].adc[2]","");
#endif
   if (n1>=0) {
      cerr << "We should not have been able to draw: tr->Draw(\"ns[40].adc[2]\",\"\");\n";
      gApplication->Terminate(1);
   }
#ifdef ClingWorkAroundUnnamedIncorrectInitOrder
   Long64_t n2 = 0;
   n2 = tr->Draw("ns[1].adc[40]>>h2","");
#else
   Long64_t n2 = tr->Draw("ns[1].adc[40]>>h2","");
#endif
#ifdef ClingWorkAroundMissingDynamicScope
   TH1F *h2 = 0;
   h2 = (TH1F*)gROOT->FindObject("h2");
#endif
   h2->Print();
#ifdef ClingWorkAroundUnnamedIncorrectInitOrder
   int mean = 0;
   mean = (int)h2->GetMean();
#else
   int mean = (int)h2->GetMean();
#endif
   double expectedMean = 1000000;
   if ( TMath::Abs(mean-expectedMean)/expectedMean > 0.1 ) {
      cerr << "ERROR: The histogramming for tr->Draw(\"ns[1].adc[49]\",\"\"); is likely to be wrong\n";
      cerr << "ERROR: The calculated mean is " << mean << " instead of " << (int)expectedMean << endl;
      gApplication->Terminate(1);
   }

   const char*val = gSystem->Getenv("FAIL");
   if (val) {
#ifdef ClingWorkAroundUnnamedIncorrectInitOrder
      Long64_t n3 = 0;
      n3 = tr->Draw("ns[1].subs.efg>>h3","");
#else
      Long64_t n3 = tr->Draw("ns[1].subs.efg>>h3","");
#endif
#ifdef ClingWorkAroundMissingDynamicScope
      TH1F *h3 = 0;
      h3 = (TH1F*)gROOT->FindObject("h3");
#endif
      h3->Print();
      mean = (int)h3->GetMean();
      expectedMean = 100000;
      if ( TMath::Abs(mean-expectedMean)/expectedMean > 0.1 ) {
         cerr << "ERROR: The histogramming for tr->Draw(\"ns[1].subs.efg>>h3\"); is likely to be wrong\n";
         cerr << "ERROR: The calculated mean is " << mean << " instead of " << (int)expectedMean << endl;
         gApplication->Terminate(1);
      }
   }

}

