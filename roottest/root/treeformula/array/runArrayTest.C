{
   // if (!gSystem->CompileMacro("Data.cxx","k")) gApplication->Terminate(1);
   TFile *f = TFile::Open("myTree.root");
#ifdef ClingWorkAroundMissingDynamicScope
   TTree *tr; f->GetObject("tr",tr);
#endif
   cerr << "We expect an error about the dimension size:\n";
   Long64_t n1 = tr->Draw("ns[40].adc[2]","");
   if (n1>=0) {
      cerr << "We should not have been able to draw: tr->Draw(\"ns[40].adc[2]\",\"\");\n";
      gApplication->Terminate(1);
   }
   Long64_t n2 = tr->Draw("ns[1].adc[40]>>h2","");
#ifdef ClingWorkAroundMissingDynamicScope
   TH1F *h2 = 0;
   h2 = (TH1F*)gROOT->FindObject("h2");
#endif
   h2->Print();
   int mean = (int)h2->GetMean();
   double expectedMean = 1000000;
   if ( TMath::Abs(mean-expectedMean)/expectedMean > 0.1 ) {
      cerr << "ERROR: The histogramming for tr->Draw(\"ns[1].adc[40]\",\"\"); is likely to be wrong\n";
      cerr << "ERROR: The calculated mean is " << mean << " instead of " << (int)expectedMean << endl;
      gApplication->Terminate(1);
   }

   const char*val = gSystem->Getenv("FAIL");
   if (val) {
      Long64_t n3 = tr->Draw("ns[1].subs.efg>>h3","");
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
   Long64_t n4 = tr->Draw("ns[2-1].adc[40]>>h4", "");
#ifdef ClingWorkAroundMissingDynamicScope
   TH1F *h4 = 0;
   h4 = (TH1F *)gROOT->FindObject("h4");
#endif
   // h4->Print();
   mean = (int)h4->GetMean();
   expectedMean = 1000000;
   if (TMath::Abs(mean - expectedMean) / expectedMean > 0.1) {
      cerr << "ERROR: The histogramming for tr->Draw(\"ns[2-1].adc[40]\",\"\"); is likely to be wrong\n";
      cerr << "ERROR: The calculated mean is " << mean << " instead of " << (int)expectedMean << endl;
      gApplication->Terminate(1);
   }
}

