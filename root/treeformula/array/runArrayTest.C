{
   if (!gSystem->CompileMacro("Data.cxx","k")) gApplication->Terminate(1);
   TFile *f = new TFile("myTree.root");
   int n1 = tr->Draw("ns[40].adc[2]","");
   if (n1>=0) {
      cerr << "We should not have been able to draw: tr->Draw(\"ns[40].adc[2]\",\"\");\n";
      gApplication->Terminate(1);
   }
   int n2 = tr->Draw("ns[1].adc[40]>>h2","");
   h2->Print();
   int mean = (int)h2->GetMean();
   double expectedMean = 1000000;
   if ( TMath::Abs(mean-expectedMean)/expectedMean > 0.1 ) {
      cerr << "ERROR: The histogramming for tr->Draw(\"ns[1].adc[49]\",\"\"); is likely to be wrong\n";
      cerr << "ERROR: The calculated mean is " << mean << " instead of " << (int)expectedMean << endl;
      gApplication->Terminate(1);
   }

   const char*val = gSystem->Getenv("FAIL");
   if (val) {
      int n3 = tr->Draw("ns[1].subs.efg>>h3","");
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

