   {
   gROOT->ProcessLine(".L libMyData.so");
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
      cerr << "The histogramming for tr->Draw(\"ns[1].adc[49]\",\"\"); is likely to be wrong\n";
      cerr << "The calculated mean is " << mean << " instead of " << (int)expectedMean << endl;
      gApplication->Terminate(1);
   }

}

