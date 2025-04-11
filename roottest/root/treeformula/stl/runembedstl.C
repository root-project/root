{
// Fill out the code of the actual test
   new TFile("Cumpb.root");
#ifdef ClingWorkAroundMissingDynamicScope
   TTree *Events;
   gFile->GetObject("Events",Events);
#endif
   Events->Draw("CrossingFrame_mix_label_11.obj.signal_._data.theEnergyLoss - CrossingFrame_mix_label_11.obj.signal_.theEnergyLoss");
#ifdef ClingWorkAroundMissingDynamicScope
   TH1F *htemp;
   htemp = (TH1F*)gROOT->FindObject("htemp");
#endif
   htemp->Print();
   cout << "The mean should be 0: " << htemp->GetMean() << endl;
#ifdef ClingWorkAroundBrokenUnnamedReturn
   gApplication->Terminate(0);
#else
   return 0;
#endif
}
