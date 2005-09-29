{
// Fill out the code of the actual test
   new TFile("Cumpb.root");
   Events->Draw("CrossingFrame_mix_label_11.obj.signal_._data.theEnergyLoss - CrossingFrame_mix_label_11.obj.signal_.theEnergyLoss");
   htemp->Print();
   cout << "The mean should be 0: " << htemp->GetMean() << endl;
   return 0;
}
