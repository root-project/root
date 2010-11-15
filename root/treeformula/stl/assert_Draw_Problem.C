{
   TFile d("Draw_Problem.root");
   TCanvas* can = new TCanvas("test");
   can->Divide(2);

   bool result = true;
   if (0) {
      can->cd(1);
      tracks->Draw("shgloblen","itrack==43","",2,42);
      can->cd(2);
      tracks->Draw("shgloblen","itrack==43","",2,42);
   } else if (0) {
      can->cd(1);
      tracks->Draw("shgloblen","itrack==43 && 81<Iteration$ && Iteration$<83","",2,42);
      can->cd(2);
      tracks->Draw("shgloblen","itrack==43 && 81<Iteration$ && Iteration$<83","",2,42);
   } else {
      can->cd(1);
      tracks->Draw("shgloblen>>hfirst","itrack==43","");
      int first_mean = hfirst->GetMean();
      can->cd(2);
      tracks->Draw("shgloblen>>hsecond","itrack==43","");
      int second_mean = hsecond->GetMean();
      if (  TMath::Abs(first_mean - second_mean) > 1e-3 ) {
         cerr << "The 1st and 2nd run are different:" << first_mean << " vs " << second_mean << "\n";
         result = false;
      }
   }
   
//   tracks->Scan("shgloblen","itrack==43 && 81<Iteration$ && Iteration$<83","",2,42);
//   tracks->Scan("shgloblen","itrack==43 && 81<Iteration$ && Iteration$<83","",2,42);

   //   tracks->Scan("shgloblen","itrack==43");
   //   tracks->Scan("shgloblen","itrack==43");
}
