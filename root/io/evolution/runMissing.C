{
   TFile *f1 = TFile::Open("skim.root");
   TFile*f2 = TFile::Open("BcMC.root");
   f1->cd(); rootTree->Draw("Bpsi.mjpi>>h1","","",10);
   f2->cd(); rootTree->Draw("Bpsi.mjpi>>h2","","same",100);
   f1->cd(); rootTree->Draw("Bpsi.ctaujpi>>h3","","",10);
   f2->cd(); rootTree->Draw("Bpsi.ctau>>h4","","same",100);

   f1->cd();
   if (h1->GetMean()==0) {
      fprintf(stdout,"Problem generating: %s\n",h1->GetTitle());
   }
   if (h3->GetMean()==0) {
      fprintf(stdout,"Problem generating: %s\n",h3->GetTitle());
   }
   f2->cd();
   if (h2->GetMean()==0) {
      fprintf(stdout,"Problem generating: %s\n",h2->GetTitle());
   }
   if (h4->GetMean()!=0) {
      fprintf(stdout,"Problem generating: %s (it should have been full of zeros ... actually this might indicate progress ::) )\n",h4->GetTitle());
   }
}
