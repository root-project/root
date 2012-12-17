{
   TFile *f1 = TFile::Open("skim.root");
   TFile*f2 = TFile::Open("BcMC.root");
#ifdef ClingWorkAroundMissingDynamicScope
   TTree* rootTree; 
   TH1F *h1, *h2, *h3, *h4;
   f1->cd(); 
   gDirectory->GetObject("rootTree",rootTree); 
   rootTree->Draw("Bpsi.mjpi>>h1","","",10);
   f2->cd(); 
   gDirectory->GetObject("rootTree",rootTree); 
   rootTree->Draw("Bpsi.mjpi>>h2","","same",100);
   f1->cd(); 
   gDirectory->GetObject("rootTree",rootTree); 
   rootTree->Draw("Bpsi.ctaujpi>>h3","","",10);
   f2->cd(); 
   gDirectory->GetObject("rootTree",rootTree); 
   rootTree->SetBranchStatus("Bpsi.ctau",1);
   f2->cd(); 
   gDirectory->GetObject("rootTree",rootTree); 
   rootTree->Draw("Bpsi.ctau>>h4","","same",100);

   f1->cd();
   h1 = (TH1F*)gROOT->FindObject("h1");
   if (h1->GetMean()==0) {
      fprintf(stdout,"Problem generating: %s\n",h1->GetTitle());
   }
   h3 = (TH1F*)gROOT->FindObject("h3");
   if (h3->GetMean()==0) {
      fprintf(stdout,"Problem generating: %s\n",h3->GetTitle());
   }
   f2->cd();
   h2 = (TH1F*)gROOT->FindObject("h2");
   if (h2->GetMean()==0) {
      fprintf(stdout,"Problem generating: %s\n",h2->GetTitle());
   }
   h4 = (TH1F*)gROOT->FindObject("h4");
   if (h4->GetMean()!=0) {
      fprintf(stdout,"Problem generating: %s (it should have been full of zeros ... actually this might indicate progress ::) )\n",h4->GetTitle());
   }
   delete f1;
   delete f2;
#else
   TTree* rootTree; 
   f1->cd(); rootTree->Draw("Bpsi.mjpi>>h1","","",10);
   f2->cd(); rootTree->Draw("Bpsi.mjpi>>h2","","same",100);
   f1->cd(); rootTree->Draw("Bpsi.ctaujpi>>h3","","",10);
   f2->cd(); rootTree->SetBranchStatus("Bpsi.ctau",1);
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
   delete f1;
   delete f2;
#endif
}
