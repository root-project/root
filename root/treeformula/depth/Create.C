void Create() {
   TFile f("depth.root","recreate");
   TTree *T=new TTree("T","junk");
   TLine *line = new TLine(0,0,1,1);
   TLine *line2 = new TLine(0.2,0.2,0.6,0.6);
   TLine *line3 = new TLine(0.4,0.4,0.8,0.8);
   TBranchElement *br =(TBranchElement*)T->Branch("line","TLine",&line);
   TBranchElement *br2=(TBranchElement*)br->Branch("line2","TLine",&line2);
   TBranchElement *br3=(TBranchElement*)br2->Branch("line3","TLine",&line3);
   Double_t a=1.88;
   br3->Branch("a",&a,"a/D");
   T->Fill();
   T->Fill();
   T->Print();
   T->Write();
   T->Show(0);
}
