struct simPos {
   simPos() : X(0),Y(-1),Z(-2) {};
   simPos(int val) : X(val),Y(val+2),Z(val+4) {};

   float X;
   float Y;
   float Z;
   float dummy; //!
};

void runleaflist(int kase = 0) {
   simPos pos(3);
   TFile::Open("leaflist.root","recreate");
   TTree *vertexTree = new TTree;
   vertexTree->Branch("simPos.",&pos,"X/F:Y/F:Z/F:3A");
   vertexTree->Fill();
   //vertexTree->Print();
   vertexTree->Draw("simPosProxy.C+");

   if (kase==0) {
#ifdef ClingWorkAroundUnloadingVTABLES
      fprintf(stderr,"Info in <TTreePlayer::DrawScript>: Will process tree/chain using generatedSel.h+\n");
#else
      gSystem->Unload("generatedSel_h");
      gSystem->Unlink("generatedSel_h.so");
      gSystem->Unlink("generatedSel_h.dll");
      TNtuple *ntuple = new TNtuple("ntuple","Demo ntuple","px:py:pz:random:i");
      ntuple->Fill(42);
      ntuple->Draw("hsimpleProxy.C+");
#endif
   }
}
