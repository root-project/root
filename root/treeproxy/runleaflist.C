
struct simPos {
   float X;
   float Y;
   float Z;
};

void runleaflist(int kase = 0) {
   simPos pos;
   TTree *vertexTree = new TTree;
   vertexTree->Branch("simPos.",&pos,"X/F:Y/F:Z/F");
   vertexTree->Fill();
   //vertexTree->Print();
   vertexTree->Draw("simPosProxy.C+");

   if (kase==0) {
      gSystem->Unload("generatedSel_h");
      gSystem->Unlink("generatedSel_h.so");
      gSystem->Unlink("generatedSel_h.dll");
      TTree *ntuple = new TNtuple("ntuple","Demo ntuple","px:py:pz:random:i");
      ntuple->Fill();
      ntuple->Draw("hsimpleProxy.C+");
   }
}