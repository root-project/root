void writeOriginal() {
#ifndef ClingWorkAroundMissingDynamicScope
   gROOT->ProcessLine(".L cloningOne.C+");
#endif
   cloning * c = new cloning;
   TFile *file = new TFile("cloning.root","RECREATE");
   TTree *tree = new TTree("T","T");
   tree->Branch("obj",&c);
   tree->Fill();
   tree->Fill();
   tree->Write();
   delete file;
}

void writeClone() {
   TFile *file = new TFile("cloning.root","READ");
   TTree *orig; file->GetObject("T",orig);
   gROOT->ProcessLine(".L cloningTwo.C+");
   TFile *out = new TFile("cloning2.root","RECREATE");
   //orig->Show(0);
   TTree *copy = orig->CloneTree();
   copy->Write();
   copy->Show(0);
   delete out;
}

void runcloning(int type = 2) 
{
   switch(type) {
      case 1: writeOriginal(); break;
      case 2: writeClone(); break;
     default: fprintf(stdout,"Missing case in runcloning\n");
   }
}
