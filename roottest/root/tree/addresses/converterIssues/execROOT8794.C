R__LOAD_LIBRARY(ExtHit/ExtHit)

void execROOT8794() {

   auto file = TFile::Open("belle2-ROOT-8794.root");
   TTree *tree; file->GetObject("tree",tree);

   // tree->SetBranchStatus("*", 0);
   // tree->SetBranchStatus("ExtHits*", 1);

   TClonesArray* exthits = 0;
   tree->SetBranchAddress("ExtHits", &exthits);
   TBranch *br = tree->GetBranch("ExtHits");

   // Dump contents of the 6x6 covariance matrix
   for (int i = 0; i < tree->GetEntries(); i++) {

      //tree->GetEntry(i); // , 0);
      //gDebug = 7;
      //     TClass::GetClass("Belle2::ExtHit")->GetStreamerInfos()->ls();
      //     br->Print("debugInfo");
      br->GetEntry(i);
      for (int j = 0; j < exthits->GetEntriesFast(); ++j) {
         Belle2::ExtHit* e = (Belle2::ExtHit*)((*exthits)[j]);
         printf("Event %d hit %d: ", i, j);
#if 0
         TMatrixDSym m(e->getCovariance());
         for (int r = 0; r < m.GetNrows(); ++r) {
            for (int c = 0; c <= r; ++c) {
               printf(" %f", m(r,c));
            }
         }
#else
         for (int r = 0; r < 21; ++r) {
            printf(" %f", e->m_Cov[r]);
         }
#endif
         printf("\n");
      }
   }
}

void trim() {
   // File was already opened by user
   auto file = TFile::Open("belle2.root");
   TTree *tree; file->GetObject("tree",tree);
   tree->SetBranchStatus("*", 0);
   tree->SetBranchStatus("ExtHits*", 1);
   TClonesArray* exthits = 0;
   tree->SetBranchAddress("ExtHits", &exthits);
   TBranch *br = tree->GetBranch("ExtHits");

   TFile *outfile = TFile::Open("b2.root","RECREATE");
   TTree *outtree = tree->CloneTree(-1);
   outfile->Write();
   
}
