{   
#ifdef ClingWorkAroundMissingImplicitAuto
   TTree *newTree;
   TFile *
#endif
   fFile = TFile::Open("copy.root","RECREATE");
#ifdef ClingWorkAroundUnnamedIncorrectInitOrder
   TTree *tree; tree = new TTree("T","T");
#else
   TTree *tree = new TTree("T","T");
#endif
   int i;
   tree->Branch("i",&i);
   tree->BranchRef();
#ifdef ClingWorkAroundUnnamedIncorrectInitOrder
   TFile *temp; temp = TFile::Open("temp.root","RECREATE");
#else
   TFile *temp = TFile::Open("temp.root","RECREATE");
#endif
   temp->cd();
   newTree = tree->CloneTree(0);
   newTree->SetDirectory(fFile);
   
   delete temp;
   newTree->Fill();
   if ( newTree->GetBranchRef()->GetFile() != fFile ) {
      cout << "Error: the branch ref is not set correctly\n";
#ifdef ClingWorkAroundBrokenUnnamedReturn
      int res = 1;
   } else {
      int res= 0;
   }
#else
      return 1;
   } else {
      return 0;
   }
#endif
}