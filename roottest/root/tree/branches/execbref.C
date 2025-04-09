{   
   auto fFile = TFile::Open("copy.root","RECREATE");
   TTree *tree = new TTree("T","T");
   int i;
   tree->Branch("i",&i);
   tree->BranchRef();
   TFile *temp = TFile::Open("temp.root","RECREATE");
   temp->cd();
   auto newTree = tree->CloneTree(0);
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
