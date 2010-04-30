{   
   
   fFile = TFile::Open("copy.root","RECREATE");
   TTree *tree = new TTree("T","T");
   int i;
   tree->Branch("i",&i);
   tree->BranchRef();
   TFile *temp = TFile::Open("temp.root","RECREATE");
   temp->cd();
   newTree = tree->CloneTree(0);
   newTree->SetDirectory(fFile);
   
   delete temp;
   newTree->Fill();
   if ( newTree->GetBranchRef()->GetFile() != fFile ) {
      cout << "Error: the branch ref is not set correctly\n";
      return 1;
   } else {
      return 0;
   }
}