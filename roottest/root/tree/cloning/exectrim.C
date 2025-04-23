
void clone(TChain *oldtree) {

   //Create a new file + a clone of old tree header. Do not copy events
   TFile * newfile = TFile::Open("output_Coulomb_LER_study_small.root","recreate");
   TTree * newtree = oldtree->CloneTree(0);
   if (!newtree) return;

   // Here we copy the branches
   newtree->CopyEntries(oldtree);

   //newtree->CopyEntries(oldtree);

   // Flush to disk
   newfile->Write();

   newtree->Print("");

   // Clean
   delete newfile;

}

void exectrim() {

   //Get old file, old tree and set top branch address
   TChain* chain=new TChain("tree");
   chain->Add("output_Coulomb_LER_study_10.root");
   // Deactivate all branches
   chain->SetBranchStatus("*",0);
   // Activate 4 branches only: our skim
   chain->SetBranchStatus("StrSimHits",1);

   clone(chain);

   // And try again with more branches.
   chain->SetBranchStatus("StrSimHits*",1);

   clone(chain);


}

