int execRefArrayCompress() {
   // gSystem->Load("./libtest");
   TFile *afile = new TFile("refArrayComp.root");
   TTree *tree; afile->GetObject("tree",tree);
   if (!tree) return 1;

   TRefArray *obj_arry = 0;
   tree->GetEntry(0);
   //tree->GetBranch("fObjAs")->Dump();

   Int_t offset;
   TClass::GetClass("Top")->GetStreamerInfo()->GetStreamerElement("fObjAs",offset);

   obj_arry = (TRefArray*)(tree->GetBranch("fObjAs")->GetAddress()+offset);
   //tree->SetBranchAddress("fObjAs",&obj_arry);
   for(Long64_t entry = 0; entry < tree->GetEntries(); ++entry) {
      tree->GetEntry(entry);
      printf("Events in TRefArray before compress: %d\n", obj_arry->GetEntriesFast());
      obj_arry->Compress();
      printf("Events in TRefArray before compress: %d\n", obj_arry->GetEntriesFast());
   }
   return 0;
}
