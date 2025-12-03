{
   // Test the ability to read a v5 formula stored member-wise in a TClonesArray.
   TFile *_file0 = TFile::Open("v5formula_clones.root");
   if (fTS->GetEntry(0) <= 0) {
     Error("","Reading the entry failed");
     return 1;
   }

   TClonesArray *arr = nullptr;
   fTS->SetBranchAddress("f_Int0",&arr);
   fTS->GetEntry(1);
   arr->Print();
   //arr->ls("noaddr");
}
