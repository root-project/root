
TFile *create(Int_t count) {

   TFile *input = new TMemFile("branchCount.root", "RECREATE", "", 0);
   TTree *tree = new TTree("ntuple","");

   Float_t *values = new Float_t[count];
   for(int i = 0; i < count; ++i) values[i] = (float)i + 1.0/3;

   tree->Branch("count", &count);
   tree->Branch("values", &(values[0]), "values[count]");

   for(int e = 0; e < 2; ++e)
      tree->Fill();

   // tree->Scan("");

   input->Write();

   delete tree;

   return input;
}

int Merge(TFile *input1, TFile *input2, bool slow, int verbosity)
{
   TFileMerger fileMerger(kFALSE, kFALSE);
   fileMerger.SetMsgPrefix("assertBranchCount");
   fileMerger.SetPrintLevel(verbosity);

   const char *outputname = "branchCountMerged.root";
   if (slow)
      outputname = "branchCountMergedSlow.root";

   int newcomp = 0;
   fileMerger.OutputFile(outputname, "RECREATE", newcomp);

   // Cause crash due to small allocated array size
   fileMerger.AddFile(input1);
   fileMerger.AddFile(input2);
   //fileMerger.AddFile(input1);

   // Slow for now.
   if (slow)
      fileMerger.SetFastMethod(kFALSE);
   else
      fileMerger.SetFastMethod(kTRUE);

   fileMerger.Merge();

   // And now check ...
   TFile *output = TFile::Open(outputname,"READ");
   if (!output || output->IsZombie()) {
      Error("assertBranchCount", "%s can not be opened properly", outputname);
      return 1;
   }
   TTree *tree;
   output->GetObject("ntuple",tree);
   if (!tree) {
      Error("assertBranchCount", "ntuple TTree is missing");
      return 2;
   }

   TLeaf *leaf = tree->GetLeaf("count");
   if (!leaf) {
      Error("assertBranchCount", "count branch is missing");
      return 3;
   }
   if (leaf->GetMaximum() != 5) {
      Error("assertBranchCount", "Count's fMaximum is unexpected %d instead of 5", leaf->GetMaximum());
      return 4;
   }

   return 0;
}

int assertBranchCount() {
   TFile *input1 = create(3);
   TFile *input2 = create(5);

   int verbosity = 0;

   int result = 0;
   result += Merge(input1, input2, kTRUE, verbosity);
   result += Merge(input1, input2, kFALSE, verbosity);
   return result;
}
