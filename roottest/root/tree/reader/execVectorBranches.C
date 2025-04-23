#include <vector>

void printReader(TTreeReaderArray<Float_t>& reader) {
   fprintf(stderr, "  Branch %s\n", reader.GetBranchName());
   fprintf(stderr, "    Size: %zu\n", reader.GetSize());
   fprintf(stderr, "    Elements: ");
   for (Int_t i = 0; i < reader.GetSize(); ++i) fprintf(stderr, " %.0f", reader[i]);
   fprintf(stderr, "\n");
}

void execVectorBranches() {
   // Phase 1 - generate tree
   {
      TFile f("testVectorBranches.root", "RECREATE"); // Create file
      TTree tree("TestVectorBranches", "Tree with vectors of built in types"); // Create tree

      // Two vectors in the tree
      std::vector<Float_t> vx;
      std::vector<Float_t> vy;

      // Each vector has a separate branch
      tree.Branch("vx", &vx);
      tree.Branch("vy", &vy);

      // Fill tree
      for (Int_t j = 1; j < 10; ++j) {
         vx.clear();
         vy.clear();
         // Add elements to X
         for (Int_t k = 0; k < j; ++k) {
            vx.push_back(100 + k);
         }
         // Add elements to Y
         for (Int_t k = 0; k < j + 5; ++k) {
            vy.push_back(200 + k);
         }
         tree.Fill();
      }
      f.Write(); f.Close(); // Write tree to file
   }
   // Phase 2 - read tree
   {
      // Open tree
      TFile f("testVectorBranches.root");
      TTreeReader reader("TestVectorBranches", &f);
      TTreeReaderArray<Float_t> reader_x(reader, "vx");
      TTreeReaderArray<Float_t> reader_y(reader, "vy");

      // Print entries
      Int_t entry = 0;
      while (reader.Next()) {
         fprintf(stderr, "Entry %d\n", entry++);
         printReader(reader_x);
         printReader(reader_y);
      }
   }
}
