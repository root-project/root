void FillTree(const char* filename, const char* treeName) {
   TFile f(filename, "RECREATE");
   TTree t(treeName, treeName);
   int b;
   t.Branch("b", &b);
   for(b = 0; b < 3; ++b)
      t.Fill();
   t.Write();
   f.Close();
}

int test_listFilesCtor() {
   for (auto f : {"file_listFilesCtor1.root", "file_listFilesCtor2.root"})
      FillTree(f, "t");
   std::vector<std::string> files = {"file_listFilesCtor1.root", "file_listFilesCtor2.root"};
   ROOT::RDataFrame d1("t", files);
   ROOT::RDataFrame d2("t", {"file_listFilesCtor1.root", "file_listFilesCtor2.root"});
   return 0;
}
