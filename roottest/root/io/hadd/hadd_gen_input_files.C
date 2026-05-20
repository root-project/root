using namespace ROOT::Experimental;

void hadd_gen_input_files(const char *fnamePrefix = "hadd_input")
{
   // Generate a bunch of files with a different set of objects
   for (int i = 0; i < 4; ++i) {
      auto fname = std::string(fnamePrefix) + std::to_string(i) + ".root";
      std::unique_ptr<TFile> file { TFile::Open(fname.c_str(), "RECREATE") };

      if (i > 0) {
         TTree tree("tree", "tree");
         int x;
         tree.Branch("x", &x);
         x = 42;
         tree.Fill();
         tree.Write();
      }
      if (i < 2) {
         TH1F h("hist", "hist", 10, 0, 10);
         for (int i = 0; i < 10; ++i)
            h.Fill(i);
         h.Write();
      }
      if (i > 1) {
         auto model = ROOT::RNTupleModel::Create();
         auto p = model->MakeField<char>("c");
         auto writer = ROOT::RNTupleWriter::Append(std::move(model), "ntpl", *file);
         *p = 22;
         writer->Fill();

         auto dir = file->mkdir("dir");
         dir->cd();
         TH1F h("hist", "hist", 10, 0, 10);
         for (int i = 0; i < 10; ++i)
            h.Fill(i);
         h.Write();
      }
      if (i == 3) {
         TFormula form("form", "sin(x)");
         file->WriteObject(&form, "form");
      }
   }
}
