int hadd_check_nested_objects_argn_23666()
{
   auto file = std::unique_ptr<TFile>(TFile::Open("test_hadd_nested_objects_128files_23666_merged.root"));
   auto tree = std::unique_ptr<TTree>(file->Get<TTree>("A/B/tree"));

   if (auto nentries = tree->GetEntries(); nentries != 128) {
      std::cerr << "Number of entries is " << nentries << " (expected 128) \n";
      return 1;
   }

   return 0;
}
