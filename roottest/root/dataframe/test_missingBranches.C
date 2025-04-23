int test_missingBranches() {
   auto filename = "test_missingBranches.root";
   ROOT::RDataFrame writer_d(1);
   writer_d.Define("a", [] { return 42; }).Snapshot("t", filename);

   // tdf with a file
   ROOT::RDataFrame d("t", filename);
   try {
      auto h = d.Histo1D("test1");
      *h;
   } catch (const std::runtime_error &e) {
      std::cerr << e.what() << std::endl;
   }
   try {
      auto m = d.Max<int>("test2");
      *m;
   } catch (const std::runtime_error &e) {
      std::cerr << e.what() << std::endl;
   }

   // tdf with no file
   ROOT::RDataFrame d2(1);
   try {
      auto h2 = d2.Max("test3");
      *h2;
   } catch (const std::runtime_error &e) {
      std::cerr << e.what() << std::endl;
   }
   try {
      auto m2 = d.Histo1D<int>("test4");
      *m2;
   } catch (const std::runtime_error &e) {
      std::cerr << e.what() << std::endl;
   }

   return 0;
}
