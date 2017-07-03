int test_missingBranches() {
   ROOT::Experimental::TDataFrame writer_d(1);
   writer_d.Define("a", [] { return 42; }).Snapshot("t", "f.root");

   // tdf with a file
   ROOT::Experimental::TDataFrame d("t", "f.root");
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
   ROOT::Experimental::TDataFrame d2(1);
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
