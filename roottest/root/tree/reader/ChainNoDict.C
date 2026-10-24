// Read a data member of a split collection of a class without a dictionary
// over a TChain, which re-runs TBranchProxy::Setup() when switching files.
int ChainNoDict()
{
   auto c = TClass::GetClass("ChainNoDictHit");
   if (c && c->HasDictionary()) {
      std::cerr << "Class ChainNoDictHit has a dictionary and it should not." << std::endl;
      return 1;
   }

   TChain chain("t");
   chain.Add("chainNoDict0.root");
   chain.Add("chainNoDict1.root");
   TTreeReader r(&chain);
   TTreeReaderArray<double> x(r, "hits.fX");
   int nValues = 0;
   while (r.Next()) {
      const auto entry = r.GetCurrentEntry();
      const int file = entry / 10;
      if (x.GetSize() != static_cast<std::size_t>(entry % 10)) {
         std::cerr << "Wrong size " << x.GetSize() << " in entry " << entry << std::endl;
         return 1;
      }
      for (std::size_t i = 0; i < x.GetSize(); ++i) {
         const double expected = 100. * file + entry % 10 + 0.1 * i;
         if (std::abs(x[i] - expected) > 1e-9) {
            std::cerr << "Wrong value " << x[i] << " in entry " << entry << ", expected " << expected << std::endl;
            return 1;
         }
         ++nValues;
      }
   }
   if (nValues != 90) {
      std::cerr << "Read " << nValues << " values, expected 90" << std::endl;
      return 1;
   }
   return 0;
}
