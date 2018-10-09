
ULong64_t triggerBug()
{
   auto filenames = {"input1.root", "input2.root"};
   TChain input_chain("MCDecayTree");
   for (const auto &filename : filenames) {
     input_chain.Add(filename);
   }
   ROOT::RDataFrame d_input(input_chain);
   auto m = d_input.Mean<float>("mup_PHI");
   return *d_input.Count();
}

int test_chainZombieFile() {

   auto c = triggerBug();
   bool isExpectedCount = 1 == c;

   ROOT::EnableImplicitMT();
   auto cmt = triggerBug();
   isExpectedCount &= 1 == cmt;

   if (!isExpectedCount) {
      std::cerr << "The test failed. The count was " << c << " and " << cmt << " repectively\n";
   }

   return isExpectedCount ? 0 : 1;
}
