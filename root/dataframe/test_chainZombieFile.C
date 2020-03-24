
void triggerBug()
{
   auto filenames = {"input1.root", "input2.root"};
   TChain input_chain("MCDecayTree");
   for (const auto &filename : filenames) {
     input_chain.Add(filename);
   }
   ROOT::RDataFrame d_input(input_chain);
   auto m = d_input.Mean<float>("mup_PHI");
   *d_input.Count();
}

int test_chainZombieFile() {

   bool allHaveThrown = true;
   try {
   triggerBug();
   } catch (const std::runtime_error &) {
      allHaveThrown &= true;
   }

   ROOT::EnableImplicitMT();
   try {
   triggerBug();
   } catch (const std::runtime_error &) {
      allHaveThrown &= true;
   }

   if (!allHaveThrown)
      std::cerr << "The test failed. RDF did not throw when opening a corrupted file in the middle of an event loop\n";

   return allHaveThrown ? 0 : 1;
}
