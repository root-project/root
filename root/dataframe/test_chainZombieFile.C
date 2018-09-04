
void triggerBug()
{
   auto filenames = {"input1.root", "input2.root"};
   TChain input_chain("MCDecayTree");
   for (const auto &filename : filenames) {
     input_chain.Add(filename);
   }
   ROOT::RDataFrame d_input(input_chain);
   *d_input.Mean<float>("mup_PHI");
}

int test_chainZombieFile() {

   triggerBug();

   ROOT::EnableImplicitMT();
   triggerBug();
   return 0;
}
