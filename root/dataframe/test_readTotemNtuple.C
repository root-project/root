using namespace ROOT::Experimental;

int test_readTotemNtuple() {

   TDataFrame d("TotemNtuple", "Slimmed_TotemNTuple_9883.040.ntuple.root");
   auto validate_x = [](double v){
      std::cout << v << std::endl;
   };
   d.Foreach(validate_x, {"track_rp_3.x"});
   return 0;
}
