using namespace ROOT::Experimental;

int test_readTotemNtuple() {

   TDataFrame d("TotemNtuple", "Slimmed_TotemNTuple_9883.040.ntuple.root");
   auto validate_x = [](double v){
      std::cout << v << std::endl;
   };
   d.Foreach(validate_x, {"track_rp_3.x"});

   // now jitted
   auto count = d.Alias("t_y", "track_rp_3.y").Filter("cout << t_y << endl; return true;").Count();
   *count;

   return 0;
}
