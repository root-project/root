

int test_readTotemNtuple() {

   ROOT::RDataFrame d("TotemNtuple", "Slimmed_TotemNTuple_9883.040.ntuple.root");

   // Foreach with dotted branch name
   std::cout << "Foreach:" << std::endl;
   auto validate_x = [](double v){
      std::cout << v << std::endl;
   };
   d.Foreach(validate_x, {"track_rp_3.x"});

   // Alias plus jitted filter
   std::cout << "Alias plus jitted filter:" << std::endl;
   auto count1 = d.Alias("t_y", "track_rp_3.y").Filter("cout << t_y << endl; return true;").Count();
   *count1;

   // Jitted filter with single-dotted branch name
   std::cout << "Jitted filter, single-dotted name:" << std::endl;
   auto count2 = d.Filter("cout << track_rp_3.y << endl; return true;").Count();
   *count2;

   // Jitted filter with multi-dotted branch name
   // In debug mode RDataFrame warns when it needs to copy collection elements for reading
   // Ignore the warnings which are not important for this test
   auto previousEIL = gErrorIgnoreLevel;
   gErrorIgnoreLevel = kError;
   std::cout << "Jitted filter, multi-dotted name:" << std::endl;
   auto count3 = d.Filter("cout << par_patterns_rp_0.u.w << endl; return true;").Count();
   *count3;
   gErrorIgnoreLevel = previousEIL;

   return 0;
}
