{
   TTree tree("tester","tree with missing branches");
   std::vector<int> values;
   tree.Branch("values",&values);
   values.push_back(1);
   values.push_back(2);
   tree.Fill();
   values.clear();
   values.push_back(3);
   values.push_back(4);
   tree.Fill();

   TTreeFormula tf("testing","values[do_not_exist]",&tree);
   if (tf.GetNdim() > 0) {
      Error("Test","Syntax error in '%s' was not detected properly",tf.GetTitle());
   }

   tree.Scan("values[do_not_exist]");
   tree.Scan("values[0] + do_not_exist");

   return 0;
}
