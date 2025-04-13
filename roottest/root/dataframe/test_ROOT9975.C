void test_ROOT9975()
{
   ROOT::RDataFrame df("t", "ROOT9975.root");
   std::vector<std::string> types;
   auto cols = df.GetColumnNames();
   std::sort(cols.begin(), cols.end());
   for (auto &&c : cols) {
      types.emplace_back(df.GetColumnType(c));
      std::cout << df.GetColumnType(c) << " ";
   }
   const std::vector<std::string> expected_types = {"A", "Int_t", "B", "A", "Int_t", "Int_t", "Int_t", "Int_t"};

   R__ASSERT(types == expected_types);
}
