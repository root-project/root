void test_ROOT9975()
{
   ROOT::RDataFrame df("t", "ROOT9975.root");
   std::vector<std::string> types;
   for (auto &&c : df.GetColumnNames())
      types.emplace_back(df.GetColumnType(c));
   const std::vector<std::string> expected_types = {"Int_t", "Int_t", "Int_t", "B", "A", "A", "Int_t", "Int_t"};

   R__ASSERT(types == expected_types);

   return 0;
}
