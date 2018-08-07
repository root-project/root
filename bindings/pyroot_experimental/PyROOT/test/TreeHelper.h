
// Helper struct for this test
struct MyStruct {
   int myint1;
   int myint2;
};

// Writes a `TTree` on a file. The `TTree` has the following branches:
// - floatb: branch of basic type (`float`)
// - arrayb: branch of type array of doubles, size `arraysize`
// - vectorb: branch of type `std::vector<double>`, size `arraysize`
// - structb: struct branch of type `MyStruct`
// - structleafb: struct branch of type `MyStruct`, created as a leaf list
void CreateTTree(const char *filename, const char *treename, int nentries, int arraysize, int more)
{
   TFile f(filename, "RECREATE");
   TTree t(treename, "Test tree");

   // Float branch
   float n;
   t.Branch("floatb", &n);

   // Array branch
   auto a = new double[arraysize];
   t.Branch("arrayb", a, std::string("arrayb[") + arraysize + "]/D");

   // Vector branch
   std::vector<double> v(arraysize);
   t.Branch("vectorb", &v);

   // Struct branches
   MyStruct mystruct;
   t.Branch("structb", &mystruct);
   t.Branch("structleaflistb", &mystruct, "myintll1/I:myintll2/I");

   for (int i = 0; i < nentries; ++i) {
      n = i + more;

      for (int j = 0; j < arraysize; ++j) {
         a[j] = v[j] = i + j;
      }

      mystruct.myint1 = i + more;
      mystruct.myint2 = i * more;

      t.Fill();
   }

   f.Write();
   f.Close();

   delete[] a;
}
