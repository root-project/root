
// Helper struct for this test
struct MyStruct {
   int myint1;
   int myint2;
};

// Writes a `TTree` on a file. The `TTree` has the following branches:
// - floatb: branch of basic type (`float`)
// - arrayb: branch of type array of doubles, size `arraysize`
// - chararrayb: branch of type array of characters, size 10
// - vectorb: branch of type `std::vector<double>`, size `arraysize`
// - structb: struct branch of type `MyStruct`
// - structleafb: struct branch of type `MyStruct`, created as a leaf list
void CreateTTree(const char *filename, const char *treename, int nentries, int arraysize, int more,
                 const char* openmode)
{
   TFile f(filename, openmode);
   TTree t(treename, "Test tree");

   // Float branch
   float n;
   t.Branch("floatb", &n);

   // Array branch
   auto a = new double[arraysize];
   t.Branch("arrayb", a, std::string("arrayb[") + arraysize + "]/D");

   // Char array branch
   char s[10] = "onetwo";
   t.Branch("chararrayb", s, std::string("chararrayb[") + sizeof(s) + "]/C");

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

      if (i % 2 == 0)
         s[3] = '\0';
      else
         s[3] = 't';

      mystruct.myint1 = i + more;
      mystruct.myint2 = i * more;

      t.Fill();
   }

   f.Write();
   f.Close();

   delete[] a;
}

// Writes a `TNtuple` and a `TNtupleD` on a file. Both tuples have three branches (x,y,z)
void CreateTNtuple(const char *filename, const char *tuplename, int nentries, int more,
                   const char* openmode)
{

   std::string tuplenamed(tuplename);
   tuplenamed += "D";

   TFile f(filename, openmode);
   TNtuple ntuple(tuplename, "Test tuple", "x:y:z");
   TNtupleD ntupled(tuplenamed.c_str(), "Test tupled", "x:y:z");

   float x, y, z;
   for (int i = 0; i < nentries; ++i) {
      x = i;
      y = i + more;
      z = i + 2 * more;
      ntuple.Fill(x, y, z);
      ntupled.Fill(x, y, z);
   }

   f.Write();
   f.Close();
}
