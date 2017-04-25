#include "TFile.h"
#include "TH1F.h"
#include "TTree.h"

#include "ROOT/TDataFrame.hxx"

void fill_tree(const char *filename, const char *treeName)
{
   TFile f(filename, "RECREATE");
   TTree t(treeName, treeName);
   int b1;
   float b2;
   t.Branch("b1", &b1);
   t.Branch("b2", &b2);
   for (int i = 0; i < 100; ++i) {
      b1 = i;
      b2 = i * i;
      t.Fill();
   }
   t.Write();
   f.Close();
   return;
}

class A {
public:
   A(){};
   A(int i):fI(i){}
   int GetI(){return fI;}
private:
   int fI = 0;
};

int test_snapshot()
{

   auto fileName = "test_snapshot.root";
   auto outFileName = "test_snapshot_output.root";
   auto treeName = "myTree";
   fill_tree(fileName, treeName);

   ROOT::Experimental::TDataFrame d(treeName, fileName);

   auto d_cut = d.Filter("b1 % 2 == 0");

   auto d2 = d_cut.Define("a","A(b1)")
                  .Define("b1_square", "b1 * b1")
                  .Define("b2_vector", [](float b2){ std::vector<float> v; for (int i=0;i < 3; i++) v.push_back(b2*i); return v;}, {"b2"});


   auto snapshot_tdf =  d2.Snapshot<int, int, std::vector<float>, A>(treeName, outFileName, {"b1", "b1_square", "b2_vector", "a"});

   // Open the new file and list the branches of the tree
   TFile f(outFileName);
   TTree* t;
   f.GetObject(treeName, t);
   for (auto branch : *t->GetListOfBranches()) {
      std::cout << "Branch: " << branch->GetName() << std::endl;
   }
   f.Close();

   auto mean_b1 = snapshot_tdf.Mean("b1");
   auto mean_a = snapshot_tdf.Define("a_val",[](A& a){return a.GetI();},{"a"}).Mean("a_val");

   std::cout << "Means:" << *mean_b1 << " " << *mean_a << std::endl;

   if (*mean_b1 != *mean_a) {
      std::cerr << "Error: the mean values of two branches which are supposed to be identical differ!\n";
      return 1;
   }

   return 0;
}
