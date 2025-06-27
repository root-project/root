#include "TFile.h"
#include "TTree.h"

#include "ROOT/RDataFrame.hxx"

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

int regression_snapshot()
{

   auto fileName = "test_regressionsnapshot.root";
   auto treeName = "myTree";
   fill_tree(fileName, treeName);

   ROOT::RDataFrame d(treeName, fileName);

   auto d_cut = d.Filter("b1 % 2 == 0");

   auto d2 = d_cut.Define("a","A(b1)")
                  .Define("b1_square", "b1 * b1")
                  .Define("b2_vector", [](float b2){ std::vector<float> v; for (int i=0;i < 3; i++) v.push_back(b2*i); return v;}, {"b2"});

   {
      auto outFileName = "test_regr_snapshot_output1.root";
      d2.Snapshot(treeName, outFileName, {"b1", "b1_square", "b2_vector", "a"});

      // Open the new file and list the branche of the trees
      TFile f(outFileName);
      TTree* t;
      f.GetObject(treeName, t);
      auto l = t->GetListOfBranches();
      std::cout << "first list of branches" << std::endl;
      for (auto branch : *t->GetListOfBranches()) {
         std::cout << "Jitted branch: " << branch->GetName() << std::endl;
      }
      f.Close();
   }

   {
      auto outFileName = "test_regr_snapshot_output2.root";
      d2.Snapshot(treeName, outFileName, {"b1", "b1_square", "b2_vector", "a"});

      // Open the new file and list the branche of the trees
      TFile f(outFileName);
      TTree* t;
      f.GetObject(treeName, t);
      std::cout << "\nsecond list of branches" << std::endl;
      for (auto branch : *t->GetListOfBranches()) {
         std::cout << "Branch: " << branch->GetName() << std::endl;
      }
      f.Close();
   }

   return 0;
}

int main(){return regression_snapshot();}
