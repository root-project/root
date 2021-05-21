#include "TFile.h"
#include "TROOT.h"
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

int do_work(const char* fileName, const char* outFileName, const char* treeName, const char* outTreeName) {
   ROOT::RDataFrame d(treeName, fileName);

   auto d_cut = d.Filter("b1 % 2 == 0");

   auto d2 = d_cut.Define("a","A(b1)")
                  .Define("b1_square", "b1 * b1")
                  .Define("b2_vector", [](float b2){ std::vector<float> v; for (int i=0;i < 3; i++) v.push_back(b2*i); return v;}, {"b2"});

   ROOT::RDF::RSnapshotOptions opts;
   // we need a small autoflush setting to trigger partial merges with TFileMerger in the MT case, which
   // tests https://github.com/root-project/root/issues/8226
   opts.fAutoFlush = 10;

   /****** non-jitted snapshot *******/
   auto snapshot_tdf = d2.Snapshot<int, int, std::vector<float>, A>(outTreeName, outFileName,
                                                                    {"b1", "b1_square", "b2_vector", "a"}, opts);

   // Open the new file and list the branches of the tree
   TFile f(outFileName);
   TTree* t;
   f.GetObject(outTreeName, t);
   for (auto branch : *t->GetListOfBranches()) {
      std::cout << "Branch: " << branch->GetName() << std::endl;
   }
   f.Close();

   auto mean_b1 = snapshot_tdf->Mean("b1");
   auto mean_a = snapshot_tdf->Define("a_val",[](A& a){return a.GetI();},{"a"}).Mean("a_val");

   std::cout << "Means:" << *mean_b1 << " " << *mean_a << std::endl;

   if (*mean_b1 != *mean_a) {
      std::cerr << "Error: the mean values of two branches which are supposed to be identical differ!\n";
      return 1;
   }

   /****** jitted snapshot *******/
   auto snapshot_jit =  d2.Snapshot(outTreeName, outFileName, {"b1", "b1_square", "b2_vector", "a"});

   // Open the new file and list the branche of the trees
   TFile jit_f(outFileName);
   TTree* jit_t;
   jit_f.GetObject(outTreeName, jit_t);
   auto l = jit_t->GetListOfBranches();
   for (auto branch : *l) {
      std::cout << "Jitted branch: " << branch->GetName() << std::endl;
   }
   jit_f.Close();

   auto jit_mean_b1 = snapshot_jit->Mean("b1");
   auto jit_mean_a = snapshot_jit->Define("a_val",[](A& a){return a.GetI();},{"a"}).Mean("a_val");

   std::cout << "Jitted means:" << *jit_mean_b1 << " " << *jit_mean_a << std::endl;

   if (*jit_mean_b1 != *jit_mean_a || *mean_b1 != *jit_mean_b1) {
      std::cerr << "Error: the jitted mean values of two branches which are supposed to be identical differ!\n";
      return 1;
   }

   // now we exercise the regexp functionality
   auto snapshot_jit2 =  d2.Snapshot(outTreeName, outFileName, "b[1,2].*");

   // Open the new file and list the branche of the trees
   TFile jit_f2(outFileName);
   TTree* jit_t2;
   jit_f2.GetObject(outTreeName, jit_t2);
   auto l2 = jit_t2->GetListOfBranches();
   for (auto branch : *l2) {
      std::cout << "Jitted branch: " << branch->GetName() << std::endl;
   }
   jit_f.Close();

   return 0;
}

int runTest() {
   auto fileName = "test_snapshot.root";
   auto outFileName = "test_snapshot_output.root";
   auto treeName = "myTree";
   auto outTreeName = "myTree";
   fill_tree(fileName, treeName);

   std::cout << "---- Now with a tree in the root directory\n";
   int ret = do_work(fileName, outFileName, treeName, outTreeName);

   // now we put the tree in a directory
   outFileName = "test_snapshot_inDirectory_output.root";
   treeName = "myTree";
   outTreeName = "a/myTree";
   std::cout << "---- Now with a tree in a subdirectory\n";
   ret += do_work(fileName, outFileName, treeName, outTreeName);

   return ret;
}

int test_snapshot()
{
   int ret(0);
   ret = runTest();
   std::cout << "+++++++++ Now MT\n";
#ifdef R__USE_IMT
   ROOT::EnableImplicitMT(4);
#endif
   ret += runTest();

   return ret;

}
