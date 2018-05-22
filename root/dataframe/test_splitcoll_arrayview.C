#include "ROOT/RDataFrame.hxx"
#include "TFile.h"
#include "TTree.h"
#include "ROOT/TSeq.hxx"
#include "ROOT/RVec.hxx"

#include <iostream>

class A{
public:
   float a=0;
   float b=0;
   A(){};
   A(float d) : a(d), b(2*d) {};
   void Set(float d) {a = d; b = 2*d;}
   ClassDef(A,2)
};

ostream& operator<<(ostream& os, const A& a)
{
    os << "[" << a.a << ", " << a.b << "]";
    return os;
}

#ifdef __ROOTCLING__
#pragma link C++ class std::vector<A>+;
#endif

void fill_tree(const char* filename, const char* treeName) {
   TFile f(filename,"RECREATE");
   TTree t(treeName,treeName);
   std::vector<A> v;
   t.Branch("v", "vector<A>",&v, 32000, 2);
   for(auto i : ROOT::TSeqI(10)) {
      v.emplace_back(i);
      t.Fill();
   }
   t.Write();
   f.Close();
   return;
}

int test_splitcoll_arrayview() {
   auto fileName = "myfile_test_splitcoll_arrayview.root";
   auto treeName = "myTree";
   fill_tree(fileName,treeName);

   TFile f(fileName);
   try {
      ROOT::RDataFrame d(treeName, fileName, {"v.a"});
      auto c = d.Filter([](ROOT::VecOps::RVec<float> d) {
                   for (auto v : d)
                      std::cout << v << std::endl;
                   return d[0] > 5;
                }).Count();
      auto val = *c;
      std::cout << "count " << val << std::endl;
   } catch (const std::runtime_error& e) {
      std::cout << "Exception caught: " << e.what() << std::endl;
   }

   ROOT::RDataFrame d(treeName, fileName, {"v"});
   auto c = d.Filter([](ROOT::VecOps::RVec<A> d) {
      int q=0;
      for (auto v : d ) {
         std::cout << v.a << std::endl;
         q += v.a;
      }
      return 0 == q%3; }).Count();
   auto val = *c;
   std::cout << "count " << val << std::endl;

   return 0;
}
