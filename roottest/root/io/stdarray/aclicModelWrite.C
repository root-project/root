#ifndef ROOTTTEST_ACLICMODELWRITE
#define ROOTTTEST_ACLICMODELWRITE
#include "TFile.h"
#include "TTree.h"
#include "infoDumper.h"
#include <array>

namespace edm2 {

   class B{};

   class A{
   public:
      std::array<int,3> a0 {{3,6,9}};
      int a1[3] = {3,6,9};

      std::array<std::array<int,3>,3> a2 {{ {{1,2,3}},{{1,2,3}},{{1,2,3}} }};
      int a3[3][3] = {{1,2,3},{1,2,3},{1,2,3}};

      std::array<B,42> a4;
      B a5[42];

      std::array<float,3> a6 {{3,6,9}};
      float a7[3] = {3,6,9};

      std::array<std::array<float,3>,3> a8 {{ {{1,2,3}},{{1,2,3}},{{1,2,3}} }};
      float a9[3][3] = {{1,2,3},{1,2,3},{1,2,3}};
   };

}

void writeTree(const char* rootfilename){
   auto f = TFile::Open(rootfilename,"RECREATE");
   edm2::A a;
   TTree t("mytree","thetree");
   t.Branch("a", &a);
   t.Fill();
   t.Write();
   f->Close();

}

void write(const char* rootfilename){
   auto f = TFile::Open(rootfilename,"RECREATE");
   edm2::A a;
   f->WriteObject(&a,"a");
   edm2::B b;
   f->WriteObject(&b,"b");
   f->Close();
}
int aclicModelWrite() {
   write("model.xml");
   write("model.root");
   writeTree("modelTree.root");
   return 0;
}
#endif
