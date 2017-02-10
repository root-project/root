#include "aclicModelWrite.C"

/*
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
*/

bool checkStdArrays(edm2::A& a, edm2::A& ref) {
   return ref.a0 != a.a0 ||
          ref.a2 != a.a2 ||
          ref.a6 != a.a6 ||
          ref.a8 != a.a8;
}

int modelCheckValues() {

   // Let's load the dictionaries
   gSystem->Load("aclicModelWrite_C");

   TFile f1("model.root");
   edm2::A ref;
   auto ap = (edm2::A*) f1.Get("a");
   auto& a = *ap;
   // Check the std::arrays
   if (checkStdArrays(a, ref)) {
      std::cerr << "Comparison of std arrays stored row-wise failed!\n";
      return 1;
   }

   TFile f2("modelTree.root");
   TTreeReader myReader("mytree", &f2);
   TTreeReaderValue<edm2::A> arv(myReader, "a");
   while (myReader.Next()) {
      if (checkStdArrays(*arv,ref)) {
         std::cerr << "Comparison of std arrays stored column-wise failed!\n";
         return 1;
      }
   }
   return 0;

}
