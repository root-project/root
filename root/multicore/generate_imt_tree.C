///
/// Used to generate http://root.cern/files/ttree_read_imt.root
#include "TFile.h"
#include "TRandom.h"
#include "TTree.h"

#include <stdlib.h>
#include <iostream>
#include <string>
#include <vector>

class B {
private:
   Float_t f0;
   Float_t f1;
   Float_t f2;
   Float_t f3;
   Float_t f4;
   Float_t f5;
   Float_t f6;
   Float_t f7;
   Float_t f8;
   Float_t f9;
public:
   B() : f0(0.f), f1(1.f), f2(2.f), f3(3.f), f4(4.f), f5(5.f), f6(6.f), f7(7.f), f8(8.f), f9(9.f) {}
};

class A {
private:
   std::vector<Double_t> fV0;
   std::vector<Double_t> fV1;
   std::vector<Double_t> fV2;
   std::vector<Double_t> fV3;
   std::vector<Double_t> fV4;
   std::vector<Double_t> fV5;
   std::vector<Double_t> fV6;
   std::vector<Double_t> fV7;
   std::vector<Double_t> fV8;
   std::vector<Double_t> fV9;
   B                     fB0;
   B                     fB1;
   B                     fB2;
   B                     fB3;
   B                     fB4;
   B                     fB5;
   B                     fB6;
   B                     fB7;
   B                     fB8;
   B                     fB9;
public:
   A() : fV0(nelems), fV1(nelems), fV2(nelems), fV3(nelems), fV4(nelems), fV5(nelems), fV6(nelems), fV7(nelems), fV8(nelems), fV9(nelems) {}
   void Build() {
      TRandom rand;
      for (int i = 0; i < nelems; ++i) {
         fV0[i] = rand.Uniform();
         fV1[i] = rand.Uniform();
         fV2[i] = rand.Uniform();
         fV3[i] = rand.Uniform();
         fV4[i] = rand.Uniform();
         fV5[i] = rand.Uniform();
         fV6[i] = rand.Uniform();
         fV7[i] = rand.Uniform();
         fV8[i] = rand.Uniform();
         fV9[i] = rand.Uniform();
      }
   }
   static const int nelems;
};

const int A::nelems = 100;


void generate_imt_tree()
{
  // Create the file and the tree
  TFile hfile("ttree_read_imt.root", "RECREATE", "File for IMT test");
  TTree tree("TreeIMT", "TTree for IMT test");

  int nvbranches = 50, nabranches = 50;
  int nentries = 1000, nvelems = 100;
  std::vector<std::vector<Double_t>> vbranches(nvbranches);
  std::vector<A> abranches(nabranches);

  // Create the tree branches
  for (int i = 0; i < nvbranches; ++i) {
    vbranches[i] = std::vector<Double_t>(nvelems);

    std::string branchname("Vbranch");
    branchname += std::to_string(i);
    branchname += std::string("."); // make the top-level branch name be included in the sub-branch names

    tree.Branch(branchname.c_str(), &vbranches[i]);
  }
  for (int i = 0; i < nabranches; ++i) {
    std::string branchname("Abranch");
    branchname += std::to_string(i);
    branchname += std::string("."); // make the top-level branch name be included in the sub-branch names

    tree.Branch(branchname.c_str(), &abranches[i]);
  }

  // Fill the tree
  TRandom rand;
  for (int i = 0; i < nentries; i++) {
    for (int i = 0; i < nvbranches; ++i) {
      for (int j = 0; j < nvelems; ++j) {
        vbranches[i][j] = rand.Uniform();
      }
    }
    for (int i = 0; i < nabranches; ++i) {
      abranches[i].Build();
    }
    Int_t nb = tree.Fill();
  }

  // Write the file 
  hfile.Write();
  hfile.Close();
}

