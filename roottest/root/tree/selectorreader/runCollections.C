// SampleClasses.h can be built in ROOT by '.L SampleClasses.h+'
R__LOAD_LIBRARY(SampleClasses_h)
#include "SampleClasses.h"
#include <string>
#include <vector>
#include <set>
#include <list>
#include "TObjArray.h"
#include "TClonesArray.h"

void generateTreeArray(std::string const &name, std::string const &dir = ".") {
   TFile f((dir + "/" + name + ".root").c_str(), "RECREATE"); // Create file
   TTree tree(name.c_str(), "Tree with a fixed size array"); // Create tree

   // Leaf variables
   Float_t arr[10];

   // Each variable has a separate branch
   tree.Branch("arr", arr, "arr[10]/F");

   // Fill tree
   for (Int_t j = 1; j < 11; ++j) {
      for(Int_t k = 0; k < 10; ++k)
         arr[k] = j * 100 + k;

      tree.Fill();
   }

   f.Write(); f.Close(); // Write tree to file
}

void generateTreeVector(std::string const &name, std::string const &dir = ".") {
   TFile f((dir + "/" + name + ".root").c_str(), "RECREATE"); // Create file
   TTree tree(name.c_str(), "Tree with vectors of built in types"); // Create tree

   // Leaf variables
   std::vector<Float_t> vpx;
   std::vector<Bool_t> vb;

   // Each variable has a separate branch
   tree.Branch("vpx", &vpx);
   tree.Branch("vb" , &vb );

   // Fill tree
   for (Int_t j = 1; j < 11; ++j) {
      vpx.clear();
      vb.clear();

      for (Int_t k = 0; k < j; ++k) {
         vpx.push_back(100 * j + k);
         vb.push_back(((k * j) % 2) == 0);
      }

      tree.Fill();
   }

   f.Write(); f.Close(); // Write tree to file
}

void generateTreeContainers(std::string const &name, std::string const &dir = ".") {
   TFile f((dir + "/" + name + ".root").c_str(), "RECREATE"); // Create file
   TTree tree(name.c_str(), "Tree with containers of built in types"); // Create tree

   // Leaf variables
   std::vector<Int_t> vi;
   std::set<Int_t> si;
   std::list<Int_t> li;

   // Each variable has a separate branch
   tree.Branch("vectorBranch", &vi);
   tree.Branch("setBranch", &si);
   tree.Branch("listBranch", &li);

   // Fill tree
   for (Int_t j = 1; j < 11; ++j) {
      vi.clear();
      si.clear();
      li.clear();

      for (Int_t k = 0; k < 10; ++k) {
         vi.push_back(100 * j + k + 10);
         si.insert(   100 * j + k + 20);
         li.push_back(100 * j + k + 30);
      }

      tree.Fill();
   }

   f.Write(); f.Close(); // Write tree to file
}

void generateTreeTObjArray(std::string const &name, std::string const &dir = ".") {
   TFile f((dir + "/" + name + ".root").c_str(), "RECREATE"); // Create file
   TTree tree(name.c_str(), "Tree with a TObjArray"); // Create tree

   // Leaf variables
   TObjArray arr;
   arr.SetOwner(kTRUE);

   // Create branch
   tree.Branch("arr", &arr);

   // Fill tree
   for (Int_t i = 1; i < 11; ++i) {

      for (Int_t j = 0; j < 5; ++j)
         arr.Add(new ClassC(i * 100 + j, j));

      tree.Fill();
      arr.Delete(); // FIXME: can I delete the array after the tree was filled?
   }

   f.Write(); f.Close(); // Write tree to file
}

void generateTreeTClonesArray(std::string const &name, std::string const &dir = ".", Int_t splitlevel = 0) {
   TFile f((dir + "/" + name + ".root").c_str(), "RECREATE"); // Create file
   TTree tree(name.c_str(), "Tree with a TClonesArray"); // Create tree

   // Leaf variables
   TClonesArray arr("ClassC", 5);
   arr.SetOwner(kTRUE);

   // Create branch
   tree.Branch("arr", &arr, 32000, splitlevel);

   // Fill tree
   for (Int_t i = 0; i < 20; ++i) {
      for (Int_t j = 0; j < 5; ++j)
         new (arr[j]) ClassC(i * 100 + j, j);

      tree.Fill();
      arr.Clear(); // FIXME: can I delete the array after the tree was filled?
   }

   f.Write(); f.Close(); // Write tree to file
}

// Generate trees
void generateAll() {
   std::string const dir = "./trees";
   generateTreeArray("TreeArray", dir);
   generateTreeVector("TreeVector", dir);
   generateTreeContainers("TreeContainers", dir);
   generateTreeTObjArray("TreeTObjArray", dir);
   generateTreeTClonesArray("TreeTClonesArray0", dir, 0);
   generateTreeTClonesArray("TreeTClonesArray2", dir, 2);
}

// Run all tests:
// - It is assumed that the trees are already generated (if not, call 'generateAll()')
// - The test will loop through the trees and generate the selector into the
//   folder 'generated_selectors' (NameOfTree.h and NameOfTree.C).
// - The test will also try accessing the data in the tree. However, since the newly
//   generated NameOfTree.C is empty, it uses a different .C file (located under
//   'test_selectors/NameOfTree.C'), which has been already filled with code accessing
//   the data. (Regarding the header file, it needs no modification, so the newly
//   generated one is used.)
void runCollections(const std::string &srcdir = ".")
{
   // Loop through test trees
   std::vector<std::string> trees = {"TreeArray",
                                     "TreeVector",
                                     "TreeContainers",
                                     //"TreeTObjArray", // Known failure
                                     //"TreeTClonesArray0", // Known failure
                                     "TreeTClonesArray2"
                                     };
   for (std::string const &treeName : trees)
   {
      fprintf(stderr, "Testing tree %s\n", treeName.c_str());

      TFile f((srcdir + "/trees/" + treeName + ".root").c_str()); // Load file
      TTree *t = (TTree*)f.Get(treeName.c_str());         // Load tree
      gSystem->cd("./generated_selectors");               // Go to gen. folder
      t->MakeSelector();                                  // Generate selector
      gSystem->cd("..");                                  // Go back
      t->Process((srcdir + "/test_selectors/" + treeName + ".C").c_str()); // Run (pre-filled) selector
   }
}

