// SampleClasses.h can be built in ROOT by '.L SampleClasses.h+'
R__LOAD_LIBRARY(SampleClasses_h)
#include "SampleClasses.h"
#include <string>
#include <vector>
#include "TClonesArray.h"

void generateTreeVectorClass(std::string const &name, std::string const &dir = ".", Int_t splitlevel = 0) {
   TFile f((dir + "/" + name + ".root").c_str(), "RECREATE"); // Create file
   TTree tree(name.c_str(), "Tree with a vector containing objects"); // Create tree

   // Leaf variables
   std::vector<Particle> vp;
   Particle p;

   // Create branch
   tree.Branch("vp", &vp, 32000, splitlevel);

   // Fill tree
   for (Int_t j = 1; j < 11; ++j) {
      vp.clear();

      for (Int_t k = 0; k < 5; ++k) {
         p.fPosX = 100 * j + 10 + k;
         p.fPosY = 100 * j + 20 + k;
         p.fPosZ = 100 * j + 30 + k;
         vp.push_back(p);
      }

      tree.Fill();
   }

   f.Write(); f.Close(); // Write tree to file
}

void generateTreeClassWithArray(std::string const &name, std::string const &dir = ".") {
   TFile f((dir + "/" + name + ".root").c_str(), "RECREATE"); // Create file
   TTree tree(name.c_str(), "Tree with a class containing an array"); // Create tree

   ClassWithArray *classWithArray = new ClassWithArray(); // One instance to fill the tree

   // Create branch for ClassWithArray
   tree.Branch("ClassWithArray_branch", &classWithArray);

   // Fill tree
   for (Int_t i = 1; i < 11; ++i) {
      for(Int_t j = 0; j < 10; ++j)
         classWithArray->arr[j] = 100 * i + j;

      tree.Fill();
   }

   f.Write(); f.Close(); // Write tree to file
}

void generateTreeClassWithVector(std::string const &name, std::string const &dir = ".") {
   TFile f((dir + "/" + name + ".root").c_str(), "RECREATE"); // Create file
   TTree tree(name.c_str(), "Tree with a class containing a vector"); // Create tree

   ClassWithVector *classWithVector = new ClassWithVector(); // One instance to fill the tree

   // Create branch for ClassWithVector
   tree.Branch("ClassWithVector_branch", &classWithVector);

   // Fill tree
   for (Int_t i = 1; i < 11; ++i) {
      classWithVector->vec.clear();
      classWithVector->vecBool.clear();
      for(Int_t j = 0; j < 5; ++j) {
         classWithVector->vec.push_back(100 * i + j);
         classWithVector->vecBool.push_back(j % 2);
      }

      tree.Fill();
   }

   f.Write(); f.Close(); // Write tree to file
}

void generateTreeClassWithClones(std::string const &name, std::string const &dir = ".") {
   TFile f((dir + "/" + name + ".root").c_str(), "RECREATE"); // Create file
   TTree tree(name.c_str(), "Tree with a class containing a TClonesArray"); // Create tree

   ClassWithClones *classWithClones = new ClassWithClones(); // One instance to fill the tree
   classWithClones->arr.SetOwner(kTRUE);

   // Create branch for ClassWithClones
   tree.Branch("ClassWithClones_branch", &classWithClones, 32000, 99);

   // Fill tree
   for (Int_t i = 1; i < 11; ++i) {
      classWithClones->arr.Clear();
      for(Int_t j = 0; j < 5; ++j) {
         new (classWithClones->arr[j]) Particle();
         Particle *p = (Particle*)classWithClones->arr[j];
         p->fPosX = 100 * i + 10 + j;
         p->fPosY = 100 * i + 20 + j;
         p->fPosZ = 100 * i + 30 + j;
      }
      tree.Fill();
   }

   f.Write(); f.Close(); // Write tree to file
}

void generateTreeNestedVector(std::string const &name, std::string const &dir = ".") {
   TFile f((dir + "/" + name + ".root").c_str(), "RECREATE"); // Create file
   TTree tree(name.c_str(), "Tree with a vector of a class containing a vector"); // Create tree

   std::vector<EventData> vec;

   // Create branch for ClassWithVector
   tree.Branch("vec_branch", &vec, 32000, 99);

   // Fill tree
   for (Int_t i = 0; i < 5; ++i) {
      vec.clear();
      for(Int_t j = 0; j <= i + 1; ++j) {
         EventData ed;
         Particle p;
         for (Int_t ip = 0; ip < 3; ++ip) {
            p.fPosX = i * 1000 + j * 100 + ip * 10 + 1;
            p.fPosY = i * 1000 + j * 100 + ip * 10 + 2;
            p.fPosZ = i * 1000 + j * 100 + ip * 10 + 3;
            ed.AddParticle(p);
         }
         ed.SetSize();
         vec.push_back(ed);
      }
      tree.Fill();
   }

   f.Write(); f.Close(); // Write tree to file
}

void generateTreeNestedClones(std::string const &name, std::string const &dir = ".") {
   TFile f((dir + "/" + name + ".root").c_str(), "RECREATE"); // Create file
   TTree tree(name.c_str(), "Tree with a TClonesArray containing a class containing a TClonesArray"); // Create tree

   // Leaf variables
   TClonesArray arr("ClassWithClones", 5);
   arr.SetOwner(kTRUE);

   // Create branch
   tree.Branch("arr", &arr, 32000, 99);

   // Fill tree
   for (Int_t i = 0; i < 5; ++i) {
      for (Int_t j = 0; j < 5; ++j) {
         new (arr[j]) ClassWithClones();
         ClassWithClones *cwc = (ClassWithClones*)arr[j];
         for (Int_t k = 0; k < 5; ++k) {
            new (cwc->arr[k]) Particle();
            Particle *p = (Particle*)cwc->arr[k];
            p->fPosX = i * 1000 + j * 100 + k * 10 + 1;
            p->fPosY = i * 1000 + j * 100 + k * 10 + 2;
            p->fPosZ = i * 1000 + j * 100 + k * 10 + 3;
         }
      }
      tree.Fill();
      arr.Clear(); // FIXME: can I delete the array after the tree was filled?
   }

   f.Write(); f.Close(); // Write tree to file
}

void generateTreeNestedArray(std::string const &name, std::string const &dir = ".") {
   TFile f((dir + "/" + name + ".root").c_str(), "RECREATE"); // Create file
   TTree tree(name.c_str(), "Tree with an array containing a class containing an array"); // Create tree

   // Leaf variables
   std::vector<ClassWithArray> vec;

   // Each variable has a separate branch
   tree.Branch("outer_vector", &vec, 32000, 99);

   int x = 0;

   // Fill tree
   for (Int_t i = 0; i < 10; ++i) {
      vec.clear();
      for (Int_t j = 0; j < 10; ++j) {
         vec.push_back(ClassWithArray());
         for(Int_t k = 0; k < 10; ++k)
            vec[j].arr[k] = ++x;
      }
      tree.Fill();
   }

   f.Write(); f.Close(); // Write tree to file
}

// Generate trees
void generateAll() {
   std::string const dir = "./trees";
   generateTreeVectorClass("TreeVectorClass0", dir, 0);
   generateTreeVectorClass("TreeVectorClass2", dir, 2);
   generateTreeClassWithArray("TreeClassWithArray", dir);
   generateTreeClassWithVector("TreeClassWithVector", dir);
   generateTreeClassWithClones("TreeClassWithClones", dir);
   generateTreeNestedVector("TreeNestedVector", dir);
   generateTreeNestedClones("TreeNestedClones", dir);
   generateTreeNestedArray("TreeNestedArray", dir);
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
void runCollectionClasses(const std::string &srcdir = ".")
{
   // Loop through test trees
   std::vector<std::string> trees = {"TreeVectorClass0",
                                     "TreeVectorClass2",
                                     "TreeClassWithArray",
                                     "TreeClassWithVector",
                                     "TreeClassWithClones",
                                     //"TreeNestedVector", // Known failure
                                     //"TreeNestedClones", // Known failure
                                     //"TreeNestedArray"   // Known failure
                                     };
   for (std::string const &treeName : trees)
   {
      fprintf(stderr, "Testing tree %s\n", treeName.c_str());

      TFile f((srcdir + "/trees/" + treeName + ".root").c_str()); // Load file
      TTree *t = (TTree*) f.Get(treeName.c_str());         // Load tree
      gSystem->cd("./generated_selectors");               // Go to gen. folder
      t->MakeSelector();                                  // Generate selector
      gSystem->cd("..");                                  // Go back
      t->Process((srcdir + "/test_selectors/" + treeName + ".C").c_str()); // Run (pre-filled) selector
   }
}

