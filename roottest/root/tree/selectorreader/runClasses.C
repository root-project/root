// SampleClasses.h can be built in ROOT by '.L SampleClasses.h+'
R__LOAD_LIBRARY(SampleClasses_h)
#include "SampleClasses.h"
#include <string>
#include <vector>

void generateTreeClass(std::string const &name, std::string const &dir = ".", Int_t splitlevel = 0) {
   TFile f((dir + "/" + name + ".root").c_str(), "RECREATE"); // Create file
   TTree tree(name.c_str(), "Tree from a class"); // Create tree

   ClassC *classC = new ClassC(); // One instance to fill the tree

   // Create branch for ClassC
   tree.Branch("ClassC_branch", "ClassC", &classC, 32000, splitlevel);

   // Fill tree
   for (Int_t i = 1; i < 11; ++i) {
      classC->Set(100 + i, 200 + i);
      tree.Fill();
   }

   f.Write(); f.Close(); // Write tree to file
}

void generateTreeClassNested(std::string const &name, std::string const &dir = ".", Int_t splitlevel = 0) {
   TFile f((dir + "/" + name + ".root").c_str(), "RECREATE"); // Create file
   TTree tree(name.c_str(), "Tree with nested classes"); // Create tree

   ClassB *classB = new ClassB(); // One instance to fill the tree

   // Create branch for ClassB
   tree.Branch("ClassB_branch", "ClassB", &classB, 32000, splitlevel);

   // Fill tree
   for (Int_t i = 1; i < 11; ++i) {
      classB->Set(100 + i, 200 + i, 300 + i);
      tree.Fill();
   }

   f.Write(); f.Close(); // Write tree to file
}

void generateTreeEventTreeSimple(std::string const &name, std::string const &dir = ".", Int_t splitlevel = 0) {
   TFile f((dir + "/" + name + ".root").c_str(), "RECREATE"); // Create file
   TTree tree(name.c_str(), "Simplified version of the EventTree from the intro tutorial"); // Create tree

   EventData *event = new EventData();
   Particle p;

   tree.Branch("Event_branch", &event, 32000, splitlevel);

   for (Int_t i = 1; i < 11; ++i) {
      event->Clear("");

      int nParticles = 5;
      for (Int_t ip = 0; ip < nParticles; ++ip) {
         p.fPosX = i * 100 + 10 + ip;
         p.fPosY = i * 100 + 20 + ip;
         p.fPosZ = i * 100 + 30 + ip;
         event->AddParticle(p);
      }

      event->SetSize();

      tree.Fill();
   }

   f.Write(); f.Close(); // Write tree to file
}

void generateTreeDuplicateName(std::string const &name, std::string const &dir = ".") {
   TFile f((dir + "/" + name + ".root").c_str(), "RECREATE"); // Create file
   TTree tree(name.c_str(), "Tree from a class with duplicate names"); // Create tree

   ClassC *classC1 = new ClassC(); // One instance to fill the tree
   ClassC *classC2 = new ClassC(); // One instance to fill the tree

   // Create branch for ClassC
   tree.Branch("C1", "ClassC", &classC1, 32000, 99);
   tree.Branch("C2", "ClassC", &classC2, 32000, 99);

   // Fill tree
   for (Int_t i = 1; i < 11; ++i) {
      classC1->Set(100 + i, 200 + i);
      classC2->Set(300 + i, 400 + i);
      tree.Fill();
   }

   f.Write(); f.Close(); // Write tree to file
}

// Generate trees
void generateAll() {
   std::string const dir = "./trees";
   generateTreeClass("TreeClass0", dir, 0);
   generateTreeClass("TreeClass2", dir, 2);
   generateTreeClassNested("TreeClassNested0", dir, 0);
   generateTreeClassNested("TreeClassNested1", dir, 1);
   generateTreeClassNested("TreeClassNested2", dir, 2);
   generateTreeEventTreeSimple("TreeEventTreeSimple0", dir, 0);
   generateTreeEventTreeSimple("TreeEventTreeSimple1", dir, 1);
   generateTreeEventTreeSimple("TreeEventTreeSimple2", dir, 2);
   generateTreeDuplicateName("TreeDuplicateName", dir);
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
void runClasses(const std::string &srcdir = ".") {
   // Loop through test trees
   std::vector<std::string> trees = {"TreeClass0",
                                     "TreeClass2",
                                     "TreeClassNested0",
                                     "TreeClassNested1",
                                     "TreeClassNested2",
                                     "TreeEventTreeSimple0",
                                     "TreeEventTreeSimple1",
                                     "TreeEventTreeSimple2",
                                     "TreeDuplicateName"
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

