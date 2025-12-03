#include <string>
#include <vector>

void generateTree(std::string const &name, std::string const &dir = ".") {
   TFile f((dir + "/" + name + ".root").c_str(), "RECREATE"); // Create file
   TTree tree(name.c_str(), "Tree with basic leaf variables"); // Create tree

   // Leaf variables
   Float_t px, py, pz;
   Double_t random;
   Int_t ev;

   // Each variable has a separate branch
   tree.Branch("px",&px,"px/F");
   tree.Branch("py",&py,"py/F");
   tree.Branch("pz",&pz,"pz/F");
   tree.Branch("random",&random,"random/D");
   tree.Branch("ev",&ev,"ev/I");

   // Fill tree
   for (Int_t i = 0; i < 10; ++i) {
      px     = 100 + i;
      py     = 200 + i;
      pz     = 300 + i;
      random = 400 + i;
      ev     = 500 + i;
      tree.Fill();
   }

   f.Write(); f.Close(); // Write tree to file
}

// Simple structure to hold the data
struct simple_t {
   Float_t px, py, pz;
   Int_t ev;
};

simple_t simple; // An instance of the structure to generate the data

void generateTreeStruct(std::string const &name, std::string const &dir = ".") {
   TFile f((dir + "/" + name + ".root").c_str(), "RECREATE"); // Create file
   TTree tree(name.c_str(), "Tree from a C structure with basic variables"); // Create tree

   // A single branch contains all variables
   tree.Branch("simple", &simple, "px/F:py/F:pz/F:ev/I");

   // Fill tree
   for (Int_t i = 0; i < 10; ++i) {
      simple.px = 100 + i;
      simple.py = 200 + i;
      simple.pz = 300 + i;
      simple.ev = 500 + i;
      tree.Fill();
   }

   f.Write(); f.Close(); // Write tree to file
}

// Generate trees
void generateAll() {
   std::string const dir = "./trees";
   generateTree("Tree", dir);
   generateTreeStruct("TreeStruct", dir);
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
void runSimple(const std::string &srcdir = ".")
{
   // Loop through test trees
   std::vector<std::string> trees = {"Tree", "TreeStruct"};
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

