// SampleClasses.h can be built in ROOT by '.L SampleClasses.h+'
R__LOAD_LIBRARY(SampleClasses_h)
#include "SampleClasses.h"
#include <string>

// Run all tests:
// - It is assumed that the tree 'TreeClassNested2.root' is already generated.
//   (If not, it can be generated with 'generateAll()' in 'testClasses.C'.
// - The test will loop through the possible options and generate the selectors
//   into the folder 'generated_selectors'.
// - The test will also try accessing the data in the tree. However, since the newly
//   generated NameOfSelector.C is empty, it uses a different .C file (located under
//   'test_selectors/NameOfSelector.C'), which has been already filled with code accessing
//   the data. (Regarding the header file, it needs no modification, so the newly
//   generated one is used.)
void runOptions(const std::string &srcdir = ".")
{
   // Loop through test trees
   std::string treeName = "TreeClassNested2";
   // List of options with their display names
   std::vector<std::string> options = {""      , "@"      , "fC" , "@fC"  , "@ClassB_branch;fC.fEv;@fC;@fC.fPx;noSuchBranch"};
   std::vector<std::string> names   = {"_empty", "_alltop", "_fC", "_atfC", "_complex"};

   for (Int_t i = 0; i < options.size(); ++i)
   {
      fprintf(stderr, "Testing option \"%s\"\n", options[i].c_str());

      std::string selectorName = treeName + names[i];

      TFile f((srcdir + "/trees/" + treeName + ".root").c_str()); // Load file
      TTree *t = (TTree*)f.Get(treeName.c_str());         // Load tree
      gSystem->cd("./generated_selectors");               // Go to gen. folder
      t->MakeSelector(selectorName.c_str(), options[i].c_str());    // Generate selector
      gSystem->cd("..");                                  // Go back
      t->Process((srcdir + "/test_selectors/" + selectorName + ".C").c_str()); // Run (pre-filled) selector
   }
}

