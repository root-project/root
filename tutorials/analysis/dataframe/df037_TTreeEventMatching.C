/// \file
/// \ingroup tutorial_dataframe
/// \notebook -nodraw
///
/// This example shows processing of a TTree-based dataset with horizontal
/// concatenations (friends) and event matching (based on TTreeIndex). In case
/// the current event being processed does not match one (or more) of the friend
/// datasets, one can use the FilterAvailable and DefaultValueFor functionalities
/// to act upon the situation.
///
/// \macro_code
/// \macro_output
///
/// \date September 2024
/// \author Vincenzo Eduardo Padulano (CERN)
#include <TChain.h>
#include <TFile.h>
#include <TTree.h>
#include <TTreeIndex.h>

#include <ROOT/RDataFrame.hxx>

#include <iostream>
#include <numeric>

// A helper class to create the dataset for the tutorial below.
struct Dataset {

   constexpr static auto fMainFile{"df037_TTreeEventMatching_C_main.root"};
   constexpr static auto fAuxFile1{"df037_TTreeEventMatching_C_aux_1.root"};
   constexpr static auto fAuxFile2{"df037_TTreeEventMatching_C_aux_2.root"};
   constexpr static auto fMainTreeName{"events"};
   constexpr static auto fAuxTreeName1{"auxdata_1"};
   constexpr static auto fAuxTreeName2{"auxdata_2"};

   Dataset()
   {
      {
         TFile f(fMainFile, "RECREATE");
         TTree mainTree(fMainTreeName, fMainTreeName);
         int idx;
         int x;
         mainTree.Branch("idx", &idx, "idx/I");
         mainTree.Branch("x", &x, "x/I");

         idx = 1;
         x = 1;
         mainTree.Fill();
         idx = 2;
         x = 2;
         mainTree.Fill();
         idx = 3;
         x = 3;
         mainTree.Fill();

         mainTree.Write();
      }
      {
         // The first auxiliary file has matching indices 1 and 2, but not 3
         TFile f(fAuxFile1, "RECREATE");
         TTree auxTree(fAuxTreeName1, fAuxTreeName1);
         int y;
         int idx;
         auxTree.Branch("idx", &idx, "idx/I");
         auxTree.Branch("y", &y, "y/I");

         idx = 1;
         y = 4;
         auxTree.Fill();
         idx = 2;
         y = 5;
         auxTree.Fill();

         auxTree.Write();
      }
      {
         // The second auxiliary file has matching indices 1 and 3, but not 2
         TFile f(fAuxFile2, "RECREATE");
         TTree auxTree(fAuxTreeName2, fAuxTreeName2);
         int z;
         int idx;
         auxTree.Branch("idx", &idx, "idx/I");
         auxTree.Branch("z", &z, "z/I");

         idx = 1;
         z = 6;
         auxTree.Fill();
         idx = 3;
         z = 7;
         auxTree.Fill();

         auxTree.Write();
      }
   }

   ~Dataset()
   {
      std::remove(fMainFile);
      std::remove(fAuxFile1);
      std::remove(fAuxFile2);
   }
};

void df037_TTreeEventMatching()
{
   // Create the dataset: one main TTree and two auxiliary. The 'idx' branch
   // is used as the index to match events between the trees.
   // - The main tree has 3 entries, with 'idx' values (1, 2, 3).
   // - The first auxiliary tree has 2 entries, with 'idx' values (1, 2).
   // - The second auxiliary tree has 2 entries, with 'idx' values (1, 3).
   // The two auxiliary trees are concatenated horizontally with the main one.
   Dataset dataset{};
   TChain mainChain{dataset.fMainTreeName};
   mainChain.Add(dataset.fMainFile);

   TChain auxChain1(dataset.fAuxTreeName1);
   auxChain1.Add(dataset.fAuxFile1);
   auxChain1.BuildIndex("idx");

   TChain auxChain2(dataset.fAuxTreeName2);
   auxChain2.Add(dataset.fAuxFile2);
   auxChain2.BuildIndex("idx");

   mainChain.AddFriend(&auxChain1);
   mainChain.AddFriend(&auxChain2);

   // Create an RDataFrame to process the input dataset. The DefaultValueFor and
   // FilterAvailable functionalities can be used to decide what to do for
   // the events that do not match entirely according to the index column 'idx'
   ROOT::RDataFrame df{mainChain};

   const std::string auxTree1ColIdx = std::string(dataset.fAuxTreeName1) + ".idx";
   const std::string auxTree1ColY = std::string(dataset.fAuxTreeName1) + ".y";
   const std::string auxTree2ColIdx = std::string(dataset.fAuxTreeName2) + ".idx";
   const std::string auxTree2ColZ = std::string(dataset.fAuxTreeName2) + ".z";

   constexpr static auto defaultValue = std::numeric_limits<int>::min();

   // Example 1: provide default values for all columns in case there was no
   // match
   auto display1 = df.DefaultValueFor(auxTree1ColIdx, defaultValue)
                      .DefaultValueFor(auxTree1ColY, defaultValue)
                      .DefaultValueFor(auxTree2ColIdx, defaultValue)
                      .DefaultValueFor(auxTree2ColZ, defaultValue)
                      .Display<int, int, int, int, int, int>(
                         {"idx", auxTree1ColIdx, auxTree2ColIdx, "x", auxTree1ColY, auxTree2ColZ});

   // Example 2: skip the entire entry when there was no match for a column
   // in the first auxiliary tree, but keep the entries when there is no match
   // in the second auxiliary tree and provide a default value for those
   auto display2 = df.DefaultValueFor(auxTree2ColIdx, defaultValue)
                      .DefaultValueFor(auxTree2ColZ, defaultValue)
                      .FilterAvailable(auxTree1ColY)
                      .Display<int, int, int, int, int, int>(
                         {"idx", auxTree1ColIdx, auxTree2ColIdx, "x", auxTree1ColY, auxTree2ColZ});

   // Example 3: Keep entries from the main tree for which there is no
   // corresponding match in entries of the first auxiliary tree
   auto display3 = df.FilterMissing(auxTree1ColIdx).Display<int, int>({"idx", "x"});

   std::cout << "Example 1: provide default values for all columns\n";
   display1->Print();
   std::cout << "Example 2: skip the entry only when the first auxiliary tree does not match\n";
   display2->Print();
   std::cout << "Example 3: keep entries from the main tree for which there is no match in the auxiliary tree\n";
   display3->Print();
}
