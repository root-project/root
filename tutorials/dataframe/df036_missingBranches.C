/// \file
/// \ingroup tutorial_dataframe
/// \notebook -nodraw
///
/// This example shows how to process a dataset where entries might be
/// incomplete due to one or more missing branches in one or more of the files
/// in the dataset. It shows usage of the FilterAvailable and DefaultValueFor
/// RDataFrame functionalities to act upon the missing entries.
///
/// \macro_code
/// \macro_output
///
/// \date September 2024
/// \author Vincenzo Eduardo Padulano (CERN)
#include <ROOT/RDataFrame.hxx>
#include <TChain.h>
#include <TFile.h>
#include <TTree.h>

#include <iostream>
#include <numeric>

// A helper class to create the dataset for the tutorial below.
struct Dataset {

   constexpr static std::array<const char *, 3> fFileNames{"df036_missingBranches_C_file_1.root",
                                                           "df036_missingBranches_C_file_2.root",
                                                           "df036_missingBranches_C_file_3.root"};
   constexpr static std::array<const char *, 3> fTreeNames{"tree_1", "tree_2", "tree_3"};
   constexpr static auto fTreeEntries{5};

   Dataset()
   {
      {
         TFile f(fFileNames[0], "RECREATE");
         TTree t(fTreeNames[0], fTreeNames[0]);
         int x{};
         int y{};
         t.Branch("x", &x, "x/I");
         t.Branch("y", &y, "y/I");
         for (int i = 1; i <= fTreeEntries; i++) {
            x = i;
            y = 2 * i;
            t.Fill();
         }

         t.Write();
      }

      {
         TFile f(fFileNames[1], "RECREATE");
         TTree t(fTreeNames[1], fTreeNames[1]);
         int y{};
         t.Branch("y", &y, "y/I");
         for (int i = 1; i <= fTreeEntries; i++) {
            y = 3 * i;
            t.Fill();
         }

         t.Write();
      }

      {
         TFile f(fFileNames[2], "RECREATE");
         TTree t(fTreeNames[2], fTreeNames[2]);
         int x{};
         t.Branch("x", &x, "x/I");
         for (int i = 1; i <= fTreeEntries; i++) {
            x = 4 * i;
            t.Fill();
         }

         t.Write();
      }
   }

   ~Dataset()
   {
      for (auto &&fileName : fFileNames)
         std::remove(fileName);
   }
};

void df036_missingBranches()
{
   // Create the example dataset. Three files are created with one TTree each.
   // The first contains branches (x, y), the second only branch y, the third
   // only branch x.
   Dataset trees{};

   // The TChain will process the three files, encountering a different missing
   // branch when switching to the next tree
   TChain c{};
   for (auto i = 0; i < trees.fFileNames.size(); i++) {
      const auto fullPath = std::string(trees.fFileNames[i]) + "?#" + trees.fTreeNames[i];
      c.Add(fullPath.c_str());
   }

   ROOT::RDataFrame df{c};

   constexpr static auto defaultValue = std::numeric_limits<int>::min();

   // Example 1: provide a default value for all missing branches
   auto display1 = df.DefaultValueFor("x", defaultValue)
                      .DefaultValueFor("y", defaultValue)
                      .Display<int, int>({"x", "y"}, /*nRows*/ 15);

   // Example 2: provide a default value for branch y, but skip events where
   // branch x is missing
   auto display2 =
      df.DefaultValueFor("y", defaultValue).FilterAvailable("x").Display<int, int>({"x", "y"}, /*nRows*/ 15);

   // Example 3: only keep events where branch y is missing and display values for branch x
   auto display3 = df.FilterMissing("y").Display<int>({"x"}, /*nRows*/ 15);

   std::cout << "Example 1: provide a default value for all missing branches\n";
   display1->Print();

   std::cout << "Example 2: provide a default value for branch y, but skip events where branch x is missing\n";
   display2->Print();

   std::cout << "Example 3: only keep events where branch y is missing and display values for branch x\n";
   display3->Print();
}
