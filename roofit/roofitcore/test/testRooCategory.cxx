// Tests for the RooCategory
// Author: Jonas Rembser, CERN  04/2021

#include <RooCategory.h>
#include <RooDataSet.h>
#include <RooGlobalFunc.h>

#include <TTree.h>

#include <gtest/gtest.h>

// GitHub issue 10278: RooDataSet incorrectly loads RooCategory values from TTree branch of type Short_t
TEST(RooCategory, CategoryDefineMultiState)
{
   TTree tree("test_tree", "Test tree");
   Short_t cat_in;
   tree.Branch("cat", &cat_in);

   cat_in = 2; // category B
   tree.Fill();

   RooCategory cat("cat", "Category", {{"B_cat", 2}, {"A_cat", 3}});
   RooDataSet data("data", "RooDataSet", RooArgSet(cat), RooFit::Import(tree));

   EXPECT_EQ(static_cast<RooCategory &>((*data.get(0))["cat"]).getCurrentIndex(), 2);
}
