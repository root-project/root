#include <TFile.h>
#include <TTree.h>
#include <ROOT/RDataFrame.hxx>
#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleWriter.hxx>

#include <gtest/gtest.h>

class UnifiedConstructor : public ::testing::Test {
protected:
   std::vector<std::string> fFileNames{"dataframe_unified_constructor_0.root", "dataframe_unified_constructor_1.root"};
   std::string fNTupleName = "ntuple";
   std::string fTTreeName = "tree";

   void SetUp() override
   {
      for (std::size_t i = 0; i < fFileNames.size(); ++i) {
         const auto &fName = fFileNames[i];
         {
            auto modelWrite = ROOT::RNTupleModel::Create();
            *modelWrite->MakeField<float>("ntuple_pt") = 11.f + i;
            auto ntuple = ROOT::RNTupleWriter::Recreate(std::move(modelWrite), fNTupleName, fName);
            ntuple->Fill();
         }
         {
            TFile f{fName.c_str(), "UPDATE"};
            float ttree_pt{22.f};
            TTree t{fTTreeName.c_str(), ""};
            t.Branch("ttree_pt", &ttree_pt);
            t.Fill();
            f.WriteObject(&t, fTTreeName.c_str());
         }
      }
   }

   void TearDown() override
   {
      for (const auto &fName : fFileNames) {
         std::remove(fName.c_str());
      }
   }
};

TEST_F(UnifiedConstructor, FromTTree)
{
   ROOT::RDataFrame df{fTTreeName, fFileNames[0]};
   auto pt = df.Take<float>("ttree_pt");
   EXPECT_EQ(pt->size(), 1);
   EXPECT_EQ(pt->at(0), 22.f);
}

TEST_F(UnifiedConstructor, FromRNTuple)
{
   ROOT::RDataFrame df{fNTupleName, fFileNames[0]};
   auto pt = df.Take<float>("ntuple_pt");
   EXPECT_EQ(pt->size(), 1);
   EXPECT_EQ(pt->at(0), 11.f);
}

TEST_F(UnifiedConstructor, FromTTrees)
{
   ROOT::RDataFrame df{fTTreeName, fFileNames};
   auto pt = df.Take<float>("ttree_pt");
   EXPECT_EQ(pt->size(), 2);
   EXPECT_EQ(pt->at(0), 22.f);
}

TEST_F(UnifiedConstructor, FromRNTuples)
{
   ROOT::RDataFrame df{fNTupleName, fFileNames};
   auto pt = df.Take<float>("ntuple_pt");
   EXPECT_EQ(pt->size(), 2);
   EXPECT_EQ(pt->at(0), 11.f);
}

TEST_F(UnifiedConstructor, FromRNTupleGlob)
{
   ROOT::RDataFrame df{fNTupleName, "dataframe_unified_constructor_*.root"};
   EXPECT_EQ(*df.Take<float>("ntuple_pt"), (std::vector<float>{11.f, 12.f}));
}

TEST_F(UnifiedConstructor, FromRNTupleGlobsAndFileNames)
{
   const std::vector<std::string> inputs{"dataframe_unified_constructor_[1].root", fFileNames[0]};
   ROOT::RDataFrame df{fNTupleName, inputs};
   EXPECT_EQ(*df.Take<float>("ntuple_pt"), (std::vector<float>{12.f, 11.f}));
}

TEST_F(UnifiedConstructor, FromRNTupleUnmatchedGlob)
{
   const std::vector<std::string> inputs{fFileNames[0], "dataframe_unified_constructor_missing_*.root"};
   EXPECT_THROW((ROOT::RDataFrame{fNTupleName, inputs}), std::invalid_argument);
}
