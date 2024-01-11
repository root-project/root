#include <TFile.h>
#include <TTree.h>
#include <ROOT/RDataFrame.hxx>
#include <ROOT/RNTuple.hxx>
#include <ROOT/RNTupleModel.hxx>

#include <gtest/gtest.h>

using ROOT::Experimental::RNTupleModel;
using ROOT::Experimental::RNTupleWriter;

class UnifiedConstructor : public ::testing::Test {
protected:
   std::string fFileName = "dataframe_unified_constructor.root";
   std::string fNTupleName = "ntuple";
   std::string fTTreeName = "tree";

   void SetUp() override
   {
      {
         auto modelWrite = RNTupleModel::Create();
         auto pt = modelWrite->MakeField<float>("ntuple_pt", 11.f);
         auto ntuple = RNTupleWriter::Recreate(std::move(modelWrite), fNTupleName, fFileName);
         ntuple->Fill();
      }
      {
         TFile f{fFileName.c_str(), "UPDATE"};
         float ttree_pt{22.f};
         TTree t{fTTreeName.c_str(), ""};
         t.Branch("ttree_pt", &ttree_pt);
         t.Fill();
         f.WriteObject(&t, fTTreeName.c_str());
      }
   }

   void TearDown() override { std::remove(fFileName.c_str()); }
};

TEST_F(UnifiedConstructor, FromTTree)
{
   ROOT::RDataFrame df{fTTreeName, fFileName};
   auto pt = df.Take<float>("ttree_pt");
   EXPECT_EQ(pt->size(), 1);
   EXPECT_EQ(pt->at(0), 22.f);
}

TEST_F(UnifiedConstructor, FromRNTuple)
{
   ROOT::RDataFrame df{fNTupleName, fFileName};
   auto pt = df.Take<float>("ntuple_pt");
   EXPECT_EQ(pt->size(), 1);
   EXPECT_EQ(pt->at(0), 11.f);
}
