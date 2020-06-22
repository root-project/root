#include <ROOT/RDataFrame.hxx>
#include <ROOT/RNTupleDS.hxx>

#include <ROOT/RNTuple.hxx>
#include <ROOT/RNTupleModel.hxx>

#include <gtest/gtest.h>

using ROOT::Experimental::RNTupleDS;
using ROOT::Experimental::RNTupleReader;
using ROOT::Experimental::RNTupleWriter;
using ROOT::Experimental::RNTupleModel;

class RNTupleDSTest : public ::testing::Test {
protected:
   // member variables are accessed by TEST_F functions
   std::string fFileName = "RNTupleDS_test.root";
   std::string fNtplName = "ntuple";
   std::unique_ptr<RNTupleReader> fNTuple = nullptr;

   void SetUp() override {
      auto modelWrite = RNTupleModel::Create();
      auto wrPt = modelWrite->MakeField<float>("pt", 42.0);
      auto wrEnergy = modelWrite->MakeField<float>("energy", 7.0);
      auto wrTag = modelWrite->MakeField<std::string>("tag", "xyz");
      auto wrJets = modelWrite->MakeField<std::vector<float>>("jets");
      wrJets->push_back(1.0);
      wrJets->push_back(2.0);
      auto wrNnlo = modelWrite->MakeField<std::vector<std::vector<float>>>("nnlo");
      wrNnlo->push_back(std::vector<float>());
      wrNnlo->push_back(std::vector<float>{1.0});
      wrNnlo->push_back(std::vector<float>{1.0, 2.0, 4.0, 8.0});
      {
         auto ntuple = RNTupleWriter::Recreate(std::move(modelWrite), fNtplName, fFileName);
         ntuple->Fill();
      }
      fNTuple = RNTupleReader::Open(fNtplName, fFileName);
   }
   void TearDown() override {
      std::remove(fFileName.c_str());
   }
};

TEST_F(RNTupleDSTest, ColTypeNames)
{
   RNTupleDS tds(std::move(fNTuple));

   auto colNames = tds.GetColumnNames();
   ASSERT_EQ(colNames.size(), 5);

   EXPECT_TRUE(tds.HasColumn("pt"));
   EXPECT_TRUE(tds.HasColumn("energy"));
   EXPECT_FALSE(tds.HasColumn("Address"));

   EXPECT_STREQ("std::string", tds.GetTypeName("tag").c_str());
   EXPECT_STREQ("float", tds.GetTypeName("energy").c_str());
}
