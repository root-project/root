#include <ROOT/RDataFrame.hxx>
#include <ROOT/RNTupleDS.hxx>

#include <ROOT/RNTuple.hxx>
#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RPageStorage.hxx>

#include <gtest/gtest.h>

using ROOT::Experimental::RNTupleDS;
using ROOT::Experimental::RNTupleWriter;
using ROOT::Experimental::RNTupleModel;
using ROOT::Experimental::Detail::RPageSource;

class RNTupleDSTest : public ::testing::Test {
protected:
   std::string fFileName = "RNTupleDS_test.root";
   std::string fNtplName = "ntuple";
   std::unique_ptr<RPageSource> fPageSource;

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
      fPageSource = RPageSource::Create(fNtplName, fFileName);
   }

   void TearDown() override {
      std::remove(fFileName.c_str());
   }
};

TEST_F(RNTupleDSTest, ColTypeNames)
{
   RNTupleDS tds(std::move(fPageSource));

   auto colNames = tds.GetColumnNames();
   ASSERT_EQ(7, colNames.size());

   EXPECT_TRUE(tds.HasColumn("pt"));
   EXPECT_TRUE(tds.HasColumn("energy"));
   EXPECT_TRUE(tds.HasColumn("R_rdf_sizeof_nnlo"));
   EXPECT_FALSE(tds.HasColumn("Address"));

   EXPECT_STREQ("std::string", tds.GetTypeName("tag").c_str());
   EXPECT_STREQ("float", tds.GetTypeName("energy").c_str());
   EXPECT_STREQ("std::size_t", tds.GetTypeName("R_rdf_sizeof_jets").c_str());
}


TEST_F(RNTupleDSTest, CardinalityColumn)
{
   auto df = ROOT::Experimental::MakeNTupleDataFrame(fNtplName, fFileName);

   // Check that the special column #<collection> works without jitting...
   auto identity = [](std::size_t sz) { return sz; };
   auto max_njets = df.Define("njets", identity, {"R_rdf_sizeof_jets"}).Max<std::size_t>("njets");
   auto max_njets2 = df.Max<std::size_t>("#jets");
   EXPECT_EQ(*max_njets, *max_njets2);
   EXPECT_EQ(*max_njets, 2);

   // ...and with jitting
   auto max_njets_jitted = df.Define("njets", "R_rdf_sizeof_jets").Max<std::size_t>("njets");
   auto max_njets_jitted2 = df.Define("njets", "#jets").Max<std::size_t>("njets");
   auto max_njets_jitted3 = df.Max("#jets");
   EXPECT_EQ(*max_njets_jitted, *max_njets_jitted2);
   EXPECT_EQ(*max_njets_jitted3, *max_njets_jitted2);
   EXPECT_EQ(2, *max_njets_jitted);
}

void ReadTest(const std::string &name, const std::string &fname) {
   auto df = ROOT::Experimental::MakeNTupleDataFrame(name, fname);

   auto count = df.Count();
   auto sumpt = df.Sum<float>("pt");
   auto tag = df.Take<std::string>("tag");
   auto njets = df.Take<ROOT::Experimental::ClusterSize_t::ValueType>("R_rdf_sizeof_jets");
   auto sumjets = df.Sum<ROOT::RVec<float>>("jets");
   auto sumnnlosize = df.Sum<ROOT::RVec<ROOT::Experimental::ClusterSize_t::ValueType>>("R_rdf_sizeof_nnlo");
   auto sumvec = [](float red, const ROOT::RVec<ROOT::RVec<float>> &nnlo) {
      auto sum = 0.f;
      for (auto &v : nnlo)
         for (auto e : v)
            sum += e;
      return red + sum;
   };
   auto sumnnlo = df.Aggregate(sumvec, std::plus<float>{}, "nnlo", 0.f);

   EXPECT_EQ(1ull, count.GetValue());
   EXPECT_DOUBLE_EQ(42.f, sumpt.GetValue());
   EXPECT_EQ(1ull, tag.GetValue().size());
   EXPECT_EQ(std::string("xyz"), tag.GetValue()[0]);
   EXPECT_EQ(1ull, njets.GetValue().size());
   EXPECT_EQ(2u, njets.GetValue()[0]);
   EXPECT_EQ(3.f, sumjets.GetValue());
   EXPECT_EQ(16.f, sumnnlo.GetValue());
   EXPECT_EQ(5u, sumnnlosize.GetValue());
}

TEST_F(RNTupleDSTest, Read)
{
   ReadTest(fNtplName, fFileName);
}

struct IMTRAII {
   IMTRAII() { ROOT::EnableImplicitMT(); }
   ~IMTRAII() { ROOT::DisableImplicitMT(); }
};

TEST_F(RNTupleDSTest, ReadMT)
{
   IMTRAII _;

   ReadTest(fNtplName, fFileName);
}
