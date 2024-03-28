#include "ntuple_test.hxx"

#include <ROOT/RNTupleProcessor.hxx>

using ROOT::Experimental::RNTupleModel;
using ROOT::Experimental::RNTupleProcessor;
using ROOT::Experimental::RNTupleWriter;

TEST(RNTupleProcessor, Basic)
{
   FileRaii fileGuard("test_ntuple_processor_basic.root");
   {
      auto model = RNTupleModel::Create();
      auto fldX = model->MakeField<float>("x");
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard.GetPath());

      for (unsigned i = 0; i < 10; ++i) {
         *fldX = static_cast<float>(i);
         ntuple->Fill();
      }
   }

   std::vector<RNTupleProcessor::RProcessorSpec> ntuples;
   try {
      RNTupleProcessor proc(ntuples);
      FAIL() << "creating a processor without at least one RNTuple should throw";
   } catch (const RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("at least one RNTuple must be provided"));
   }

   ntuples = {{"ntuple", fileGuard.GetPath()}};
   std::uint64_t nEntries = 0;

   for (const auto &entry : RNTupleProcessor(ntuples)) {
      auto x = entry.GetPtr<float>("x");
      EXPECT_EQ(static_cast<float>(nEntries), *x);
      ++nEntries;
   }
   EXPECT_EQ(nEntries, 10);
}

TEST(RNTupleProcessor, SimpleChain)
{
   FileRaii fileGuard1("test_ntuple_processor_simple_chain1.root");
   {
      auto model = RNTupleModel::Create();
      auto fldX = model->MakeField<float>("x");
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard1.GetPath());

      for (unsigned i = 0; i < 10; ++i) {
         *fldX = static_cast<float>(i);
         ntuple->Fill();
      }
   }
   FileRaii fileGuard2("test_ntuple_processor_simple_chain2.root");
   {
      auto model = RNTupleModel::Create();
      auto fldX = model->MakeField<float>("x");
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard2.GetPath());

      for (unsigned i = 10; i < 25; ++i) {
         *fldX = static_cast<float>(i);
         ntuple->Fill();
      }
   }

   std::vector<RNTupleProcessor::RProcessorSpec> ntuples = {{"ntuple", fileGuard1.GetPath()},
                                                            {"ntuple", fileGuard2.GetPath()}};
   std::uint64_t nEntries = 0;
   for (const auto &entry : RNTupleProcessor(ntuples)) {
      auto x = entry.GetPtr<float>("x");
      EXPECT_EQ(static_cast<float>(nEntries), *x);
      ++nEntries;
   }
   EXPECT_EQ(nEntries, 25);
}
