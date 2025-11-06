#include "ntuple_test.hxx"

#include <limits>

TEST(RNTuple, DISABLED_LargeVector)
{
   FileRaii fileGuard("test_ntuple_large_vector.root");

   // write out a vector too large for RVec
   {
      auto m = RNTupleModel::Create();
      auto vec = m->MakeField<std::vector<int8_t>>("v");
      auto writer = RNTupleWriter::Recreate(std::move(m), "r", fileGuard.GetPath());
      vec->push_back(1);
      writer->Fill();
      vec->resize(std::numeric_limits<std::int32_t>::max());
      writer->Fill();
      vec->push_back(2);
      writer->Fill();
      vec->clear();
      writer->Fill();
   }

   ROOT::RNTupleReadOptions options;
   options.SetClusterCache(ROOT::RNTupleReadOptions::EClusterCache::kOff);
   auto reader = RNTupleReader::Open("r", fileGuard.GetPath(), options);
   ASSERT_EQ(4u, reader->GetNEntries());

   auto viewRVec = reader->GetView<ROOT::RVec<int8_t>>("v");
   EXPECT_EQ(1u, viewRVec(0).size());
   EXPECT_EQ(1, viewRVec(0).at(0));
   const auto &v1 = viewRVec(1);
   EXPECT_EQ(std::numeric_limits<std::int32_t>::max(), v1.size());
   EXPECT_EQ(1, v1.at(0));
   EXPECT_EQ(0, v1.at(1000));
   EXPECT_THROW(viewRVec(2), ROOT::RException);
   EXPECT_TRUE(viewRVec(3).empty());

   auto viewVector = reader->GetView<std::vector<int8_t>>("v");
   const auto &v3 = viewVector(2);
   EXPECT_EQ(static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()) + 1, v3.size());
   EXPECT_EQ(1, v3.at(0));
   EXPECT_EQ(0, v3.at(1000));
   EXPECT_EQ(2, v3.at(std::numeric_limits<std::int32_t>::max()));
}
