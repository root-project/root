#include <ROOT/RConfig.hxx>

#include "ntuple_test.hxx"

#include <TRandom3.h>

#include <algorithm>

// Stress test the asynchronous cluster pool by a deliberately unfavourable read pattern
TEST(RNTuple, RandomAccess)
{
#ifdef R__USE_IMT
   IMTRAII _;
#endif
   FileRaii fileGuard("test_ntuple_random_access.root");

   auto modelWrite = RNTupleModel::Create();
   auto wrValue = modelWrite->MakeField<std::int32_t>("value");

   constexpr unsigned int nEvents = 1000000;
   {
      RNTupleWriteOptions options;
      options.SetCompression(0);
      options.SetEnablePageChecksums(false);
      options.SetApproxZippedClusterSize(nEvents * sizeof(std::int32_t) / 10);
      auto ntuple = RNTupleWriter::Recreate(std::move(modelWrite), "myNTuple", fileGuard.GetPath(), options);
      for (unsigned int i = 0; i < nEvents; ++i) {
         *wrValue = i;
         ntuple->Fill();
      }
   }

   RNTupleReadOptions options;
   options.SetClusterCache(RNTupleReadOptions::EClusterCache::kOn);
   auto ntuple = RNTupleReader::Open("myNTuple", fileGuard.GetPath(), options);
   EXPECT_EQ(10, ntuple->GetDescriptor().GetNClusters());

   auto viewValue = ntuple->GetView<std::int32_t>("value");

   std::int64_t sum = 0;
   std::int64_t expected = 0;
   constexpr unsigned int nSamples = 50000;
   TRandom3 rnd(42);
   for (unsigned int i = 0; i < nSamples; ++i) {
      auto entryId = floor(rnd.Rndm() * (nEvents - 1));
      expected += entryId;
      sum += viewValue(entryId);
   }
   EXPECT_EQ(expected, sum);
}
