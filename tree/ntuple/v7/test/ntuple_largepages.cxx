#include <ROOT/RConfig.hxx>

#include "ntuple_test.hxx"

#include <TROOT.h>

#include <algorithm>
#include <random>

TEST(RNTuple, LargePages)
{
   FileRaii fileGuard("test_ntuple_large_pages_copy.root");

   for (const auto useBufferedWrite : {true, false}) {
      {
         auto model = RNTupleModel::Create();
         auto fldRnd = model->MakeField<std::uint32_t>("rnd");
         RNTupleWriteOptions options;
         // Larger than the 16MB compression block limit
         options.SetMaxUnzippedPageSize(32 * 1024 * 1024);
         options.SetUseBufferedWrite(useBufferedWrite);
         auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath(), options);

         std::mt19937 gen;
         std::uniform_int_distribution<std::uint32_t> distrib;
         for (int i = 0; i < 25 * 1000 * 1000; ++i) { // 100 MB of int data
            *fldRnd = distrib(gen);
            writer->Fill();
         }
         writer.reset();
      }

      auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
      const auto &desc = reader->GetDescriptor();
      const auto rndColId = desc.FindPhysicalColumnId(desc.FindFieldId("rnd"), 0, 0);
      const auto &clusterDesc = desc.GetClusterDescriptor(desc.FindClusterId(rndColId, 0));
      EXPECT_GT(clusterDesc.GetPageRange(rndColId).Find(0).GetLocator().GetNBytesOnStorage(), kMAXZIPBUF);

      auto viewRnd = reader->GetView<std::uint32_t>("rnd");
      std::mt19937 gen;
      std::uniform_int_distribution<std::uint32_t> distrib;
      for (const auto i : reader->GetEntryRange()) {
         EXPECT_EQ(distrib(gen), viewRnd(i));
      }
   }
}
