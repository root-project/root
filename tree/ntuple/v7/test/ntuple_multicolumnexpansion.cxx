#include <ROOT/RConfig.hxx>

#include "ntuple_test.hxx"

#include <algorithm>
#include <random>

TEST(RNTuple, MultiColumnExpansion)
{
   // Tests if on-disk columns that expand to multiple in-memory types are correctly handled
#ifdef R__USE_IMT
   IMTRAII _;
#endif
   FileRaii fileGuard("test_ntuple_multi_column_expansion_copy.root");

   constexpr int kNEvents = 1000;

   {
      auto model = RNTupleModel::Create();
      model->AddField(RFieldBase::Create("pt", "Double32_t").Unwrap());
      RNTupleWriteOptions options;
      options.SetInitialUnzippedPageSize(8);
      options.SetMaxUnzippedPageSize(32);
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath(), options);

      auto ptrPt = writer->GetModel().GetDefaultEntry().GetPtr<double>("pt");

      for (int i = 0; i < kNEvents; ++i) {
         *ptrPt = i;
         writer->Fill();
         if (i % 50 == 0)
            writer->CommitCluster();
      }
   }

   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
   auto viewPt = reader->GetView<double>("pt");
   auto viewPtAsFloat = reader->GetView<float>("pt");

   std::random_device rd;
   std::mt19937 gen(rd());
   std::vector<unsigned int> indexes;
   indexes.reserve(kNEvents);
   for (unsigned int i = 0; i < kNEvents; ++i)
      indexes.emplace_back(i);
   std::shuffle(indexes.begin(), indexes.end(), gen);

   std::bernoulli_distribution dist(0.5);
   for (auto idx : indexes) {
      if (dist(gen)) {
         EXPECT_DOUBLE_EQ(idx, viewPt(idx));
         EXPECT_DOUBLE_EQ(idx, viewPtAsFloat(idx));
      } else {
         EXPECT_DOUBLE_EQ(idx, viewPtAsFloat(idx));
         EXPECT_DOUBLE_EQ(idx, viewPt(idx));
      }
   }
}
