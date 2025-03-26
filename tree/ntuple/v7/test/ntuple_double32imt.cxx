#include <ROOT/RConfig.hxx>

#include "ntuple_test.hxx"

#include <algorithm>

TEST(RNTuple, Double32IMT)
{
   // Tests if parallel decompression correctly compresses the on-disk float to an in-memory double
#ifdef R__USE_IMT
   IMTRAII _;
#endif
   FileRaii fileGuard("test_ntuple_double32_imt_copy.root");

   constexpr int kNEvents = 10;

   {
      auto model = RNTupleModel::Create();
      model->AddField(RFieldBase::Create("pt", "Double32_t").Unwrap());
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());

      auto ptrPt = writer->GetModel().GetDefaultEntry().GetPtr<double>("pt");

      for (int i = 0; i < kNEvents; ++i) {
         *ptrPt = i;
         writer->Fill();
      }
   }

   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
   auto viewPt = reader->GetView<double>("pt");
   for (int i = 0; i < kNEvents; ++i) {
      EXPECT_DOUBLE_EQ(i, viewPt(i));
   }
}
