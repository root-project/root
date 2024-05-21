#include "ntuple_test.hxx"

#include <optional>

TEST(RNTuple, Variant)
{
   FileRaii fileGuard("test_ntuple_variant.root");

   auto modelWrite = RNTupleModel::Create();
   auto wrVariant = modelWrite->MakeField<std::variant<double, int>>("variant");
   *wrVariant = 2.0;

   modelWrite->Freeze();
   auto modelRead = std::unique_ptr<RNTupleModel>(modelWrite->Clone());

   {
      auto writer = RNTupleWriter::Recreate(std::move(modelWrite), "myNTuple", fileGuard.GetPath());
      writer->Fill();
      writer->CommitCluster();
      *wrVariant = 4;
      writer->Fill();
      *wrVariant = 8.0;
      writer->Fill();
   }
   auto rdVariant = modelRead->GetDefaultEntry().GetPtr<std::variant<double, int>>("variant").get();

   auto reader = RNTupleReader::Open(std::move(modelRead), "myNTuple", fileGuard.GetPath());
   EXPECT_EQ(3U, reader->GetNEntries());

   reader->LoadEntry(0);
   EXPECT_EQ(2.0, *std::get_if<double>(rdVariant));
   reader->LoadEntry(1);
   EXPECT_EQ(4, *std::get_if<int>(rdVariant));
   reader->LoadEntry(2);
   EXPECT_EQ(8.0, *std::get_if<double>(rdVariant));
}

TEST(RNTuple, VariantComplex)
{
   FileRaii fileGuard("test_ntuple_variant_complex.root");

   {
      auto model = RNTupleModel::Create();
      auto ptrVec = model->MakeField<std::vector<std::variant<std::optional<int>, float>>>("vvar");
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());

      for (int i = 0; i < 10; i++) {
         ptrVec->clear();

         for (int j = 0; j < 5; ++j) {
            std::variant<std::optional<int>, float> var(1.0);
            ptrVec->emplace_back(var);
         }

         writer->Fill();
      }
   }
}
