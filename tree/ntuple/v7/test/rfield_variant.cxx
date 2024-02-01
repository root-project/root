#include "ntuple_test.hxx"

TEST(RNTuple, Variant)
{
   FileRaii fileGuard("test_ntuple_variant.root");

   auto modelWrite = RNTupleModel::Create();
   auto wrVariant = modelWrite->MakeField<std::variant<double, int>>("variant");
   *wrVariant = 2.0;

   modelWrite->Freeze();
   auto modelRead = std::unique_ptr<RNTupleModel>(modelWrite->Clone());

   {
      RNTupleWriter ntuple(std::move(modelWrite),
         std::make_unique<RPageSinkFile>("myNTuple", fileGuard.GetPath(), RNTupleWriteOptions()));
      ntuple.Fill();
      ntuple.CommitCluster();
      *wrVariant = 4;
      ntuple.Fill();
      *wrVariant = 8.0;
      ntuple.Fill();
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
