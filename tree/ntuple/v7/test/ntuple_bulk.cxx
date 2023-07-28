#include "ntuple_test.hxx"

TEST(RNTupleBulk, Basics)
{
   FileRaii fileGuard("test_ntuple_bulk_basics.root");
   {
      auto model = RNTupleModel::Create();
      auto fldInt = model->MakeField<int>("int");
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
      for (int i = 0; i < 20; ++i) {
         *fldInt = i;
         writer->Fill();
      }
   }

   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
   // TODO(jblomer): find a better way to expose the GenerateBulk method of the target field
   auto fieldZero = reader->GetModel()->GetFieldZero();
   std::unique_ptr<RFieldBase::RBulk> bulk;
   for (auto &f : *fieldZero) {
      if (f.GetName() != "int")
         continue;
      bulk = std::make_unique<RFieldBase::RBulk>(f.GenerateBulk());
   }

   auto mask = std::make_unique<bool[]>(10);
   memset(mask.get(), 1, 10);
   auto intArr = static_cast<int *>(bulk->ReadBulk(RClusterIndex(0, 0), mask.get(), 10));
   for (int i = 0; i < 10; ++i) {
      EXPECT_EQ(i, intArr[i]);
   }
}
