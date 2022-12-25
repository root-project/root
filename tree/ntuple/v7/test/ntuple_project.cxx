#include "ntuple_test.hxx"

TEST(RNTupleProjection, Basics)
{
   FileRaii fileGuard("test_ntuple_projection_basics.root");

   auto model = RNTupleModel::Create();
   model->MakeField<float>("met", 42.0);

   auto f1 = RFieldBase::Create("missingE", "float").Unwrap();
   model->AddProjectedField(std::move(f1), [](const std::string &) { return "met"; });

   {
      auto writer = RNTupleWriter::Recreate(std::move(model), "A", fileGuard.GetPath());
      writer->Fill();
   }

   auto reader = RNTupleReader::Open("A", fileGuard.GetPath());
   auto viewMissingE = reader->GetView<float>("missingE");
   EXPECT_FLOAT_EQ(42.0, viewMissingE(0));
}

TEST(RNTupleProjection, CatchReaderWithProjectedFields)
{
   FileRaii fileGuard("test_ntuple_projection_catch_reader_with_projected_fields.root");

   auto model = RNTupleModel::Create();
   model->MakeField<float>("met", 42.0);

   auto f1 = RFieldBase::Create("missingE", "float").Unwrap();
   model->AddProjectedField(std::move(f1), [](const std::string &) { return "met"; });

   auto modelRead = model->Clone();

   {
      auto writer = RNTupleWriter::Recreate(std::move(model), "A", fileGuard.GetPath());
   }

   try {
      auto reader = RNTupleReader::Open(std::move(modelRead), "A", fileGuard.GetPath());
      FAIL() << "creating a reader with a model with projected fields should throw";
   } catch (const RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("model has projected fields"));
   }
}
