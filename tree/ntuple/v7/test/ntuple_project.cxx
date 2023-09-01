#include "ntuple_test.hxx"

TEST(RNTupleProjection, Basics)
{
   FileRaii fileGuard("test_ntuple_projection_basics.root");

   auto model = RNTupleModel::Create();
   model->MakeField<float>("met", 42.0);
   auto fvec = model->MakeField<std::vector<float>>("vec");
   fvec->emplace_back(1.0);
   fvec->emplace_back(2.0);

   auto f1 = RFieldBase::Create("missingE", "float").Unwrap();
   model->AddProjectedField(std::move(f1), [](const std::string &) { return "met"; });
   auto f2 = RFieldBase::Create("aliasVec", "std::vector<float>").Unwrap();
   model->AddProjectedField(std::move(f2), [](const std::string &fieldName) {
      if (fieldName == "aliasVec")
         return "vec";
      else
         return "vec._0";
   });
   auto f3 = RFieldBase::Create("vecSize", "ROOT::Experimental::RNTupleCardinality<std::uint64_t>").Unwrap();
   model->AddProjectedField(std::move(f3), [](const std::string &) { return "vec"; });

   {
      auto writer = RNTupleWriter::Recreate(std::move(model), "A", fileGuard.GetPath());
      writer->Fill();
   }

   auto reader = RNTupleReader::Open("A", fileGuard.GetPath());
   auto viewMissingE = reader->GetView<float>("missingE");
   auto viewAliasVec = reader->GetView<std::vector<float>>("aliasVec");
   auto viewVecSize = reader->GetView<ROOT::Experimental::RNTupleCardinality<std::uint64_t>>("vecSize");
   EXPECT_FLOAT_EQ(42.0, viewMissingE(0));
   EXPECT_EQ(2U, viewAliasVec(0).size());
   EXPECT_FLOAT_EQ(1.0, viewAliasVec(0).at(0));
   EXPECT_FLOAT_EQ(2.0, viewAliasVec(0).at(1));
   EXPECT_EQ(2U, viewVecSize(0));
}

TEST(RNTupleProjection, CatchInvalidMappings)
{
   FileRaii fileGuard("test_ntuple_projection_catch_invalid_mappings.root");

   auto model = RNTupleModel::Create();
   model->MakeField<float>("met", 42.0);
   model->MakeField<std::vector<float>>("vec");
   model->MakeField<std::variant<int, float>>("variant");
   model->MakeField<std::vector<std::vector<float>>>("nnlo");

   auto f1 = RFieldBase::Create("fail", "float").Unwrap();
   try {
      model->AddProjectedField(std::move(f1), [](const std::string &) { return "na"; }).ThrowOnError();
      FAIL() << "mapping to unknown field should throw";
   } catch (const RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("no such field"));
   }

   auto f2 = RFieldBase::Create("fail", "std::vector<float>").Unwrap();
   try {
      model
         ->AddProjectedField(std::move(f2),
                             [](const std::string &name) {
                                if (name == "fail")
                                   return "vec";
                                else
                                   return "na";
                             })
         .ThrowOnError();
      FAIL() << "mapping to unknown field should throw";
   } catch (const RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("no such field"));
   }

   auto f3 = RFieldBase::Create("fail", "std::vector<float>").Unwrap();
   try {
      model->AddProjectedField(std::move(f3), [](const std::string &) { return "met"; }).ThrowOnError();
      FAIL() << "mapping with structural mismatch should throw";
   } catch (const RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("field mapping structural mismatch"));
   }

   auto f4 = RFieldBase::Create("fail", "int").Unwrap();
   try {
      model->AddProjectedField(std::move(f4), [](const std::string &) { return "met"; }).ThrowOnError();
      FAIL() << "mapping without matching type should throw";
   } catch (const RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("field mapping type mismatch"));
   }

   auto f5 = RFieldBase::Create("fail", "std::variant<int, float>").Unwrap();
   try {
      model
         ->AddProjectedField(std::move(f5),
                             [](const std::string &fieldName) {
                                if (fieldName == "fail")
                                   return "variant";
                                if (fieldName == "fail._0")
                                   return "variant._0";
                                return "variant._1";
                             })
         .ThrowOnError();
      FAIL() << "mapping of variant should throw";
   } catch (const RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("unsupported field mapping "));
   }

   auto f6 = RFieldBase::Create("fail", "std::vector<float>").Unwrap();
   try {
      model
         ->AddProjectedField(std::move(f6),
                             [](const std::string &fieldName) {
                                if (fieldName == "fail")
                                   return "nnlo._0";
                                return "nnlo._0._0";
                             })
         .ThrowOnError();
      FAIL() << "mapping scrambling the source structure should throw";
   } catch (const RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("field mapping structure mismatch"));
   }
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
