#include "ntuple_test.hxx"

TEST(RNTupleProjection, Basics)
{
   FileRaii fileGuard("test_ntuple_projection_basics.root");

   auto model = RNTupleModel::Create();
   *model->MakeField<float>("met") = 42.0;
   *model->MakeField<std::atomic<int>>("atomicNumber") = 7;
   auto fvec = model->MakeField<std::vector<float>>("vec");
   fvec->emplace_back(1.0);
   fvec->emplace_back(2.0);

   auto f1 = RFieldBase::Create("missingE", "float").Unwrap();
   model->AddProjectedField(std::move(f1), [](const std::string &) { return "met"; });
   auto f2 = RFieldBase::Create("number", "int").Unwrap();
   model->AddProjectedField(std::move(f2), [](const std::string &) { return "atomicNumber._0"; });
   auto f3 = RFieldBase::Create("aliasVec", "std::vector<float>").Unwrap();
   model->AddProjectedField(std::move(f3), [](const std::string &fieldName) {
      if (fieldName == "aliasVec")
         return "vec";
      else
         return "vec._0";
   });
   auto f4 = RFieldBase::Create("vecSize", "ROOT::RNTupleCardinality<std::uint64_t>").Unwrap();
   model->AddProjectedField(std::move(f4), [](const std::string &) { return "vec"; });

   {
      auto writer = RNTupleWriter::Recreate(std::move(model), "A", fileGuard.GetPath());
      writer->Fill();
   }

   auto reader = RNTupleReader::Open("A", fileGuard.GetPath());
   const auto &desc = reader->GetDescriptor();
   const auto metFieldId = desc.FindFieldId("met");
   const auto missingEFieldId = desc.FindFieldId("missingE");
   EXPECT_FALSE(desc.GetFieldDescriptor(metFieldId).IsProjectedField());
   EXPECT_TRUE(desc.GetFieldDescriptor(missingEFieldId).IsProjectedField());
   EXPECT_EQ(metFieldId, desc.GetFieldDescriptor(missingEFieldId).GetProjectionSourceId());
   auto viewMissingE = reader->GetView<float>("missingE");
   auto viewNumber = reader->GetView<int>("number");
   auto viewAliasVec = reader->GetView<std::vector<float>>("aliasVec");
   auto viewVecSize = reader->GetView<ROOT::RNTupleCardinality<std::uint64_t>>("vecSize");
   EXPECT_FLOAT_EQ(42.0, viewMissingE(0));
   EXPECT_EQ(7, viewNumber(0));
   EXPECT_EQ(2U, viewAliasVec(0).size());
   EXPECT_FLOAT_EQ(1.0, viewAliasVec(0).at(0));
   EXPECT_FLOAT_EQ(2.0, viewAliasVec(0).at(1));
   EXPECT_EQ(2U, viewVecSize(0));

   RNTupleDescriptor::RCreateModelOptions options;
   options.fReconstructProjections = true;
   auto reconstructedModel = reader->GetDescriptor().CreateModel(options);
   auto itrFields = reconstructedModel->GetConstFieldZero().cbegin();
   EXPECT_EQ("met", itrFields->GetQualifiedFieldName());
   EXPECT_EQ("atomicNumber", (++itrFields)->GetQualifiedFieldName());
   EXPECT_EQ("atomicNumber._0", (++itrFields)->GetQualifiedFieldName());
   EXPECT_EQ("vec", (++itrFields)->GetQualifiedFieldName());
   EXPECT_EQ("vec._0", (++itrFields)->GetQualifiedFieldName());
   EXPECT_EQ(reconstructedModel->GetConstFieldZero().cend(), ++itrFields);
   auto &projectedFields = ROOT::Experimental::Internal::GetProjectedFieldsOfModel(*reconstructedModel);
   auto itrProjectedFields = projectedFields.GetFieldZero().cbegin();
   EXPECT_EQ("missingE", itrProjectedFields->GetQualifiedFieldName());
   EXPECT_EQ("number", (++itrProjectedFields)->GetQualifiedFieldName());
   EXPECT_EQ("aliasVec", (++itrProjectedFields)->GetQualifiedFieldName());
   EXPECT_EQ("aliasVec._0", (++itrProjectedFields)->GetQualifiedFieldName());
   EXPECT_EQ("vecSize", (++itrProjectedFields)->GetQualifiedFieldName());
   EXPECT_EQ(projectedFields.GetFieldZero().cend(), ++itrProjectedFields);
}

TEST(RNTupleProjection, CatchInvalidMappings)
{
   FileRaii fileGuard("test_ntuple_projection_catch_invalid_mappings.root");

   auto model = RNTupleModel::Create();
   model->MakeField<float>("met");
   model->MakeField<std::vector<float>>("vec");
   model->MakeField<std::variant<int, float>>("variant");
   model->MakeField<std::vector<std::vector<float>>>("nnlo");
   model->MakeField<std::array<float, 3>>("lorentz");

   auto f1 = RFieldBase::Create("fail", "float").Unwrap();
   try {
      model->AddProjectedField(std::move(f1), [](const std::string &) { return "na"; }).ThrowOnError();
      FAIL() << "mapping to unknown field should throw";
   } catch (const ROOT::RException &err) {
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
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("no such field"));
   }

   auto f3 = RFieldBase::Create("fail", "std::vector<float>").Unwrap();
   try {
      model->AddProjectedField(std::move(f3), [](const std::string &) { return "met"; }).ThrowOnError();
      FAIL() << "mapping with structural mismatch should throw";
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("field mapping structural mismatch"));
   }

   auto f4 = RFieldBase::Create("fail", "int").Unwrap();
   try {
      model->AddProjectedField(std::move(f4), [](const std::string &) { return "met"; }).ThrowOnError();
      FAIL() << "mapping without matching type should throw";
   } catch (const ROOT::RException &err) {
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
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("unsupported field mapping "));
   }

   auto f6 = RFieldBase::Create("fail", "float").Unwrap();
   try {
      model->AddProjectedField(std::move(f6), [](const std::string &) { return "lorentz._0"; }).ThrowOnError();
      FAIL() << "mapping across fixed-size array should throw";
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("unsupported field mapping "));
   }

   auto f7 = RFieldBase::Create("fail", "std::vector<float>").Unwrap();
   try {
      model
         ->AddProjectedField(std::move(f7),
                             [](const std::string &fieldName) {
                                if (fieldName == "fail")
                                   return "nnlo._0";
                                return "nnlo._0._0";
                             })
         .ThrowOnError();
      FAIL() << "mapping scrambling the source structure should throw";
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("field mapping structure mismatch"));
   }
}

TEST(RNTupleProjection, CatchReaderWithProjectedFields)
{
   FileRaii fileGuard("test_ntuple_projection_catch_reader_with_projected_fields.root");

   auto model = RNTupleModel::Create();
   model->MakeField<float>("met");

   auto f1 = RFieldBase::Create("missingE", "float").Unwrap();
   model->AddProjectedField(std::move(f1), [](const std::string &) { return "met"; });

   auto modelRead = model->Clone();

   {
      auto writer = RNTupleWriter::Recreate(std::move(model), "A", fileGuard.GetPath());
   }

   try {
      auto reader = RNTupleReader::Open(std::move(modelRead), "A", fileGuard.GetPath());
      FAIL() << "creating a reader with a model with projected fields should throw";
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("model has projected fields"));
   }
}
