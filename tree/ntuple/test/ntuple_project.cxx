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
   options.SetReconstructProjections(true);
   auto reconstructedModel = reader->GetDescriptor().CreateModel(options);
   auto itrFields = reconstructedModel->GetConstFieldZero().cbegin();
   EXPECT_EQ("met", itrFields->GetQualifiedFieldName());
   EXPECT_EQ("atomicNumber", (++itrFields)->GetQualifiedFieldName());
   EXPECT_EQ("atomicNumber._0", (++itrFields)->GetQualifiedFieldName());
   EXPECT_EQ("vec", (++itrFields)->GetQualifiedFieldName());
   EXPECT_EQ("vec._0", (++itrFields)->GetQualifiedFieldName());
   EXPECT_EQ(reconstructedModel->GetConstFieldZero().cend(), ++itrFields);
   auto &projectedFields = ROOT::Internal::GetProjectedFieldsOfModel(*reconstructedModel);
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

TEST(RNTupleProjection, AliasQuantDiffPrec)
{
   // Try creating alias columns of Real32Quant columns with a different precision

   FileRaii fileGuard("test_ntuple_projection_quant_diff_prec.root");

   auto model = RNTupleModel::Create();

   auto field = std::make_unique<RField<float>>("q");
   field->SetQuantized(0, 1, 20); // Set quantized with 20 bits of precision
   model->AddField(std::move(field));

   auto modelRead = model->Clone();

   {
      ROOT::TestSupport::CheckDiagsRAII diags;
      diags.requiredDiag(kWarning, "RProjectedFields", "on a projected field has no effect", false);

      auto projField = std::make_unique<RField<float>>("projq");
      projField->SetQuantized(0, 1, 30); // Set quantized with 30 bits of precision
      model->AddProjectedField(std::move(projField), [](const auto &) { return "q"; });
   }

   {
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
      auto pq = writer->GetModel().GetDefaultEntry().GetPtr<float>("q");
      for (int i = 0; i < 10; ++i) {
         *pq = i * 0.05f;
         writer->Fill();
      }
   }

   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
   auto vq = reader->GetView<float>("q");
   auto vpq = reader->GetView<float>("projq");
   for (int i = 0; i < 10; ++i) {
      EXPECT_FLOAT_EQ(vq(i), vpq(i));
   }
}

TEST(RNTupleProjection, AliasQuantDiffRange)
{
   // Try creating alias columns of Real32Quant columns with a different range

   FileRaii fileGuard("test_ntuple_projection_quant_diff_range.root");

   auto model = RNTupleModel::Create();

   auto field = std::make_unique<RField<float>>("q");
   field->SetQuantized(0, 1, 20);
   model->AddField(std::move(field));

   auto modelRead = model->Clone();

   {
      ROOT::TestSupport::CheckDiagsRAII diags;
      diags.requiredDiag(kWarning, "RProjectedFields", "on a projected field has no effect", false);

      auto projField = std::make_unique<RField<float>>("projq");
      projField->SetQuantized(0, 2, 20);
      model->AddProjectedField(std::move(projField), [](const auto &) { return "q"; });
   }

   {
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
      auto pq = writer->GetModel().GetDefaultEntry().GetPtr<float>("q");
      for (int i = 0; i < 10; ++i) {
         *pq = i * 0.05f;
         writer->Fill();
      }
   }

   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
   auto pq = reader->GetModel().GetDefaultEntry().GetPtr<float>("q");
   auto ppq = reader->GetModel().GetDefaultEntry().GetPtr<float>("projq");
   for (int i = 0; i < 10; ++i) {
      reader->LoadEntry(i);
      EXPECT_FLOAT_EQ(*pq, *ppq);
   }
}

TEST(RNTupleProjection, AliasTruncDiffPrec)
{
   // Try creating alias columns of Real32Trunc columns with a different precision

   FileRaii fileGuard("test_ntuple_projection_quant_diff_prec.root");

   auto model = RNTupleModel::Create();

   auto field = std::make_unique<RField<float>>("q");
   field->SetTruncated(20);
   model->AddField(std::move(field));

   auto modelRead = model->Clone();

   {
      ROOT::TestSupport::CheckDiagsRAII diags;
      diags.requiredDiag(kWarning, "RProjectedFields", "on a projected field has no effect", false);

      auto projField = std::make_unique<RField<float>>("projq");
      projField->SetTruncated(20);
      model->AddProjectedField(std::move(projField), [](const auto &) { return "q"; });
   }

   {
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
      auto pq = writer->GetModel().GetDefaultEntry().GetPtr<float>("q");
      for (int i = 0; i < 10; ++i) {
         *pq = i * 0.05f;
         writer->Fill();
      }
   }

   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
   auto vq = reader->GetView<float>("q");
   auto vpq = reader->GetView<float>("projq");
   for (int i = 0; i < 10; ++i) {
      EXPECT_FLOAT_EQ(vq(i), vpq(i));
   }
}

TEST(RNTupleProjection, AliasTruncQuantProj)
{
   // Try creating alias columns mixing Real32Trunc and Real32Quant columns

   FileRaii fileGuard("test_ntuple_projection_trunc_quant.root");

   auto model = RNTupleModel::Create();

   auto fTrunc = std::make_unique<RField<float>>("trunc");
   fTrunc->SetTruncated(20);
   model->AddField(std::move(fTrunc));
   auto fQuant = std::make_unique<RField<float>>("quant");
   fQuant->SetQuantized(-1, 1, 30);
   model->AddField(std::move(fQuant));

   auto modelRead = model->Clone();

   {
      ROOT::TestSupport::CheckDiagsRAII diags;
      diags.requiredDiag(kWarning, "RProjectedFields", "on a projected field has no effect", false);

      auto projTruncToQuant = std::make_unique<RField<float>>("truncToQuant");
      projTruncToQuant->SetQuantized(0, 10, 20);
      model->AddProjectedField(std::move(projTruncToQuant), [](const auto &) { return "trunc"; });

      auto projQuantToTrunc = std::make_unique<RField<float>>("quantToTrunc");
      projQuantToTrunc->SetTruncated(10);
      model->AddProjectedField(std::move(projQuantToTrunc), [](const auto &) { return "quant"; });
   }

   {
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
      auto pQuant = writer->GetModel().GetDefaultEntry().GetPtr<float>("quant");
      auto pTrunc = writer->GetModel().GetDefaultEntry().GetPtr<float>("trunc");
      for (int i = 0; i < 10; ++i) {
         *pQuant = i * 0.05f;
         *pTrunc = i * 2.f;
         writer->Fill();
      }
   }

   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
   auto vq = reader->GetView<double>("quant");
   auto vqt = reader->GetView<double>("quantToTrunc");
   auto vt = reader->GetView<double>("trunc");
   auto vtq = reader->GetView<double>("truncToQuant");
   for (int i = 0; i < 10; ++i) {
      EXPECT_FLOAT_EQ(vq(i), vqt(i));
      EXPECT_FLOAT_EQ(vt(i), vtq(i));
   }
}
