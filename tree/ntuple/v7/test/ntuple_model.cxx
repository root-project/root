#include "ntuple_test.hxx"

TEST(RNTupleModel, EnforceValidFieldNames)
{
   auto model = RNTupleModel::Create();

   // MakeField
   try {
      auto field3 = model->MakeField<float>("");
      FAIL() << "empty string as field name should throw";
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("name cannot be empty string"));
   }
   try {
      auto field3 = model->MakeField<float>("pt.pt");
      FAIL() << "field name with periods should throw";
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("name 'pt.pt' cannot contain character '.'"));
   }

   // Previous failures to create 'pt' should not block the name
   auto field = model->MakeField<float>("pt");

   try {
      model->MakeField<float>("pt");
      FAIL() << "repeated field names should throw";
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("field name 'pt' already exists"));
   }

   // AddField
   try {
      model->AddField(std::make_unique<RField<float>>("pt"));
      FAIL() << "repeated field names should throw";
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("field name 'pt' already exists"));
   }
}

TEST(RNTupleModel, FieldDescriptions)
{
   FileRaii fileGuard("test_ntuple_field_descriptions.root");
   auto model = RNTupleModel::Create();

   model->MakeField<float>("pt", "transverse momentum");

   auto charge = std::make_unique<RField<float>>(RField<float>("charge"));
   charge->SetDescription("electric charge");
   model->AddField(std::move(charge));

   {
      RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard.GetPath());
   }

   auto ntuple = RNTupleReader::Open("ntuple", fileGuard.GetPath());
   std::vector<std::string> fieldDescriptions;
   for (auto &f : ntuple->GetDescriptor().GetTopLevelFields()) {
      fieldDescriptions.push_back(f.GetFieldDescription());
   }
   ASSERT_EQ(2u, fieldDescriptions.size());
   EXPECT_EQ(std::string("transverse momentum"), fieldDescriptions[0]);
   EXPECT_EQ(std::string("electric charge"), fieldDescriptions[1]);
}

TEST(RNTupleModel, GetField)
{
   auto m = RNTupleModel::Create();
   m->MakeField<int>("x");
   m->MakeField<CustomStruct>("cs");
   m->GetMutableField("cs.v1._0").SetColumnRepresentatives({{ROOT::ENTupleColumnType::kReal32}});
   m->Freeze();
   EXPECT_EQ(m->GetConstField("x").GetFieldName(), "x");
   EXPECT_EQ(m->GetConstField("x").GetTypeName(), "std::int32_t");
   EXPECT_EQ(m->GetConstField("cs.v1").GetFieldName(), "v1");
   EXPECT_EQ(m->GetConstField("cs.v1").GetTypeName(), "std::vector<float>");
   EXPECT_EQ(m->GetConstField("cs.v1._0").GetColumnRepresentatives()[0][0], ROOT::ENTupleColumnType::kReal32);
   try {
      m->GetConstField("nonexistent");
      FAIL() << "invalid field name should throw";
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("invalid field"));
   }
   try {
      m->GetConstField("");
      FAIL() << "empty field name should throw";
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("invalid field"));
   }
   try {
      m->GetMutableFieldZero();
      FAIL() << "GetMutableFieldZero should throw";
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("frozen model"));
   }
   try {
      m->GetMutableField("x");
      FAIL() << "GetMutableField should throw";
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("frozen model"));
   }
   EXPECT_EQ("", m->GetConstFieldZero().GetQualifiedFieldName());
   EXPECT_EQ("x", m->GetConstField("x").GetQualifiedFieldName());
   EXPECT_EQ("cs", m->GetConstField("cs").GetQualifiedFieldName());
   EXPECT_EQ("cs.v1", m->GetConstField("cs.v1").GetQualifiedFieldName());
}

TEST(RNTupleModel, EstimateWriteMemoryUsage)
{
   auto model = RNTupleModel::CreateBare();
   auto customStructVec = model->MakeField<std::vector<CustomStruct>>("CustomStructVec");

   static constexpr std::size_t NumColumns = 10;
   static constexpr std::size_t InitialPageSize = 8;
   static constexpr std::size_t MaxPageSize = 100;
   static constexpr std::size_t ClusterSize = 6789;
   RNTupleWriteOptions options;
   options.SetInitialUnzippedPageSize(InitialPageSize);
   options.SetMaxUnzippedPageSize(MaxPageSize);
   options.SetApproxZippedClusterSize(ClusterSize);

   // Buffered writing on, IMT not disabled.
   static constexpr std::size_t Expected1 = NumColumns * MaxPageSize + NumColumns * InitialPageSize + 3 * ClusterSize;
   EXPECT_EQ(model->EstimateWriteMemoryUsage(options), Expected1);

   static constexpr std::size_t PageBufferBudget = 800;
   options.SetPageBufferBudget(PageBufferBudget);
   static constexpr std::size_t Expected2 = PageBufferBudget + NumColumns * InitialPageSize + 3 * ClusterSize;
   EXPECT_EQ(model->EstimateWriteMemoryUsage(options), Expected2);

   // Disable IMT.
   options.SetUseImplicitMT(RNTupleWriteOptions::EImplicitMT::kOff);
   static constexpr std::size_t Expected3 = PageBufferBudget + NumColumns * InitialPageSize + ClusterSize;
   EXPECT_EQ(model->EstimateWriteMemoryUsage(options), Expected3);

   // Disable buffered writing.
   options.SetUseBufferedWrite(false);
   static constexpr std::size_t Expected4 = PageBufferBudget;
   EXPECT_EQ(model->EstimateWriteMemoryUsage(options), Expected4);
}

TEST(RNTupleModel, Clone)
{
   auto model = RNTupleModel::Create();
   model->MakeField<float>("f");
   model->MakeField<std::vector<float>>("vec");
   model->MakeField<CustomStruct>("struct");
   model->MakeField<TObject>("obj");
   model->MakeField<CustomEnumUInt32>("enum");

   for (auto &f : model->GetMutableFieldZero()) {
      if (f.GetTypeName() == "float") {
         f.SetColumnRepresentatives({{ROOT::ENTupleColumnType::kReal32}});
      }
      if (f.GetTypeName() == "std::uint32_t") {
         f.SetColumnRepresentatives({{ROOT::ENTupleColumnType::kUInt32}});
      }
   }

   model->Freeze();
   auto clone = model->Clone();

   for (const auto &f : clone->GetConstFieldZero()) {
      if (f.GetTypeName() == "float") {
         EXPECT_EQ(ROOT::ENTupleColumnType::kReal32, f.GetColumnRepresentatives()[0][0]);
      }
      if (f.GetTypeName() == "std::uint32_t") {
         EXPECT_EQ(ROOT::ENTupleColumnType::kUInt32, f.GetColumnRepresentatives()[0][0]);
      }
   }
   EXPECT_TRUE(clone->GetConstField("struct").GetTraits() & RFieldBase::kTraitTypeChecksum);
   EXPECT_TRUE(clone->GetConstField("obj").GetTraits() & RFieldBase::kTraitTypeChecksum);
}

TEST(RNTupleModel, RegisterSubfield)
{
   FileRaii fileGuard("test_rentry_subfields.root");
   {
      auto model = RNTupleModel::Create();
      *model->MakeField<float>("a") = 3.14f;
      *model->MakeField<CustomStruct>("struct") = CustomStruct{1.f, {2.f, 3.f}, {{4.f}, {5.f, 6.f, 7.f}}, "foo"};
      *model->MakeField<std::vector<CustomStruct>>("structVec") =
         std::vector{CustomStruct{.1f, {.2f, .3f}, {{.4f}, {.5f, .6f, .7f}}, "bar"},
                     CustomStruct{-1.f, {-2.f, -3.f}, {{-4.f}, {-5.f, -6.f, -7.f}}, "baz"}};
      *model->MakeField<std::pair<CustomStruct, int>>("structPair") =
         std::pair{CustomStruct{.1f, {.2f, .3f}, {{.4f}, {.5f, .6f, .7f}}, "bar"}, 42};

      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard.GetPath());
      ntuple->Fill();
   }

   auto model = RNTupleModel::Create();
   auto fldA = RFieldBase::Create("a", "float").Unwrap();
   auto fldStruct = RFieldBase::Create("struct", "CustomStruct").Unwrap();
   auto fldStructVec = RFieldBase::Create("structVec", "std::vector<CustomStruct>").Unwrap();
   auto fldStructPair = RFieldBase::Create("structPair", "std::pair<CustomStruct, int>").Unwrap();

   model->AddField(std::move(fldA));
   model->AddField(std::move(fldStruct));
   model->AddField(std::move(fldStructVec));
   model->AddField(std::move(fldStructPair));

   model->RegisterSubfield("struct.a");

   try {
      model->RegisterSubfield("struct.a");
      FAIL() << "attempting to re-register subfield should throw";
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("subfield \"struct.a\" already registered"));
   }

   try {
      model->RegisterSubfield("struct.doesnotexist");
      FAIL() << "attempting to register a nonexistent subfield should throw";
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("could not find subfield \"struct.doesnotexist\" in model"));
   }

   try {
      model->RegisterSubfield("struct");
      FAIL() << "attempting to register a top-level field as subfield should throw";
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("cannot register top-level field \"struct\" as a subfield"));
   }

   try {
      model->RegisterSubfield("structVec._0.s");
      FAIL() << "attempting to register a subfield in a collection should throw";
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(
         err.what(),
         testing::HasSubstr(
            "registering a subfield as part of a collection, fixed-sized array or std::variant is not supported"));
   }

   model->RegisterSubfield("structPair._0.s");

   model->Freeze();
   try {
      model->RegisterSubfield("struct.v1");
      FAIL() << "attempting to register a subfield in a frozen model should throw";
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("invalid attempt to modify frozen model"));
   }

   const auto &defaultEntry = model->GetDefaultEntry();
   const auto entry = model->CreateEntry();
   const auto bareEntry = model->CreateBareEntry();
   auto ntuple = RNTupleReader::Open(std::move(model), "ntuple", fileGuard.GetPath());

   // default entry
   ntuple->LoadEntry(0);
   EXPECT_FLOAT_EQ(1.f, *defaultEntry.GetPtr<float>("struct.a"));
   EXPECT_EQ("bar", *defaultEntry.GetPtr<std::string>("structPair._0.s"));
   auto defaultTokenStructA = defaultEntry.GetToken("struct.a");
   EXPECT_FLOAT_EQ(1.f, *defaultEntry.GetPtr<float>(defaultTokenStructA));

   // explicitly created entry
   ntuple->LoadEntry(0, *entry);
   EXPECT_FLOAT_EQ(1.f, *entry->GetPtr<float>("struct.a"));
   EXPECT_EQ("bar", *entry->GetPtr<std::string>("structPair._0.s"));
   auto tokenStructA = entry->GetToken("struct.a");
   EXPECT_FLOAT_EQ(1.f, *entry->GetPtr<float>(tokenStructA));

   // bare entry
   EXPECT_EQ(nullptr, bareEntry->GetPtr<float>("struct.a"));
   EXPECT_EQ(nullptr, bareEntry->GetPtr<std::string>("structPair._0.s"));

   auto cs = std::make_shared<CustomStruct>();
   bareEntry->BindValue("struct", cs);
   bareEntry->BindRawPtr("struct.a", &cs->a);
   auto pcs = std::make_shared<std::pair<CustomStruct, int>>();
   bareEntry->BindValue("structPair", pcs);
   bareEntry->BindRawPtr("structPair._0.s", &pcs->first.s);
   bareEntry->BindValue("a", std::make_shared<float>());
   bareEntry->BindValue("structVec", std::make_shared<std::vector<CustomStruct>>());

   ntuple->LoadEntry(0, *bareEntry);
   EXPECT_FLOAT_EQ(1.f, *bareEntry->GetPtr<float>("struct.a"));
   EXPECT_FLOAT_EQ(1.f, cs->a);
   EXPECT_EQ("bar", *bareEntry->GetPtr<std::string>("structPair._0.s"));
   EXPECT_EQ("bar", pcs->first.s);
   auto bareTokenStructA = bareEntry->GetToken("struct.a");
   EXPECT_FLOAT_EQ(1.f, *bareEntry->GetPtr<float>(bareTokenStructA));

   // sanity check
   try {
      *defaultEntry.GetPtr<std::vector<float>>("struct.v1");
      *entry->GetPtr<std::vector<float>>("struct.v1");
      FAIL() << "subfields not explicitly registered shouldn't be present in the entry";
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("invalid field name: struct.v1"));
   }
}

TEST(RNTupleModel, RegisterSubfieldBare)
{
   auto model = RNTupleModel::CreateBare();
   model->MakeField<CustomStruct>("struct");
   model->RegisterSubfield("struct.a");
   model->Freeze();

   EXPECT_THROW(model->GetDefaultEntry(), ROOT::RException);

   const auto entry = model->CreateEntry();
   EXPECT_TRUE(entry->GetPtr<float>("struct.a"));
}

TEST(RNTupleModel, CloneRegisteredSubfield)
{
   auto model = RNTupleModel::Create();
   model->MakeField<CustomStruct>("struct");
   model->RegisterSubfield("struct.a");
   model->Freeze();

   auto clone = model->Clone();
   EXPECT_TRUE(clone->GetDefaultEntry().GetPtr<float>("struct.a"));
}

TEST(RNTupleModel, Expire)
{
   auto model = RNTupleModel::Create();
   model->MakeField<CustomStruct>("struct")->a = 1.0;

   try {
      model->Expire();
      FAIL() << "attempting expire unfrozen model should fail";
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("invalid attempt to expire unfrozen model"));
   }

   model->Freeze();
   model->Expire();

   EXPECT_EQ(0u, model->GetModelId());
   EXPECT_NE(0, model->GetSchemaId());
   auto clone = model->Clone();
   EXPECT_NE(0, clone->GetModelId());
   EXPECT_EQ(model->GetSchemaId(), clone->GetSchemaId());

   auto token = model->GetToken("struct");
   EXPECT_FLOAT_EQ(1.0, model->GetDefaultEntry().GetPtr<CustomStruct>(token)->a);

   EXPECT_TRUE(model->GetRegisteredSubfieldNames().empty());
   EXPECT_TRUE(model->GetDescription().empty());

   EXPECT_THROW(model->MakeField<float>("E"), ROOT::RException);
   EXPECT_THROW(model->AddField(RFieldBase::Create("E", "float").Unwrap()), ROOT::RException);
   EXPECT_THROW(model->RegisterSubfield("struct.a"), ROOT::RException);
   EXPECT_THROW(model->AddProjectedField(RFieldBase::Create("a", "float").Unwrap(),
                                         [](const std::string &) { return "struct.a"; }),
                ROOT::RException);
   EXPECT_THROW(model->CreateEntry(), ROOT::RException);
   EXPECT_THROW(model->CreateBareEntry(), ROOT::RException);
   EXPECT_THROW(model->CreateBulk("struct"), ROOT::RException);
   EXPECT_THROW(model->GetMutableFieldZero(), ROOT::RException);
   EXPECT_THROW(model->GetMutableField("struct"), ROOT::RException);
   EXPECT_THROW(model->SetDescription("x"), ROOT::RException);

   FileRaii fileGuard("test_ntuple_model_expire.root");
   EXPECT_THROW(RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath()), ROOT::RException);
   auto writer = RNTupleWriter::Recreate(std::move(clone), "ntpl", fileGuard.GetPath());
   writer.reset();
   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
   EXPECT_EQ(0u, reader->GetNEntries());
}

TEST(RNTupleModel, ExpireWithWriter)
{
   FileRaii fileGuard("test_ntuple_model_expire_with_writer.root");

   auto model = RNTupleModel::Create();
   *model->MakeField<float>("pt") = 1.0;
   auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
   writer->Fill();
   writer->CommitDataset();

   writer->CommitCluster(true); // noop
   writer->FlushCluster();      // noop;
   writer->CommitDataset();     // noop
   EXPECT_EQ(1u, writer->GetNEntries());
   EXPECT_EQ(1u, writer->GetLastFlushed());
   EXPECT_EQ(1u, writer->GetLastCommitted());
   EXPECT_EQ(1u, writer->GetLastCommittedClusterGroup());

   EXPECT_THROW(writer->Fill(), ROOT::RException);
   EXPECT_THROW(writer->FlushColumns(), ROOT::RException);
   EXPECT_THROW(writer->CreateEntry(), ROOT::RException);
   EXPECT_THROW(writer->CreateModelUpdater(), ROOT::RException);

   EXPECT_FLOAT_EQ(1.0, *writer->GetModel().GetDefaultEntry().GetPtr<float>("pt"));
}
