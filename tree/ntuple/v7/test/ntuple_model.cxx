#include "ntuple_test.hxx"

TEST(RNTupleModel, EnforceValidFieldNames)
{
   auto model = RNTupleModel::Create();

   // MakeField
   try {
      auto field3 = model->MakeField<float>("", 42.0);
      FAIL() << "empty string as field name should throw";
   } catch (const RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("name cannot be empty string"));
   }
   try {
      auto field3 = model->MakeField<float>("pt.pt", 42.0);
      FAIL() << "field name with periods should throw";
   } catch (const RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("name 'pt.pt' cannot contain character '.'"));
   }

   // Previous failures to create 'pt' should not block the name
   auto field = model->MakeField<float>("pt", 42.0);

   try {
      auto field2 = model->MakeField<float>("pt", 42.0);
      FAIL() << "repeated field names should throw";
   } catch (const RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("field name 'pt' already exists"));
   }

   // AddField
   try {
      model->AddField(std::make_unique<RField<float>>("pt"));
      FAIL() << "repeated field names should throw";
   } catch (const RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("field name 'pt' already exists"));
   }
}

TEST(RNTupleModel, FieldDescriptions)
{
   FileRaii fileGuard("test_ntuple_field_descriptions.root");
   auto model = RNTupleModel::Create();

   auto pt = model->MakeField<float>({"pt", "transverse momentum"}, 42.0);

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
   m->GetMutableField("cs.v1._0").SetColumnRepresentatives({{EColumnType::kReal32}});
   m->Freeze();
   EXPECT_EQ(m->GetConstField("x").GetFieldName(), "x");
   EXPECT_EQ(m->GetConstField("x").GetTypeName(), "std::int32_t");
   EXPECT_EQ(m->GetConstField("cs.v1").GetFieldName(), "v1");
   EXPECT_EQ(m->GetConstField("cs.v1").GetTypeName(), "std::vector<float>");
   EXPECT_EQ(m->GetConstField("cs.v1._0").GetColumnRepresentatives()[0][0], EColumnType::kReal32);
   try {
      m->GetConstField("nonexistent");
      FAIL() << "invalid field name should throw";
   } catch (const RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("invalid field"));
   }
   try {
      m->GetConstField("");
      FAIL() << "empty field name should throw";
   } catch (const RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("invalid field"));
   }
   try {
      m->GetMutableFieldZero();
      FAIL() << "GetMutableFieldZero should throw";
   } catch (const RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("frozen model"));
   }
   try {
      m->GetMutableField("x");
      FAIL() << "GetMutableField should throw";
   } catch (const RException &err) {
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
   static constexpr std::size_t ColumnElementsSize = 8 + 4 + 8 + 4 + 8 + 8 + 4 + 8 + 1 + 1;
   static constexpr std::size_t InitialNElementsPerPage = 1;
   static constexpr std::size_t MaxPageSize = 100;
   static constexpr std::size_t ClusterSize = 6789;
   RNTupleWriteOptions options;
   options.SetInitialNElementsPerPage(InitialNElementsPerPage);
   options.SetMaxUnzippedPageSize(MaxPageSize);
   options.SetApproxZippedClusterSize(ClusterSize);

   // Tail page optimization and buffered writing on, IMT not disabled.
   static constexpr std::size_t Expected1 = NumColumns * MaxPageSize + ColumnElementsSize + 3 * ClusterSize;
   EXPECT_EQ(model->EstimateWriteMemoryUsage(options), Expected1);

   static constexpr std::size_t PageBufferBudget = 800;
   options.SetPageBufferBudget(PageBufferBudget);
   static constexpr std::size_t Expected2 = PageBufferBudget + ColumnElementsSize + 3 * ClusterSize;
   EXPECT_EQ(model->EstimateWriteMemoryUsage(options), Expected2);

   // Disable IMT.
   options.SetUseImplicitMT(RNTupleWriteOptions::EImplicitMT::kOff);
   static constexpr std::size_t Expected3 = PageBufferBudget + ColumnElementsSize + ClusterSize;
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
         f.SetColumnRepresentatives({{EColumnType::kReal32}});
      }
      if (f.GetTypeName() == "std::uint32_t") {
         f.SetColumnRepresentatives({{EColumnType::kUInt32}});
      }
   }

   model->Freeze();
   auto clone = model->Clone();

   for (const auto &f : clone->GetConstFieldZero()) {
      if (f.GetTypeName() == "float") {
         EXPECT_EQ(EColumnType::kReal32, f.GetColumnRepresentatives()[0][0]);
      }
      if (f.GetTypeName() == "std::uint32_t") {
         EXPECT_EQ(EColumnType::kUInt32, f.GetColumnRepresentatives()[0][0]);
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
      model->MakeField<float>("a", 3.14f);
      model->MakeField<CustomStruct>("struct", CustomStruct{1.f, {2.f, 3.f}, {{4.f}, {5.f, 6.f, 7.f}}, "foo"});
      model->MakeField<std::vector<CustomStruct>>(
         "structVec", std::vector{CustomStruct{.1f, {.2f, .3f}, {{.4f}, {.5f, .6f, .7f}}, "bar"},
                                  CustomStruct{-1.f, {-2.f, -3.f}, {{-4.f}, {-5.f, -6.f, -7.f}}, "baz"}});
      model->MakeField<std::pair<CustomStruct, int>>(
         "structPair", std::pair{CustomStruct{.1f, {.2f, .3f}, {{.4f}, {.5f, .6f, .7f}}, "bar"}, 42});

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
   } catch (const RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("subfield \"struct.a\" already registered"));
   }

   try {
      model->RegisterSubfield("struct.doesnotexist");
      FAIL() << "attempting to register a nonexistent subfield should throw";
   } catch (const RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("could not find subfield \"struct.doesnotexist\" in model"));
   }

   try {
      model->RegisterSubfield("struct");
      FAIL() << "attempting to register a top-level field as subfield should throw";
   } catch (const RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("cannot register top-level field \"struct\" as a subfield"));
   }

   try {
      model->RegisterSubfield("structVec._0.s");
      FAIL() << "attempting to register a subfield in a collection should throw";
   } catch (const RException &err) {
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
   } catch (const RException &err) {
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
   } catch (const RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("invalid field name: struct.v1"));
   }
}

TEST(RNTupleModel, RegisterSubfieldBare)
{
   auto model = RNTupleModel::CreateBare();
   model->MakeField<CustomStruct>("struct");
   model->RegisterSubfield("struct.a");
   model->Freeze();

   EXPECT_THROW(model->GetDefaultEntry(), RException);

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
