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
      EXPECT_THAT(err.what(), testing::HasSubstr("name 'pt.pt' cannot contain dot characters '.'"));
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
   m->Freeze();
   EXPECT_EQ(m->GetConstField("x").GetFieldName(), "x");
   EXPECT_EQ(m->GetConstField("x").GetTypeName(), "std::int32_t");
   EXPECT_EQ(m->GetConstField("cs.v1").GetFieldName(), "v1");
   EXPECT_EQ(m->GetConstField("cs.v1").GetTypeName(), "std::vector<float>");
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
