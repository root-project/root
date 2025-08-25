#include "ntuple_test.hxx"

#include <TVirtualStreamerInfo.h>

TEST(RFieldDescriptorBuilder, MakeDescriptorErrors)
{
   // minimum requirements for making a field descriptor from scratch
   RFieldDescriptor fieldDesc = RFieldDescriptorBuilder()
                                   .FieldId(1)
                                   .Structure(ROOT::ENTupleStructure::kCollection)
                                   .FieldName("someField")
                                   .MakeDescriptor()
                                   .Unwrap();

   // MakeDescriptor() returns an RResult<RFieldDescriptor>
   // -- here we check the error cases

   // must set field id
   auto fieldDescRes = RFieldDescriptorBuilder().MakeDescriptor();
   ASSERT_FALSE(fieldDescRes) << "default constructed dangling descriptors should throw";
   EXPECT_THAT(fieldDescRes.GetError()->GetReport(), testing::HasSubstr("invalid field id"));

   // must set field structure
   fieldDescRes = RFieldDescriptorBuilder().FieldId(1).MakeDescriptor();
   ASSERT_FALSE(fieldDescRes) << "field descriptors without structure should throw";
   EXPECT_THAT(fieldDescRes.GetError()->GetReport(), testing::HasSubstr("invalid field structure"));

   // must set field name
   fieldDescRes =
      RFieldDescriptorBuilder().FieldId(1).ParentId(1).Structure(ROOT::ENTupleStructure::kCollection).MakeDescriptor();
   ASSERT_FALSE(fieldDescRes) << "unnamed field descriptors should throw";
   EXPECT_THAT(fieldDescRes.GetError()->GetReport(), testing::HasSubstr("name cannot be empty string"));
}

TEST(RNTupleDescriptorBuilder, CatchBadLinks)
{
   RNTupleDescriptorBuilder descBuilder;
   descBuilder.AddField(
      RFieldDescriptorBuilder().FieldId(0).Structure(ROOT::ENTupleStructure::kRecord).MakeDescriptor().Unwrap());
   descBuilder.AddField(RFieldDescriptorBuilder()
                           .FieldId(1)
                           .FieldName("field")
                           .TypeName("int32_t")
                           .Structure(ROOT::ENTupleStructure::kPlain)
                           .MakeDescriptor()
                           .Unwrap());
   try {
      descBuilder.AddFieldLink(1, 0);
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("cannot make FieldZero a child field"));
   }
   try {
      descBuilder.AddFieldLink(1, 1);
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("cannot make field '1' a child of itself"));
   }
   try {
      descBuilder.AddFieldLink(1, 10);
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("child field with id '10' doesn't exist in NTuple"));
   }
}

TEST(RNTupleDescriptorBuilder, CatchBadProjections)
{
   RNTupleDescriptorBuilder descBuilder;
   descBuilder.AddField(
      RFieldDescriptorBuilder().FieldId(0).Structure(ROOT::ENTupleStructure::kRecord).MakeDescriptor().Unwrap());
   descBuilder.AddField(RFieldDescriptorBuilder()
                           .FieldId(1)
                           .FieldName("field")
                           .TypeName("int32_t")
                           .Structure(ROOT::ENTupleStructure::kPlain)
                           .MakeDescriptor()
                           .Unwrap());
   descBuilder.AddField(RFieldDescriptorBuilder()
                           .FieldId(2)
                           .FieldName("projField")
                           .TypeName("int32_t")
                           .Structure(ROOT::ENTupleStructure::kPlain)
                           .MakeDescriptor()
                           .Unwrap());
   descBuilder.AddField(RFieldDescriptorBuilder()
                           .FieldId(3)
                           .FieldName("projField")
                           .TypeName("int32_t")
                           .Structure(ROOT::ENTupleStructure::kPlain)
                           .MakeDescriptor()
                           .Unwrap());

   try {
      descBuilder.AddFieldProjection(1, 4);
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("doesn't exist"));
   }
   try {
      descBuilder.AddFieldProjection(4, 2);
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("doesn't exist"));
   }
   try {
      descBuilder.AddFieldProjection(1, 0);
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("cannot make FieldZero a projected field"));
   }
   try {
      descBuilder.AddFieldProjection(2, 2);
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("projection of itself"));
   }

   descBuilder.AddFieldProjection(1, 2);
   try {
      descBuilder.AddFieldProjection(2, 1);
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("projection of an already projected field"));
   }
   try {
      descBuilder.AddFieldProjection(3, 2);
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("has already a projection source"));
   }
}

TEST(RNTupleDescriptorBuilder, CatchBadColumnDescriptors)
{
   RNTupleDescriptorBuilder descBuilder;
   descBuilder.AddField(
      RFieldDescriptorBuilder().FieldId(0).Structure(ROOT::ENTupleStructure::kRecord).MakeDescriptor().Unwrap());
   descBuilder.AddField(RFieldDescriptorBuilder()
                           .FieldId(1)
                           .FieldName("field")
                           .TypeName("int32_t")
                           .Structure(ROOT::ENTupleStructure::kPlain)
                           .MakeDescriptor()
                           .Unwrap());
   descBuilder.AddField(RFieldDescriptorBuilder()
                           .FieldId(2)
                           .FieldName("fieldAlias")
                           .TypeName("int32_t")
                           .Structure(ROOT::ENTupleStructure::kPlain)
                           .MakeDescriptor()
                           .Unwrap());
   descBuilder.AddFieldLink(0, 1);
   descBuilder.AddFieldLink(0, 2);
   ROOT::ENTupleColumnType colType{ROOT::ENTupleColumnType::kInt32};
   RColumnDescriptorBuilder colBuilder1;
   colBuilder1.LogicalColumnId(0).PhysicalColumnId(0).BitsOnStorage(32).Type(colType).FieldId(1).Index(0);
   descBuilder.AddColumn(colBuilder1.MakeDescriptor().Unwrap()).ThrowOnError();

   RColumnDescriptorBuilder colBuilder2;
   colBuilder2.LogicalColumnId(1).PhysicalColumnId(0).BitsOnStorage(32).Type(colType).FieldId(42).Index(0);
   try {
      descBuilder.AddColumn(colBuilder2.MakeDescriptor().Unwrap()).ThrowOnError();
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("doesn't exist"));
   }

   RColumnDescriptorBuilder colBuilder3;
   colBuilder3.LogicalColumnId(0).PhysicalColumnId(0).BitsOnStorage(32).Type(colType).FieldId(1).Index(0);
   try {
      descBuilder.AddColumn(colBuilder3.MakeDescriptor().Unwrap()).ThrowOnError();
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("column index clash"));
   }

   RColumnDescriptorBuilder colBuilder4;
   colBuilder4.LogicalColumnId(1).PhysicalColumnId(0).BitsOnStorage(32).Type(colType).FieldId(2).Index(1);
   try {
      descBuilder.AddColumn(colBuilder4.MakeDescriptor().Unwrap()).ThrowOnError();
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("out of bounds column index"));
   }

   RColumnDescriptorBuilder colBuilder5;
   ROOT::ENTupleColumnType falseType(ROOT::ENTupleColumnType::kInt64);
   colBuilder5.LogicalColumnId(1).PhysicalColumnId(0).BitsOnStorage(64).Type(falseType).FieldId(2).Index(0);
   try {
      descBuilder.AddColumn(colBuilder5.MakeDescriptor().Unwrap()).ThrowOnError();
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("alias column type mismatch"));
   }

   RColumnDescriptorBuilder colBuilder6;
   colBuilder6.LogicalColumnId(1).PhysicalColumnId(0).BitsOnStorage(32).Type(colType).FieldId(2).Index(0);
   descBuilder.AddColumn(colBuilder6.MakeDescriptor().Unwrap()).ThrowOnError();
}

TEST(RNTupleDescriptorBuilder, CatchInvalidDescriptors)
{
   RNTupleDescriptorBuilder descBuilder;
   descBuilder.SetVersionForWriting();

   // empty string is not a valid NTuple name
   descBuilder.SetNTuple("", "");
   try {
      descBuilder.EnsureValidDescriptor();
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("name cannot be empty string"));
   }
   descBuilder.SetNTuple("something", "");
   descBuilder.EnsureValidDescriptor();
}

TEST(RFieldDescriptorBuilder, HeaderExtension)
{
   RNTupleDescriptorBuilder descBuilder;
   descBuilder.SetVersionForWriting();
   descBuilder.SetNTuple("ntpl", "");
   descBuilder.AddField(
      RFieldDescriptorBuilder().FieldId(0).Structure(ROOT::ENTupleStructure::kRecord).MakeDescriptor().Unwrap());
   descBuilder.AddField(RFieldDescriptorBuilder()
                           .FieldId(1)
                           .FieldName("i32")
                           .TypeName("int32_t")
                           .Structure(ROOT::ENTupleStructure::kPlain)
                           .MakeDescriptor()
                           .Unwrap());
   descBuilder.AddColumn(RColumnDescriptorBuilder()
                            .LogicalColumnId(0)
                            .PhysicalColumnId(0)
                            .BitsOnStorage(32)
                            .Type(ROOT::ENTupleColumnType::kInt32)
                            .FieldId(1)
                            .Index(0)
                            .MakeDescriptor()
                            .Unwrap());
   descBuilder.AddFieldLink(0, 1);

   EXPECT_TRUE(descBuilder.GetDescriptor().GetHeaderExtension() == nullptr);
   descBuilder.BeginHeaderExtension();
   EXPECT_TRUE(descBuilder.GetDescriptor().GetHeaderExtension() != nullptr);

   descBuilder.AddField(RFieldDescriptorBuilder()
                           .FieldId(2)
                           .FieldName("topLevel1")
                           .Structure(ROOT::ENTupleStructure::kRecord)
                           .MakeDescriptor()
                           .Unwrap());
   descBuilder.AddField(RFieldDescriptorBuilder()
                           .FieldId(3)
                           .FieldName("i64")
                           .TypeName("int64_t")
                           .Structure(ROOT::ENTupleStructure::kPlain)
                           .MakeDescriptor()
                           .Unwrap());
   descBuilder.AddColumn(RColumnDescriptorBuilder()
                            .LogicalColumnId(1)
                            .PhysicalColumnId(1)
                            .BitsOnStorage(64)
                            .Type(ROOT::ENTupleColumnType::kInt64)
                            .FieldId(3)
                            .Index(0)
                            .FirstElementIndex(1002)
                            .MakeDescriptor()
                            .Unwrap());
   descBuilder.AddFieldLink(2, 3);
   descBuilder.AddFieldLink(0, 2);
   descBuilder.AddField(RFieldDescriptorBuilder()
                           .FieldId(4)
                           .FieldName("topLevel2")
                           .TypeName("bool")
                           .Structure(ROOT::ENTupleStructure::kPlain)
                           .MakeDescriptor()
                           .Unwrap());
   descBuilder.AddColumn(RColumnDescriptorBuilder()
                            .LogicalColumnId(2)
                            .PhysicalColumnId(2)
                            .BitsOnStorage(1)
                            .Type(ROOT::ENTupleColumnType::kBit)
                            .FieldId(4)
                            .Index(0)
                            .FirstElementIndex(1100)
                            .MakeDescriptor()
                            .Unwrap());
   descBuilder.AddFieldLink(0, 4);
   descBuilder.AddField(RFieldDescriptorBuilder()
                           .FieldId(5)
                           .FieldName("projected")
                           .TypeName("int64_t")
                           .Structure(ROOT::ENTupleStructure::kPlain)
                           .MakeDescriptor()
                           .Unwrap());
   descBuilder.AddColumn(RColumnDescriptorBuilder()
                            .LogicalColumnId(3)
                            .PhysicalColumnId(1)
                            .BitsOnStorage(64)
                            .Type(ROOT::ENTupleColumnType::kInt64)
                            .FieldId(5)
                            .Index(0)
                            .MakeDescriptor()
                            .Unwrap());
   descBuilder.AddFieldLink(0, 5);

   auto desc = descBuilder.MoveDescriptor();
   ASSERT_EQ(desc.GetNFields(), 6);
   ASSERT_EQ(desc.GetNLogicalColumns(), 4);
   ASSERT_EQ(desc.GetNPhysicalColumns(), 3);
   {
      std::string_view child_names[] = {"i32", "topLevel1", "topLevel2", "projected"};
      unsigned i = 0;
      for (auto &child_field : desc.GetTopLevelFields())
         EXPECT_EQ(child_field.GetFieldName(), child_names[i++]);
   }
   auto xHeader = desc.GetHeaderExtension();
   EXPECT_EQ(xHeader->GetNFields(), 4);
   EXPECT_EQ(xHeader->GetNLogicalColumns(), 3);
   EXPECT_EQ(xHeader->GetNPhysicalColumns(), 2);
   {
      std::string_view child_names[] = {"topLevel1", "topLevel2", "projected"};
      unsigned i = 0;
      for (auto child_field : xHeader->GetTopLevelFields(desc))
         EXPECT_EQ(desc.GetFieldDescriptor(child_field).GetFieldName(), child_names[i++]);
   }
   EXPECT_EQ(desc.GetColumnDescriptor(0).GetFirstElementIndex(), 0U);
   EXPECT_EQ(desc.GetColumnDescriptor(1).GetFirstElementIndex(), 1002U);
   EXPECT_EQ(desc.GetColumnDescriptor(2).GetFirstElementIndex(), 1100U);
}

TEST(RNTupleDescriptor, QualifiedFieldName)
{
   auto model = RNTupleModel::Create();
   model->MakeField<std::int32_t>("ints");
   model->MakeField<std::vector<float>>("jets");
   FileRaii fileGuard("test_field_qualified.root");
   {
      RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard.GetPath());
   }

   auto ntuple = RNTupleReader::Open("ntuple", fileGuard.GetPath());
   const auto &desc = ntuple->GetDescriptor();
   EXPECT_TRUE(desc.GetQualifiedFieldName(desc.GetFieldZeroId()).empty());
   auto fldIdInts = desc.FindFieldId("ints");
   EXPECT_STREQ("ints", desc.GetQualifiedFieldName(fldIdInts).c_str());
   auto fldIdJets = desc.FindFieldId("jets");
   auto fldIdInner = desc.FindFieldId("_0", fldIdJets);
   EXPECT_STREQ("jets", desc.GetQualifiedFieldName(fldIdJets).c_str());
   EXPECT_STREQ("jets._0", desc.GetQualifiedFieldName(fldIdInner).c_str());
}

// https://github.com/root-project/root/issues/19442
TEST(RNTupleDescriptor, GetTypeNameForComparison)
{
   static const std::string MetaTypeName = "Template<Template<float> >";
   static const std::string RenormalizedTypeName = "Template<Template<float>>";

   RFieldDescriptor fieldDesc = RFieldDescriptorBuilder()
                                   .FieldId(1)
                                   .Structure(ROOT::ENTupleStructure::kRecord)
                                   .FieldName("f")
                                   .TypeName(MetaTypeName)
                                   .MakeDescriptor()
                                   .Unwrap();
   {
      RNTupleDescriptorBuilder descBuilder;
      // Pretend the descriptor was created before the spec version that clarified type normalization (ROOT v6.34)
      descBuilder.SetVersion(1, 0, 0, 0);
      descBuilder.SetNTuple("ntpl", "");

      auto desc = descBuilder.MoveDescriptor();
      ASSERT_EQ(desc.GetTypeNameForComparison(fieldDesc), RenormalizedTypeName);
   }

   {
      RNTupleDescriptorBuilder descBuilder;
      // For later spec versions, we should not hide the problem
      descBuilder.SetVersion(1, 0, 0, 1);
      descBuilder.SetNTuple("ntpl", "");

      auto desc = descBuilder.MoveDescriptor();
      ASSERT_EQ(desc.GetTypeNameForComparison(fieldDesc), MetaTypeName);
   }
}

TEST(RFieldDescriptorIterable, IterateOverFieldNames)
{
   auto model = RNTupleModel::Create();
   auto floats = model->MakeField<std::vector<float>>("jets");
   auto bools = model->MakeField<std::vector<bool>>("bools");
   auto bool_vec_vec = model->MakeField<std::vector<std::vector<bool>>>("bool_vec_vec");
   auto ints = model->MakeField<std::int32_t>("ints");

   FileRaii fileGuard("test_field_iterator.root");
   {
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard.GetPath());
      writer->Fill();
   }

   auto ntuple = RNTupleReader::Open("ntuple", fileGuard.GetPath());
   // iterate over top-level fields
   std::vector<std::string> names{};
   for (auto &f : ntuple->GetDescriptor().GetTopLevelFields()) {
      names.push_back(f.GetFieldName());
   }
   ASSERT_EQ(names.size(), 4);
   EXPECT_EQ(names[0], std::string("jets"));
   EXPECT_EQ(names[1], std::string("bools"));
   EXPECT_EQ(names[2], std::string("bool_vec_vec"));
   EXPECT_EQ(names[3], std::string("ints"));

   const auto &ntuple_desc = ntuple->GetDescriptor();
   auto top_level_fields = ntuple_desc.GetTopLevelFields();

   // iterate over child field ranges
   const auto &float_vec_desc = *top_level_fields.begin();
   EXPECT_EQ(float_vec_desc.GetFieldName(), std::string("jets"));
   auto float_vec_child_range = ntuple_desc.GetFieldIterable(float_vec_desc);
   std::vector<std::string> child_names{};
   for (auto &child_field : float_vec_child_range) {
      child_names.push_back(child_field.GetFieldName());
      // check the empty range
      auto float_child_range = ntuple_desc.GetFieldIterable(child_field);
      EXPECT_EQ(float_child_range.begin(), float_child_range.end());
   }
   ASSERT_EQ(child_names.size(), 1);
   EXPECT_EQ(child_names[0], std::string("_0"));

   // check if canonical iterator methods work
   auto iter = top_level_fields.begin();
   std::advance(iter, 2);
   const auto &bool_vec_vec_desc = *iter;
   EXPECT_EQ(bool_vec_vec_desc.GetFieldName(), std::string("bool_vec_vec"));

   child_names.clear();
   for (auto &child_field : ntuple_desc.GetFieldIterable(bool_vec_vec_desc)) {
      child_names.push_back(child_field.GetFieldName());
   }
   ASSERT_EQ(child_names.size(), 1);
   EXPECT_EQ(child_names[0], std::string("_0"));
}

TEST(RFieldDescriptorIterable, SortByLambda)
{
   auto model = RNTupleModel::Create();
   auto floats = model->MakeField<std::vector<float>>("jets");
   auto bools = model->MakeField<std::vector<bool>>("bools");
   auto bool_vec_vec = model->MakeField<std::vector<std::vector<bool>>>("bool_vec_vec");
   auto ints = model->MakeField<std::int32_t>("ints");

   FileRaii fileGuard("test_field_iterator.root");
   {
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard.GetPath());
      writer->Fill();
   }

   auto ntuple = RNTupleReader::Open("ntuple", fileGuard.GetPath());
   const auto &ntuple_desc = ntuple->GetDescriptor();
   auto alpha_order = [&](auto lhs, auto rhs) {
      return ntuple_desc.GetFieldDescriptor(lhs).GetFieldName() < ntuple_desc.GetFieldDescriptor(rhs).GetFieldName();
   };

   std::vector<std::string> sorted_names = {};
   for (auto &f : ntuple_desc.GetTopLevelFields(alpha_order)) {
      sorted_names.push_back(f.GetFieldName());
   }
   ASSERT_EQ(sorted_names.size(), 4);
   EXPECT_EQ(sorted_names[0], std::string("bool_vec_vec"));
   EXPECT_EQ(sorted_names[1], std::string("bools"));
   EXPECT_EQ(sorted_names[2], std::string("ints"));
   EXPECT_EQ(sorted_names[3], std::string("jets"));

   // reverse alphabetical
   sorted_names.clear();
   for (auto &f : ntuple_desc.GetTopLevelFields([&](auto lhs, auto rhs) { return !alpha_order(lhs, rhs); })) {
      sorted_names.push_back(f.GetFieldName());
   }
   ASSERT_EQ(sorted_names.size(), 4);
   EXPECT_EQ(sorted_names[0], std::string("jets"));
   EXPECT_EQ(sorted_names[1], std::string("ints"));
   EXPECT_EQ(sorted_names[2], std::string("bools"));
   EXPECT_EQ(sorted_names[3], std::string("bool_vec_vec"));

   // alphabetical by type name
   std::vector<std::string> sorted_by_typename = {};
   for (auto &f : ntuple_desc.GetTopLevelFields([&](auto lhs, auto rhs) {
           return ntuple_desc.GetFieldDescriptor(lhs).GetTypeName() < ntuple_desc.GetFieldDescriptor(rhs).GetTypeName();
        })) {
      sorted_by_typename.push_back(f.GetFieldName());
   }
   // int32_t, vector<bool>, vector<float>, vector<vector<bool>
   ASSERT_EQ(sorted_by_typename.size(), 4);
   EXPECT_EQ(sorted_by_typename[0], std::string("ints"));
   EXPECT_EQ(sorted_by_typename[1], std::string("bools"));
   EXPECT_EQ(sorted_by_typename[2], std::string("jets"));
   EXPECT_EQ(sorted_by_typename[3], std::string("bool_vec_vec"));
}

TEST(RColumnDescriptorIterable, IterateOverColumns)
{
   auto model = RNTupleModel::Create();
   auto floats = model->MakeField<std::vector<float>>("jets");
   auto bools = model->MakeField<std::string>("tag");

   FileRaii fileGuard("test_column_iterator.root");
   {
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard.GetPath());
      writer->Fill();
   }

   auto ntuple = RNTupleReader::Open("ntuple", fileGuard.GetPath());
   const auto &desc = ntuple->GetDescriptor();

   // No column attached to the zero field
   unsigned int counter = 0;
   for (const auto &c : desc.GetColumnIterable(desc.GetFieldZeroId())) {
      (void)c;
      counter++;
   }
   EXPECT_EQ(0u, counter);

   const auto tagId = desc.FindFieldId("tag");
   for (const auto &c : desc.GetColumnIterable(tagId)) {
      EXPECT_EQ(tagId, c.GetFieldId());
      EXPECT_EQ(counter, c.GetIndex());
      counter++;
   }
   EXPECT_EQ(2, counter);

   const auto jetsId = desc.FindFieldId("jets");
   for (const auto &c : desc.GetColumnIterable(desc.FindFieldId("jets"))) {
      EXPECT_EQ(jetsId, c.GetFieldId());
      counter++;
   }
   EXPECT_EQ(3, counter);

   counter = 0;
   for (const auto &c : desc.GetColumnIterable()) {
      (void)c;
      counter++;
   }
   EXPECT_EQ(desc.GetNLogicalColumns(), counter);
}

TEST(RClusterDescriptor, GetNBytesOnStorage)
{
   auto model = RNTupleModel::Create();
   auto fldJets = model->MakeField<std::vector<float>>("jets");
   auto fldTag = model->MakeField<std::string>("tag");

   FileRaii fileGuard("test_descriptor_bytes_on_storage.root");
   {
      RNTupleWriteOptions options;
      options.SetCompression(0);
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard.GetPath(), options);
      fldJets->push_back(1.0);
      fldJets->push_back(2.0);
      *fldTag = "abc";
      writer->Fill();
   }

   auto ntuple = RNTupleReader::Open("ntuple", fileGuard.GetPath());
   const auto &desc = ntuple->GetDescriptor();

   auto clusterID = desc.FindClusterId(0, 0);
   ASSERT_NE(ROOT::kInvalidDescriptorId, clusterID);
   EXPECT_EQ(8 + 8 + 8 + 3, desc.GetClusterDescriptor(clusterID).GetNBytesOnStorage());
}

TEST(RNTupleDescriptor, Clone)
{
   auto model = RNTupleModel::Create();
   auto floats = model->MakeField<std::vector<float>>("jets");
   auto bools = model->MakeField<std::string>("tag");

   FileRaii fileGuard("test_ntuple_descriptor_clone.root");
   {
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard.GetPath());
      writer->Fill();
      writer->CommitCluster(true /* commitClusterGroup */);
      writer->Fill();
   }

   auto ntuple = RNTupleReader::Open("ntuple", fileGuard.GetPath());
   const auto &desc = ntuple->GetDescriptor();
   auto clone = desc.Clone();
   EXPECT_EQ(desc, clone);
}

TEST(RNTupleDescriptor, BuildStreamerInfos)
{
   auto fnBuildStreamerInfosOf = [](const RFieldBase &field) -> RNTupleSerializer::StreamerInfoMap_t {
      RNTupleDescriptorBuilder descBuilder;
      descBuilder.SetNTuple("test", "");
      descBuilder.AddField(
         RFieldDescriptorBuilder().FieldId(0).Structure(ROOT::ENTupleStructure::kRecord).MakeDescriptor().Unwrap());
      auto fieldBuilder = RFieldDescriptorBuilder::FromField(field);
      descBuilder.AddField(fieldBuilder.FieldId(1).MakeDescriptor().Unwrap());
      descBuilder.AddFieldLink(0, 1);
      int i = 2;
      // In this test, we only support field hierarchies up to 2 levels
      for (const auto &child : field.GetConstSubfields()) {
         fieldBuilder = RFieldDescriptorBuilder::FromField(*child);
         descBuilder.AddField(fieldBuilder.FieldId(i).MakeDescriptor().Unwrap());
         descBuilder.AddFieldLink(1, i);
         const auto childId = i;
         i++;
         for (const auto &grandChild : child->GetConstSubfields()) {
            fieldBuilder = RFieldDescriptorBuilder::FromField(*grandChild);
            descBuilder.AddField(fieldBuilder.FieldId(i).MakeDescriptor().Unwrap());
            descBuilder.AddFieldLink(childId, i);
            i++;
         }
      }
      return descBuilder.BuildStreamerInfos();
   };

   RNTupleSerializer::StreamerInfoMap_t streamerInfoMap;

   streamerInfoMap = fnBuildStreamerInfosOf(*RFieldBase::Create("f", "float").Unwrap());
   EXPECT_TRUE(streamerInfoMap.empty());

   streamerInfoMap = fnBuildStreamerInfosOf(*RFieldBase::Create("f", "std::vector<float>").Unwrap());
   EXPECT_TRUE(streamerInfoMap.empty());

   streamerInfoMap = fnBuildStreamerInfosOf(*RFieldBase::Create("f", "std::pair<float, float>").Unwrap());
   EXPECT_TRUE(streamerInfoMap.empty());

   streamerInfoMap = fnBuildStreamerInfosOf(*RFieldBase::Create("f", "std::map<int, float>").Unwrap());
   EXPECT_TRUE(streamerInfoMap.empty());

   streamerInfoMap = fnBuildStreamerInfosOf(*RFieldBase::Create("f", "std::unordered_map<int, float>").Unwrap());
   EXPECT_TRUE(streamerInfoMap.empty());

   std::vector<std::unique_ptr<RFieldBase>> itemFields;
   streamerInfoMap = fnBuildStreamerInfosOf(ROOT::RRecordField("f", std::move(itemFields)));
   EXPECT_TRUE(streamerInfoMap.empty());

   streamerInfoMap = fnBuildStreamerInfosOf(*RFieldBase::Create("f", "CustomStruct").Unwrap());
   EXPECT_EQ(1u, streamerInfoMap.size());
   EXPECT_STREQ("CustomStruct", streamerInfoMap.begin()->second->GetName());

   streamerInfoMap = fnBuildStreamerInfosOf(*RFieldBase::Create("f", "std::vector<CustomStruct>").Unwrap());
   EXPECT_EQ(1u, streamerInfoMap.size());
   EXPECT_STREQ("CustomStruct", streamerInfoMap.begin()->second->GetName());

   streamerInfoMap = fnBuildStreamerInfosOf(*RFieldBase::Create("f", "std::map<int, CustomStruct>").Unwrap());
   EXPECT_EQ(1u, streamerInfoMap.size());
   EXPECT_STREQ("CustomStruct", streamerInfoMap.begin()->second->GetName());

   streamerInfoMap = fnBuildStreamerInfosOf(*RFieldBase::Create("f", "DerivedA").Unwrap());
   EXPECT_EQ(2u, streamerInfoMap.size());
   std::vector<std::string> typeNames;
   for (const auto &[_, si] : streamerInfoMap) {
      typeNames.emplace_back(si->GetName());
   }
   std::sort(typeNames.begin(), typeNames.end());
   EXPECT_STREQ("CustomStruct", typeNames[0].c_str());
   EXPECT_STREQ("DerivedA", typeNames[1].c_str());

   streamerInfoMap = fnBuildStreamerInfosOf(*RFieldBase::Create("f", "std::pair<CustomStruct, DerivedA>").Unwrap());
   EXPECT_EQ(2u, streamerInfoMap.size());
   typeNames.clear();
   for (const auto &[_, si] : streamerInfoMap) {
      typeNames.emplace_back(si->GetName());
   }
   std::sort(typeNames.begin(), typeNames.end());
   EXPECT_STREQ("CustomStruct", typeNames[0].c_str());
   EXPECT_STREQ("DerivedA", typeNames[1].c_str());
}

TEST(RNTupleDescriptor, CloneSchema)
{
   FileRaii fileGuard("test_ntuple_desc_cloneschema.root");

   {
      auto model = RNTupleModel::Create();
      auto pFlt = model->MakeField<float>("flt");
      auto pVec = model->MakeField<std::vector<int>>("vec");
      auto fProj = RFieldBase::Create("flt2", "float").Unwrap();
      model->AddProjectedField(std::move(fProj), [](const std::string &) { return "flt"; });
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());

      for (int i = 0; i < 100; ++i) {
         *pFlt = i;
         pVec->push_back(i);
         writer->Fill();
         if ((i % 10) == 0)
            writer->CommitCluster();
      }

      // late model extend to get a header extension in the descriptor
      auto updater = writer->CreateModelUpdater();
      updater->BeginUpdate();
      updater->AddField(RFieldBase::Create("ext", "int").Unwrap());
      auto fProjDef = RFieldBase::Create("vec2", "std::vector<int>").Unwrap();
      updater
         ->AddProjectedField(std::move(fProjDef),
                             [](const std::string &name) { return name == "vec2" ? "vec" : "vec._0"; })
         .ThrowOnError();
      updater->CommitUpdate();

      for (int i = 0; i < 100; ++i) {
         writer->Fill();
         if ((i % 10) == 0)
            writer->CommitCluster();
      }
   }

   // Read back the RNTuple
   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
   const auto &desc = reader->GetDescriptor();
   EXPECT_EQ(desc.GetNFields(), 8);
   EXPECT_EQ(desc.GetNLogicalColumns(), 7);
   EXPECT_EQ(desc.GetNPhysicalColumns(), 4);
   EXPECT_EQ(desc.GetNClusters(), 21);
   EXPECT_EQ(desc.GetNClusterGroups(), 1);
   EXPECT_EQ(desc.GetNEntries(), 200);
   EXPECT_NE(desc.GetHeaderExtension(), nullptr);
   EXPECT_NE(desc.GetOnDiskHeaderXxHash3(), 0);
   EXPECT_NE(desc.GetOnDiskHeaderSize(), 0);
   EXPECT_NE(desc.GetOnDiskFooterSize(), 0);

   const auto descOnlySchema = ROOT::Internal::CloneDescriptorSchema(desc);
   EXPECT_EQ(descOnlySchema.GetNFields(), 8);
   EXPECT_EQ(descOnlySchema.GetNLogicalColumns(), 7);
   EXPECT_EQ(descOnlySchema.GetNPhysicalColumns(), 4);
   EXPECT_EQ(descOnlySchema.GetNClusters(), 0);
   EXPECT_EQ(descOnlySchema.GetNClusterGroups(), 0);
   EXPECT_EQ(descOnlySchema.GetNEntries(), 0);
   EXPECT_NE(descOnlySchema.GetHeaderExtension(), nullptr);
   EXPECT_EQ(descOnlySchema.GetOnDiskHeaderXxHash3(), 0);
   EXPECT_EQ(descOnlySchema.GetOnDiskHeaderSize(), 0);
   EXPECT_EQ(descOnlySchema.GetOnDiskFooterSize(), 0);
}
