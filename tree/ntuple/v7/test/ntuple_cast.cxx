#include "ntuple_test.hxx"

#include <cstdint>
#include <utility>

template <typename T>
static std::shared_ptr<T>
MakeField(std::string_view name, ROOT::Experimental::EColumnType colType, ROOT::Experimental::RNTupleModel &model)
{
   auto field = std::make_unique<RField<T>>(name);
   field->SetColumnRepresentatives({{colType}});
   model.AddField(std::move(field));
   return model.GetDefaultEntry().GetPtr<T>(name);
}

TEST(RNTuple, TypeCastInvalid)
{
   FileRaii fileGuard("test_ntuple_type_cast_invalid.root");

   {
      auto model = RNTupleModel::Create();
      model->MakeField<float>("x");
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
   }

   auto castModelInvalid = RNTupleModel::Create();
   castModelInvalid->MakeField<int>("x");
   EXPECT_THROW(RNTupleReader::Open(std::move(castModelInvalid), "ntpl", fileGuard.GetPath()), ROOT::RException);

   auto castModel = RNTupleModel::Create();
   castModel->MakeField<double>("x");
   EXPECT_NO_THROW(RNTupleReader::Open(std::move(castModel), "ntpl", fileGuard.GetPath()));
}

TEST(RNTuple, TypeCastInCollection)
{
   FileRaii fileGuard("test_ntuple_type_cast_in_collection.root");

   {
      auto model = RNTupleModel::Create();
      auto ptrVInt8 = model->MakeField<std::vector<std::int8_t>>("vint8");
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
      ptrVInt8->push_back(1);
      ptrVInt8->push_back(0);
      ptrVInt8->push_back(-128);
      writer->Fill();
   }

   auto castModel = RNTupleModel::Create();
   auto ptrVInt8 = castModel->MakeField<std::vector<bool>>("vint8");
   auto reader = RNTupleReader::Open(std::move(castModel), "ntpl", fileGuard.GetPath());
   reader->LoadEntry(0);
   EXPECT_EQ(3u, ptrVInt8->size());
   EXPECT_TRUE(ptrVInt8->at(0));
   EXPECT_FALSE(ptrVInt8->at(1));
   EXPECT_TRUE(ptrVInt8->at(2));
}

TEST(RNTuple, TypeCastBool)
{
   FileRaii fileGuard("test_ntuple_type_cast_bool.root");

   {
      auto model = RNTupleModel::Create();
      auto ptrChar = model->MakeField<char>("char");
      auto ptrInt8 = model->MakeField<std::int8_t>("int8");
      auto ptrUInt8 = model->MakeField<std::uint8_t>("uint8");
      auto ptrInt16Split = model->MakeField<std::int16_t>("int16_split");
      auto ptrUInt16Split = model->MakeField<std::uint16_t>("uint16_split");
      auto ptrInt32Split = model->MakeField<std::int32_t>("int32_split");
      auto ptrUInt32Split = model->MakeField<std::uint32_t>("uint32_split");
      auto ptrInt64Split = model->MakeField<std::int64_t>("int64_split");
      auto ptrUInt64Split = model->MakeField<std::uint64_t>("uint64_split");

      auto ptrInt16Unsplit = MakeField<std::int16_t>("int16_unsplit", EColumnType::kInt16, *model);
      auto ptrUInt16Unsplit = MakeField<std::uint16_t>("uint16_unsplit", EColumnType::kUInt16, *model);
      auto ptrInt32Unsplit = MakeField<std::int32_t>("int32_unsplit", EColumnType::kInt32, *model);
      auto ptrUInt32Unsplit = MakeField<std::uint32_t>("uint32_unsplit", EColumnType::kUInt32, *model);
      auto ptrInt64Unsplit = MakeField<std::int64_t>("int64_unsplit", EColumnType::kInt64, *model);
      auto ptrUInt64Unsplit = MakeField<std::uint64_t>("uint64_unsplit", EColumnType::kUInt64, *model);

      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
      *ptrChar = 127;
      *ptrInt8 = -128;
      *ptrUInt8 = 255;
      *ptrInt16Split = -1;
      *ptrUInt16Split = 1;
      *ptrInt32Split = -1;
      *ptrUInt32Split = 1;
      *ptrInt64Split = -1;
      *ptrUInt64Split = 1;
      *ptrInt16Unsplit = -1;
      *ptrUInt16Unsplit = 1;
      *ptrInt32Unsplit = -1;
      *ptrUInt32Unsplit = 1;
      *ptrInt64Unsplit = -1;
      *ptrUInt64Unsplit = 1;
      writer->Fill();
      *ptrChar = 0;
      *ptrInt8 = 0;
      *ptrUInt8 = 0;
      *ptrInt16Split = 0;
      *ptrUInt16Split = 0;
      *ptrInt32Split = 0;
      *ptrUInt32Split = 0;
      *ptrInt64Split = 0;
      *ptrUInt64Split = 0;
      *ptrInt16Unsplit = 0;
      *ptrUInt16Unsplit = 0;
      *ptrInt32Unsplit = 0;
      *ptrUInt32Unsplit = 0;
      *ptrInt64Unsplit = 0;
      *ptrUInt64Unsplit = 0;
      writer->Fill();
   }

   auto castModel = RNTupleModel::Create();
   auto ptrChar = castModel->MakeField<bool>("char");
   auto ptrInt8 = castModel->MakeField<bool>("int8");
   auto ptrUInt8 = castModel->MakeField<bool>("uint8");
   auto ptrInt16Split = castModel->MakeField<bool>("int16_split");
   auto ptrUInt16Split = castModel->MakeField<bool>("uint16_split");
   auto ptrInt32Split = castModel->MakeField<bool>("int32_split");
   auto ptrUInt32Split = castModel->MakeField<bool>("uint32_split");
   auto ptrInt64Split = castModel->MakeField<bool>("int64_split");
   auto ptrUInt64Split = castModel->MakeField<bool>("uint64_split");
   auto ptrInt16Unsplit = castModel->MakeField<bool>("int16_unsplit");
   auto ptrUInt16Unsplit = castModel->MakeField<bool>("uint16_unsplit");
   auto ptrInt32Unsplit = castModel->MakeField<bool>("int32_unsplit");
   auto ptrUInt32Unsplit = castModel->MakeField<bool>("uint32_unsplit");
   auto ptrInt64Unsplit = castModel->MakeField<bool>("int64_unsplit");
   auto ptrUInt64Unsplit = castModel->MakeField<bool>("uint64_unsplit");
   auto reader = RNTupleReader::Open(std::move(castModel), "ntpl", fileGuard.GetPath());
   reader->LoadEntry(0);
   EXPECT_TRUE(*ptrChar);
   EXPECT_TRUE(*ptrInt8);
   EXPECT_TRUE(*ptrUInt8);
   EXPECT_TRUE(*ptrInt16Split);
   EXPECT_TRUE(*ptrUInt16Split);
   EXPECT_TRUE(*ptrInt32Split);
   EXPECT_TRUE(*ptrUInt32Split);
   EXPECT_TRUE(*ptrInt64Split);
   EXPECT_TRUE(*ptrUInt64Split);
   EXPECT_TRUE(*ptrInt16Unsplit);
   EXPECT_TRUE(*ptrUInt16Unsplit);
   EXPECT_TRUE(*ptrInt32Unsplit);
   EXPECT_TRUE(*ptrUInt32Unsplit);
   EXPECT_TRUE(*ptrInt64Unsplit);
   EXPECT_TRUE(*ptrUInt64Unsplit);
   reader->LoadEntry(1);
   EXPECT_FALSE(*ptrChar);
   EXPECT_FALSE(*ptrInt8);
   EXPECT_FALSE(*ptrUInt8);
   EXPECT_FALSE(*ptrInt16Split);
   EXPECT_FALSE(*ptrUInt16Split);
   EXPECT_FALSE(*ptrInt32Split);
   EXPECT_FALSE(*ptrUInt32Split);
   EXPECT_FALSE(*ptrInt64Split);
   EXPECT_FALSE(*ptrUInt64Split);
   EXPECT_FALSE(*ptrInt16Unsplit);
   EXPECT_FALSE(*ptrUInt16Unsplit);
   EXPECT_FALSE(*ptrInt32Unsplit);
   EXPECT_FALSE(*ptrUInt32Unsplit);
   EXPECT_FALSE(*ptrInt64Unsplit);
   EXPECT_FALSE(*ptrUInt64Unsplit);
}

TEST(RNTuple, TypeCastReal)
{
   FileRaii fileGuard("test_ntuple_type_cast_real.root");

   {
      auto model = RNTupleModel::Create();
      auto ptrFloat = model->MakeField<float>("float");
      auto ptrDouble = model->MakeField<double>("double");

      auto fieldFloatHalf = std::make_unique<RField<float>>("float_half");
      fieldFloatHalf->SetHalfPrecision();
      model->AddField(std::move(fieldFloatHalf));
      auto ptrFloatHalf = model->GetDefaultEntry().GetPtr<float>("float_half");

      auto fieldDoubleHalf = std::make_unique<RField<double>>("double_half");
      fieldDoubleHalf->SetHalfPrecision();
      model->AddField(std::move(fieldDoubleHalf));
      auto ptrDoubleHalf = model->GetDefaultEntry().GetPtr<double>("double_half");

      auto fieldFloatTrunc = std::make_unique<RField<float>>("float_trunc");
      fieldFloatTrunc->SetTruncated(20);
      model->AddField(std::move(fieldFloatTrunc));
      auto ptrFloatTrunc = model->GetDefaultEntry().GetPtr<float>("float_trunc");

      auto fieldDoubleTrunc = std::make_unique<RField<double>>("double_trunc");
      fieldDoubleTrunc->SetTruncated(16);
      model->AddField(std::move(fieldDoubleTrunc));
      auto ptrDoubleTrunc = model->GetDefaultEntry().GetPtr<double>("double_trunc");

      auto fieldFloatQuant = std::make_unique<RField<float>>("float_quant");
      fieldFloatQuant->SetQuantized(-1, 1, 12);
      model->AddField(std::move(fieldFloatQuant));
      auto ptrFloatQuant = model->GetDefaultEntry().GetPtr<float>("float_quant");

      auto fieldDoubleQuant = std::make_unique<RField<double>>("double_quant");
      fieldDoubleQuant->SetQuantized(0, 1, 32);
      model->AddField(std::move(fieldDoubleQuant));
      auto ptrDoubleQuant = model->GetDefaultEntry().GetPtr<double>("double_quant");

      auto fieldDouble32 = std::make_unique<RField<double>>("double32");
      fieldDouble32->SetDouble32();
      model->AddField(std::move(fieldDouble32));
      auto ptrDouble32 = model->GetDefaultEntry().GetPtr<double>("double32");

      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
      *ptrFloat = 42.4242;
      *ptrDouble = -55.567;
      *ptrFloatHalf = 12.345;
      *ptrDoubleHalf = -777.777;
      *ptrFloatTrunc = 0.00345;
      *ptrDoubleTrunc = -0.001;
      *ptrFloatQuant = -0.567;
      *ptrDoubleQuant = 0.782;
      *ptrDouble32 = 11.043;
      writer->Fill();
      *ptrFloat = -42.4242;
      *ptrDouble = 55.567;
      *ptrFloatHalf = -12.345;
      *ptrDoubleHalf = 777.777;
      *ptrFloatTrunc = -0.00345;
      *ptrDoubleTrunc = 0.001;
      *ptrFloatQuant = 0.567;
      *ptrDoubleQuant = 0.113;
      *ptrDouble32 = -11.043;
      writer->Fill();
   }

   // Read back each pointer as the "opposite" type
   auto castModel = RNTupleModel::Create();
   auto ptrFloat = castModel->MakeField<double>("float");
   auto ptrDouble = castModel->MakeField<float>("double");
   auto ptrFloatHalf = castModel->MakeField<double>("float_half");
   auto ptrDoubleHalf = castModel->MakeField<float>("double_half");
   auto ptrFloatTrunc = castModel->MakeField<double>("float_trunc");
   auto ptrDoubleTrunc = castModel->MakeField<float>("double_trunc");
   auto ptrFloatQuant = castModel->MakeField<double>("float_quant");
   auto ptrDoubleQuant = castModel->MakeField<float>("double_quant");
   auto ptrDouble32 = castModel->MakeField<float>("double32");

   auto reader = RNTupleReader::Open(std::move(castModel), "ntpl", fileGuard.GetPath());
   reader->LoadEntry(0);
   EXPECT_FLOAT_EQ(*ptrFloat, 42.4242);
   EXPECT_FLOAT_EQ(*ptrDouble, -55.567);
   EXPECT_NEAR(*ptrFloatHalf, 12.345, 0.005);
   EXPECT_NEAR(*ptrDoubleHalf, -777.777, 0.5);
   EXPECT_NEAR(*ptrFloatTrunc, 0.00345, 0.05);
   EXPECT_NEAR(*ptrDoubleTrunc, -0.001, 0.05);
   EXPECT_NEAR(*ptrFloatQuant, -0.567, 0.005);
   EXPECT_FLOAT_EQ(*ptrDoubleQuant, 0.782);
   EXPECT_FLOAT_EQ(*ptrDouble32, 11.043);
   reader->LoadEntry(1);
   EXPECT_FLOAT_EQ(*ptrFloat, -42.4242);
   EXPECT_FLOAT_EQ(*ptrDouble, 55.567);
   EXPECT_NEAR(*ptrFloatHalf, -12.345, 0.005);
   EXPECT_NEAR(*ptrDoubleHalf, 777.777, 0.5);
   EXPECT_NEAR(*ptrFloatTrunc, -0.00345, 0.05);
   EXPECT_NEAR(*ptrDoubleTrunc, 0.001, 0.05);
   EXPECT_NEAR(*ptrFloatQuant, 0.567, 0.005);
   EXPECT_FLOAT_EQ(*ptrDoubleQuant, 0.113);
   EXPECT_FLOAT_EQ(*ptrDouble32, -11.043);
}

TEST(RNTuple, TypeCastChar)
{
   FileRaii fileGuard("test_ntuple_type_cast_char.root");

   {
      auto model = RNTupleModel::Create();
      auto ptrBool = model->MakeField<bool>("bool");
      auto ptrInt8 = model->MakeField<std::int8_t>("int8");
      auto ptrUInt8 = model->MakeField<std::uint8_t>("uint8");

      auto ptrInt16Split = MakeField<std::int16_t>("int16_split", EColumnType::kSplitInt16, *model);
      auto ptrUInt16Split = MakeField<std::uint16_t>("uint16_split", EColumnType::kSplitUInt16, *model);
      auto ptrInt32Split = MakeField<std::int32_t>("int32_split", EColumnType::kSplitInt32, *model);
      auto ptrUInt32Split = MakeField<std::uint32_t>("uint32_split", EColumnType::kSplitUInt32, *model);
      auto ptrInt64Split = MakeField<std::int64_t>("int64_split", EColumnType::kSplitInt64, *model);
      auto ptrUInt64Split = MakeField<std::uint64_t>("uint64_split", EColumnType::kSplitUInt64, *model);

      auto ptrInt16Unsplit = MakeField<std::int16_t>("int16_unsplit", EColumnType::kInt16, *model);
      auto ptrUInt16Unsplit = MakeField<std::uint16_t>("uint16_unsplit", EColumnType::kUInt16, *model);
      auto ptrInt32Unsplit = MakeField<std::int32_t>("int32_unsplit", EColumnType::kInt32, *model);
      auto ptrUInt32Unsplit = MakeField<std::uint32_t>("uint32_unsplit", EColumnType::kUInt32, *model);
      auto ptrInt64Unsplit = MakeField<std::int64_t>("int64_unsplit", EColumnType::kInt64, *model);
      auto ptrUInt64Unsplit = MakeField<std::uint64_t>("uint64_unsplit", EColumnType::kUInt64, *model);

      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
      *ptrBool = true;
      *ptrInt8 = 2;
      *ptrUInt8 = 3;
      *ptrInt16Split = 4;
      *ptrUInt16Split = 5;
      *ptrInt32Split = 6;
      *ptrUInt32Split = 7;
      *ptrInt64Split = 8;
      *ptrUInt64Split = 9;
      *ptrInt16Unsplit = 10;
      *ptrUInt16Unsplit = 11;
      *ptrInt32Unsplit = 12;
      *ptrUInt32Unsplit = 13;
      *ptrInt64Unsplit = 14;
      *ptrUInt64Unsplit = 15;
      writer->Fill();
      *ptrBool = false;
      *ptrInt8 = 127;
      *ptrUInt8 = 126;
      *ptrInt16Split = 125;
      *ptrUInt16Split = 124;
      *ptrInt32Split = 123;
      *ptrUInt32Split = 122;
      *ptrInt64Split = 121;
      *ptrUInt64Split = 120;
      *ptrInt16Unsplit = 119;
      *ptrUInt16Unsplit = 118;
      *ptrInt32Unsplit = 117;
      *ptrUInt32Unsplit = 116;
      *ptrInt64Unsplit = 115;
      *ptrUInt64Unsplit = 114;
      writer->Fill();
      writer->CommitCluster();
      *ptrInt16Unsplit = 313;
      writer->Fill();
   }

   auto castModel = RNTupleModel::Create();
   auto ptrBool = castModel->MakeField<char>("bool");
   auto ptrInt8 = castModel->MakeField<char>("int8");
   auto ptrUInt8 = castModel->MakeField<char>("uint8");
   auto ptrInt16Split = castModel->MakeField<char>("int16_split");
   auto ptrUInt16Split = castModel->MakeField<char>("uint16_split");
   auto ptrInt32Split = castModel->MakeField<char>("int32_split");
   auto ptrUInt32Split = castModel->MakeField<char>("uint32_split");
   auto ptrInt64Split = castModel->MakeField<char>("int64_split");
   auto ptrUInt64Split = castModel->MakeField<char>("uint64_split");
   auto ptrInt16Unsplit = castModel->MakeField<char>("int16_unsplit");
   auto ptrUInt16Unsplit = castModel->MakeField<char>("uint16_unsplit");
   auto ptrInt32Unsplit = castModel->MakeField<char>("int32_unsplit");
   auto ptrUInt32Unsplit = castModel->MakeField<char>("uint32_unsplit");
   auto ptrInt64Unsplit = castModel->MakeField<char>("int64_unsplit");
   auto ptrUInt64Unsplit = castModel->MakeField<char>("uint64_unsplit");
   auto reader = RNTupleReader::Open(std::move(castModel), "ntpl", fileGuard.GetPath());
   reader->LoadEntry(0);
   EXPECT_EQ(static_cast<char>(1), *ptrBool);
   EXPECT_EQ(static_cast<char>(2), *ptrInt8);
   EXPECT_EQ(static_cast<char>(3), *ptrUInt8);
   EXPECT_EQ(static_cast<char>(4), *ptrInt16Split);
   EXPECT_EQ(static_cast<char>(5), *ptrUInt16Split);
   EXPECT_EQ(static_cast<char>(6), *ptrInt32Split);
   EXPECT_EQ(static_cast<char>(7), *ptrUInt32Split);
   EXPECT_EQ(static_cast<char>(8), *ptrInt64Split);
   EXPECT_EQ(static_cast<char>(9), *ptrUInt64Split);
   EXPECT_EQ(static_cast<char>(10), *ptrInt16Unsplit);
   EXPECT_EQ(static_cast<char>(11), *ptrUInt16Unsplit);
   EXPECT_EQ(static_cast<char>(12), *ptrInt32Unsplit);
   EXPECT_EQ(static_cast<char>(13), *ptrUInt32Unsplit);
   EXPECT_EQ(static_cast<char>(14), *ptrInt64Unsplit);
   EXPECT_EQ(static_cast<char>(15), *ptrUInt64Unsplit);
   reader->LoadEntry(1);
   EXPECT_EQ(static_cast<char>(0), *ptrBool);
   EXPECT_EQ(static_cast<char>(127), *ptrInt8);
   EXPECT_EQ(static_cast<char>(126), *ptrUInt8);
   EXPECT_EQ(static_cast<char>(125), *ptrInt16Split);
   EXPECT_EQ(static_cast<char>(124), *ptrUInt16Split);
   EXPECT_EQ(static_cast<char>(123), *ptrInt32Split);
   EXPECT_EQ(static_cast<char>(122), *ptrUInt32Split);
   EXPECT_EQ(static_cast<char>(121), *ptrInt64Split);
   EXPECT_EQ(static_cast<char>(120), *ptrUInt64Split);
   EXPECT_EQ(static_cast<char>(119), *ptrInt16Unsplit);
   EXPECT_EQ(static_cast<char>(118), *ptrUInt16Unsplit);
   EXPECT_EQ(static_cast<char>(117), *ptrInt32Unsplit);
   EXPECT_EQ(static_cast<char>(116), *ptrUInt32Unsplit);
   EXPECT_EQ(static_cast<char>(115), *ptrInt64Unsplit);
   EXPECT_EQ(static_cast<char>(114), *ptrUInt64Unsplit);
   try {
      reader->LoadEntry(2);
      FAIL() << "value out of range should throw";
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("value out of range"));
   }
}

TEST(RNTuple, TypeCastInt8)
{
   FileRaii fileGuard("test_ntuple_type_cast_Int8.root");

   {
      auto model = RNTupleModel::Create();
      auto ptrBool = model->MakeField<bool>("bool");
      auto ptrChar = model->MakeField<char>("char");
      auto ptrUInt8 = model->MakeField<std::uint8_t>("uint8");

      auto ptrInt16Split = MakeField<std::int16_t>("int16_split", EColumnType::kSplitInt16, *model);
      auto ptrUInt16Split = MakeField<std::uint16_t>("uint16_split", EColumnType::kSplitUInt16, *model);
      auto ptrInt32Split = MakeField<std::int32_t>("int32_split", EColumnType::kSplitInt32, *model);
      auto ptrUInt32Split = MakeField<std::uint32_t>("uint32_split", EColumnType::kSplitUInt32, *model);
      auto ptrInt64Split = MakeField<std::int64_t>("int64_split", EColumnType::kSplitInt64, *model);
      auto ptrUInt64Split = MakeField<std::uint64_t>("uint64_split", EColumnType::kSplitUInt64, *model);

      auto ptrInt16Unsplit = MakeField<std::int16_t>("int16_unsplit", EColumnType::kInt16, *model);
      auto ptrUInt16Unsplit = MakeField<std::uint16_t>("uint16_unsplit", EColumnType::kUInt16, *model);
      auto ptrInt32Unsplit = MakeField<std::int32_t>("int32_unsplit", EColumnType::kInt32, *model);
      auto ptrUInt32Unsplit = MakeField<std::uint32_t>("uint32_unsplit", EColumnType::kUInt32, *model);
      auto ptrInt64Unsplit = MakeField<std::int64_t>("int64_unsplit", EColumnType::kInt64, *model);
      auto ptrUInt64Unsplit = MakeField<std::uint64_t>("uint64_unsplit", EColumnType::kUInt64, *model);

      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
      *ptrBool = true;
      *ptrChar = 2;
      *ptrUInt8 = 1;
      *ptrInt16Split = 127;
      *ptrUInt16Split = 1;
      *ptrInt32Split = 127;
      *ptrUInt32Split = 1;
      *ptrInt64Split = 127;
      *ptrUInt64Split = 1;
      *ptrInt16Unsplit = 127;
      *ptrUInt16Unsplit = 1;
      *ptrInt32Unsplit = 127;
      *ptrUInt32Unsplit = 1;
      *ptrInt64Unsplit = 127;
      *ptrUInt64Unsplit = 1;
      writer->Fill();
      *ptrBool = false;
      *ptrChar = 127;
      *ptrUInt8 = 127;
      *ptrInt16Split = -128;
      *ptrUInt16Split = 127;
      *ptrInt32Split = -128;
      *ptrUInt32Split = 127;
      *ptrInt64Split = -128;
      *ptrUInt64Split = 127;
      *ptrInt16Unsplit = -128;
      *ptrUInt16Unsplit = 127;
      *ptrInt32Unsplit = -128;
      *ptrUInt32Unsplit = 127;
      *ptrInt64Unsplit = -128;
      *ptrUInt64Unsplit = 127;
      writer->Fill();
      writer->CommitCluster();
      *ptrInt16Unsplit = -313;
      writer->Fill();
   }

   auto castModel = RNTupleModel::Create();
   auto ptrBool = castModel->MakeField<std::int8_t>("bool");
   auto ptrChar = castModel->MakeField<std::int8_t>("char");
   auto ptrUInt8 = castModel->MakeField<std::int8_t>("uint8");
   auto ptrInt16Split = castModel->MakeField<std::int8_t>("int16_split");
   auto ptrUInt16Split = castModel->MakeField<std::int8_t>("uint16_split");
   auto ptrInt32Split = castModel->MakeField<std::int8_t>("int32_split");
   auto ptrUInt32Split = castModel->MakeField<std::int8_t>("uint32_split");
   auto ptrInt64Split = castModel->MakeField<std::int8_t>("int64_split");
   auto ptrUInt64Split = castModel->MakeField<std::int8_t>("uint64_split");
   auto ptrInt16Unsplit = castModel->MakeField<std::int8_t>("int16_unsplit");
   auto ptrUInt16Unsplit = castModel->MakeField<std::int8_t>("uint16_unsplit");
   auto ptrInt32Unsplit = castModel->MakeField<std::int8_t>("int32_unsplit");
   auto ptrUInt32Unsplit = castModel->MakeField<std::int8_t>("uint32_unsplit");
   auto ptrInt64Unsplit = castModel->MakeField<std::int8_t>("int64_unsplit");
   auto ptrUInt64Unsplit = castModel->MakeField<std::int8_t>("uint64_unsplit");
   auto reader = RNTupleReader::Open(std::move(castModel), "ntpl", fileGuard.GetPath());
   reader->LoadEntry(0);
   EXPECT_EQ(1, *ptrBool);
   EXPECT_EQ(2, *ptrChar);
   EXPECT_EQ(1, *ptrUInt8);
   EXPECT_EQ(127, *ptrInt16Split);
   EXPECT_EQ(1, *ptrUInt16Split);
   EXPECT_EQ(127, *ptrInt32Split);
   EXPECT_EQ(1, *ptrUInt32Split);
   EXPECT_EQ(127, *ptrInt64Split);
   EXPECT_EQ(1, *ptrUInt64Split);
   EXPECT_EQ(127, *ptrInt16Unsplit);
   EXPECT_EQ(1, *ptrUInt16Unsplit);
   EXPECT_EQ(127, *ptrInt32Unsplit);
   EXPECT_EQ(1, *ptrUInt32Unsplit);
   EXPECT_EQ(127, *ptrInt64Unsplit);
   EXPECT_EQ(1, *ptrUInt64Unsplit);
   reader->LoadEntry(1);
   EXPECT_EQ(0, *ptrBool);
   EXPECT_EQ(127, *ptrChar);
   EXPECT_EQ(127, *ptrUInt8);
   EXPECT_EQ(-128, *ptrInt16Split);
   EXPECT_EQ(127, *ptrUInt16Split);
   EXPECT_EQ(-128, *ptrInt32Split);
   EXPECT_EQ(127, *ptrUInt32Split);
   EXPECT_EQ(-128, *ptrInt64Split);
   EXPECT_EQ(127, *ptrUInt64Split);
   EXPECT_EQ(-128, *ptrInt16Unsplit);
   EXPECT_EQ(127, *ptrUInt16Unsplit);
   EXPECT_EQ(-128, *ptrInt32Unsplit);
   EXPECT_EQ(127, *ptrUInt32Unsplit);
   EXPECT_EQ(-128, *ptrInt64Unsplit);
   EXPECT_EQ(127, *ptrUInt64Unsplit);
   try {
      reader->LoadEntry(2);
      FAIL() << "value out of range should throw";
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("value out of range"));
   }
}

TEST(RNTuple, TypeCastUInt8)
{
   FileRaii fileGuard("test_ntuple_type_cast_UInt8.root");

   {
      auto model = RNTupleModel::Create();
      auto ptrBool = model->MakeField<bool>("bool");
      auto ptrChar = model->MakeField<char>("char");
      auto ptrInt8 = model->MakeField<std::int8_t>("int8");

      auto ptrInt16Split = MakeField<std::int16_t>("int16_split", EColumnType::kSplitInt16, *model);
      auto ptrUInt16Split = MakeField<std::uint16_t>("uint16_split", EColumnType::kSplitUInt16, *model);
      auto ptrInt32Split = MakeField<std::int32_t>("int32_split", EColumnType::kSplitInt32, *model);
      auto ptrUInt32Split = MakeField<std::uint32_t>("uint32_split", EColumnType::kSplitUInt32, *model);
      auto ptrInt64Split = MakeField<std::int64_t>("int64_split", EColumnType::kSplitInt64, *model);
      auto ptrUInt64Split = MakeField<std::uint64_t>("uint64_split", EColumnType::kSplitUInt64, *model);

      auto ptrInt16Unsplit = MakeField<std::int16_t>("int16_unsplit", EColumnType::kInt16, *model);
      auto ptrUInt16Unsplit = MakeField<std::uint16_t>("uint16_unsplit", EColumnType::kUInt16, *model);
      auto ptrInt32Unsplit = MakeField<std::int32_t>("int32_unsplit", EColumnType::kInt32, *model);
      auto ptrUInt32Unsplit = MakeField<std::uint32_t>("uint32_unsplit", EColumnType::kUInt32, *model);
      auto ptrInt64Unsplit = MakeField<std::int64_t>("int64_unsplit", EColumnType::kInt64, *model);
      auto ptrUInt64Unsplit = MakeField<std::uint64_t>("uint64_unsplit", EColumnType::kUInt64, *model);

      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
      *ptrBool = true;
      *ptrChar = 2;
      *ptrInt8 = 3;
      *ptrInt16Split = 4;
      *ptrUInt16Split = 5;
      *ptrInt32Split = 6;
      *ptrUInt32Split = 7;
      *ptrInt64Split = 8;
      *ptrUInt64Split = 9;
      *ptrInt16Unsplit = 10;
      *ptrUInt16Unsplit = 11;
      *ptrInt32Unsplit = 12;
      *ptrUInt32Unsplit = 13;
      *ptrInt64Unsplit = 14;
      *ptrUInt64Unsplit = 15;
      writer->Fill();
      *ptrBool = false;
      *ptrChar = 127;
      *ptrInt8 = 126;
      *ptrInt16Split = 255;
      *ptrUInt16Split = 254;
      *ptrInt32Split = 253;
      *ptrUInt32Split = 252;
      *ptrInt64Split = 251;
      *ptrUInt64Split = 250;
      *ptrInt16Unsplit = 249;
      *ptrUInt16Unsplit = 248;
      *ptrInt32Unsplit = 247;
      *ptrUInt32Unsplit = 246;
      *ptrInt64Unsplit = 245;
      *ptrUInt64Unsplit = 244;
      writer->Fill();
      writer->CommitCluster();
      *ptrInt16Split = -313;
      writer->Fill();
   }

   auto castModel = RNTupleModel::Create();
   auto ptrBool = castModel->MakeField<std::uint8_t>("bool");
   auto ptrChar = castModel->MakeField<std::uint8_t>("char");
   auto ptrInt8 = castModel->MakeField<std::uint8_t>("int8");
   auto ptrInt16Split = castModel->MakeField<std::uint8_t>("int16_split");
   auto ptrUInt16Split = castModel->MakeField<std::uint8_t>("uint16_split");
   auto ptrInt32Split = castModel->MakeField<std::uint8_t>("int32_split");
   auto ptrUInt32Split = castModel->MakeField<std::uint8_t>("uint32_split");
   auto ptrInt64Split = castModel->MakeField<std::uint8_t>("int64_split");
   auto ptrUInt64Split = castModel->MakeField<std::uint8_t>("uint64_split");
   auto ptrInt16Unsplit = castModel->MakeField<std::uint8_t>("int16_unsplit");
   auto ptrUInt16Unsplit = castModel->MakeField<std::uint8_t>("uint16_unsplit");
   auto ptrInt32Unsplit = castModel->MakeField<std::uint8_t>("int32_unsplit");
   auto ptrUInt32Unsplit = castModel->MakeField<std::uint8_t>("uint32_unsplit");
   auto ptrInt64Unsplit = castModel->MakeField<std::uint8_t>("int64_unsplit");
   auto ptrUInt64Unsplit = castModel->MakeField<std::uint8_t>("uint64_unsplit");
   auto reader = RNTupleReader::Open(std::move(castModel), "ntpl", fileGuard.GetPath());
   reader->LoadEntry(0);
   EXPECT_EQ(1, *ptrBool);
   EXPECT_EQ(2, *ptrChar);
   EXPECT_EQ(3, *ptrInt8);
   EXPECT_EQ(4, *ptrInt16Split);
   EXPECT_EQ(5, *ptrUInt16Split);
   EXPECT_EQ(6, *ptrInt32Split);
   EXPECT_EQ(7, *ptrUInt32Split);
   EXPECT_EQ(8, *ptrInt64Split);
   EXPECT_EQ(9, *ptrUInt64Split);
   EXPECT_EQ(10, *ptrInt16Unsplit);
   EXPECT_EQ(11, *ptrUInt16Unsplit);
   EXPECT_EQ(12, *ptrInt32Unsplit);
   EXPECT_EQ(13, *ptrUInt32Unsplit);
   EXPECT_EQ(14, *ptrInt64Unsplit);
   EXPECT_EQ(15, *ptrUInt64Unsplit);
   reader->LoadEntry(1);
   EXPECT_EQ(0, *ptrBool);
   EXPECT_EQ(127, *ptrChar);
   EXPECT_EQ(126, *ptrInt8);
   EXPECT_EQ(255, *ptrInt16Split);
   EXPECT_EQ(254, *ptrUInt16Split);
   EXPECT_EQ(253, *ptrInt32Split);
   EXPECT_EQ(252, *ptrUInt32Split);
   EXPECT_EQ(251, *ptrInt64Split);
   EXPECT_EQ(250, *ptrUInt64Split);
   EXPECT_EQ(249, *ptrInt16Unsplit);
   EXPECT_EQ(248, *ptrUInt16Unsplit);
   EXPECT_EQ(247, *ptrInt32Unsplit);
   EXPECT_EQ(246, *ptrUInt32Unsplit);
   EXPECT_EQ(245, *ptrInt64Unsplit);
   EXPECT_EQ(244, *ptrUInt64Unsplit);
   try {
      reader->LoadEntry(2);
      FAIL() << "value out of range should throw";
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("value out of range"));
   }
}

TEST(RNTuple, TypeCastInt16)
{
   FileRaii fileGuard("test_ntuple_type_cast_Int16.root");

   {
      auto model = RNTupleModel::Create();
      auto ptrBool = model->MakeField<bool>("bool");
      auto ptrChar = model->MakeField<char>("char");
      auto ptrInt8 = model->MakeField<std::int8_t>("int8");
      auto ptrUInt8 = model->MakeField<std::uint8_t>("uint8");

      auto ptrUInt16Split = MakeField<std::uint16_t>("uint16_split", EColumnType::kSplitUInt16, *model);
      auto ptrInt32Split = MakeField<std::int32_t>("int32_split", EColumnType::kSplitInt32, *model);
      auto ptrUInt32Split = MakeField<std::uint32_t>("uint32_split", EColumnType::kSplitUInt32, *model);
      auto ptrInt64Split = MakeField<std::int64_t>("int64_split", EColumnType::kSplitInt64, *model);
      auto ptrUInt64Split = MakeField<std::uint64_t>("uint64_split", EColumnType::kSplitUInt64, *model);

      auto ptrUInt16Unsplit = MakeField<std::uint16_t>("uint16_unsplit", EColumnType::kUInt16, *model);
      auto ptrInt32Unsplit = MakeField<std::int32_t>("int32_unsplit", EColumnType::kInt32, *model);
      auto ptrUInt32Unsplit = MakeField<std::uint32_t>("uint32_unsplit", EColumnType::kUInt32, *model);
      auto ptrInt64Unsplit = MakeField<std::int64_t>("int64_unsplit", EColumnType::kInt64, *model);
      auto ptrUInt64Unsplit = MakeField<std::uint64_t>("uint64_unsplit", EColumnType::kUInt64, *model);

      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
      *ptrBool = true;
      *ptrChar = 0;
      *ptrInt8 = 127;
      *ptrUInt8 = 1;
      *ptrUInt16Split = 1;
      *ptrInt32Split = -1;
      *ptrUInt32Split = 2;
      *ptrInt64Split = -2;
      *ptrUInt64Split = 3;
      *ptrUInt16Unsplit = 4;
      *ptrInt32Unsplit = -3;
      *ptrUInt32Unsplit = 5;
      *ptrInt64Unsplit = -4;
      *ptrUInt64Unsplit = 6;
      writer->Fill();
      *ptrBool = false;
      *ptrChar = 127;
      *ptrInt8 = -128;
      *ptrUInt8 = 127;
      *ptrUInt16Split = std::numeric_limits<std::int16_t>::max();
      *ptrInt32Split = std::numeric_limits<std::int16_t>::min();
      *ptrUInt32Split = std::numeric_limits<std::int16_t>::max() - 1;
      *ptrInt64Split = std::numeric_limits<std::int16_t>::min() + 1;
      *ptrUInt64Split = std::numeric_limits<std::int16_t>::max() - 2;
      *ptrUInt16Unsplit = std::numeric_limits<std::int16_t>::max() - 3;
      *ptrInt32Unsplit = std::numeric_limits<std::int16_t>::min() + 2;
      *ptrUInt32Unsplit = std::numeric_limits<std::int16_t>::max() - 4;
      *ptrInt64Unsplit = std::numeric_limits<std::int16_t>::min() + 3;
      *ptrUInt64Unsplit = std::numeric_limits<std::int16_t>::max() - 5;
      writer->Fill();
      writer->CommitCluster();
      *ptrInt32Split = 68000;
      writer->Fill();
   }

   auto castModel = RNTupleModel::Create();
   auto ptrBool = castModel->MakeField<std::int16_t>("bool");
   auto ptrChar = castModel->MakeField<std::int16_t>("char");
   auto ptrInt8 = castModel->MakeField<std::int16_t>("int8");
   auto ptrUInt8 = castModel->MakeField<std::int16_t>("uint8");
   auto ptrUInt16Split = castModel->MakeField<std::int16_t>("uint16_split");
   auto ptrInt32Split = castModel->MakeField<std::int16_t>("int32_split");
   auto ptrUInt32Split = castModel->MakeField<std::int16_t>("uint32_split");
   auto ptrInt64Split = castModel->MakeField<std::int16_t>("int64_split");
   auto ptrUInt64Split = castModel->MakeField<std::int16_t>("uint64_split");
   auto ptrUInt16Unsplit = castModel->MakeField<std::int16_t>("uint16_unsplit");
   auto ptrInt32Unsplit = castModel->MakeField<std::int16_t>("int32_unsplit");
   auto ptrUInt32Unsplit = castModel->MakeField<std::int16_t>("uint32_unsplit");
   auto ptrInt64Unsplit = castModel->MakeField<std::int16_t>("int64_unsplit");
   auto ptrUInt64Unsplit = castModel->MakeField<std::int16_t>("uint64_unsplit");
   auto reader = RNTupleReader::Open(std::move(castModel), "ntpl", fileGuard.GetPath());
   reader->LoadEntry(0);
   EXPECT_EQ(1, *ptrBool);
   EXPECT_EQ(0, *ptrChar);
   EXPECT_EQ(127, *ptrInt8);
   EXPECT_EQ(1, *ptrUInt8);
   EXPECT_EQ(1, *ptrUInt16Split);
   EXPECT_EQ(-1, *ptrInt32Split);
   EXPECT_EQ(2, *ptrUInt32Split);
   EXPECT_EQ(-2, *ptrInt64Split);
   EXPECT_EQ(3, *ptrUInt64Split);
   EXPECT_EQ(4, *ptrUInt16Unsplit);
   EXPECT_EQ(-3, *ptrInt32Unsplit);
   EXPECT_EQ(5, *ptrUInt32Unsplit);
   EXPECT_EQ(-4, *ptrInt64Unsplit);
   EXPECT_EQ(6, *ptrUInt64Unsplit);
   reader->LoadEntry(1);
   EXPECT_EQ(0, *ptrBool);
   EXPECT_EQ(127, *ptrChar);
   EXPECT_EQ(-128, *ptrInt8);
   EXPECT_EQ(127, *ptrUInt8);
   EXPECT_EQ(std::numeric_limits<std::int16_t>::max(), *ptrUInt16Split);
   EXPECT_EQ(std::numeric_limits<std::int16_t>::min(), *ptrInt32Split);
   EXPECT_EQ(std::numeric_limits<std::int16_t>::max() - 1, *ptrUInt32Split);
   EXPECT_EQ(std::numeric_limits<std::int16_t>::min() + 1, *ptrInt64Split);
   EXPECT_EQ(std::numeric_limits<std::int16_t>::max() - 2, *ptrUInt64Split);
   EXPECT_EQ(std::numeric_limits<std::int16_t>::max() - 3, *ptrUInt16Unsplit);
   EXPECT_EQ(std::numeric_limits<std::int16_t>::min() + 2, *ptrInt32Unsplit);
   EXPECT_EQ(std::numeric_limits<std::int16_t>::max() - 4, *ptrUInt32Unsplit);
   EXPECT_EQ(std::numeric_limits<std::int16_t>::min() + 3, *ptrInt64Unsplit);
   EXPECT_EQ(std::numeric_limits<std::int16_t>::max() - 5, *ptrUInt64Unsplit);
   try {
      reader->LoadEntry(2);
      FAIL() << "value out of range should throw";
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("value out of range"));
   }
}

TEST(RNTuple, TypeCastUInt16)
{
   FileRaii fileGuard("test_ntuple_type_cast_UInt16.root");

   {
      auto model = RNTupleModel::Create();
      auto ptrBool = model->MakeField<bool>("bool");
      auto ptrChar = model->MakeField<char>("char");
      auto ptrInt8 = model->MakeField<std::int8_t>("int8");
      auto ptrUInt8 = model->MakeField<std::uint8_t>("uint8");

      auto ptrInt16Split = MakeField<std::int16_t>("int16_split", EColumnType::kSplitInt16, *model);
      auto ptrInt32Split = MakeField<std::int32_t>("int32_split", EColumnType::kSplitInt32, *model);
      auto ptrUInt32Split = MakeField<std::uint32_t>("uint32_split", EColumnType::kSplitUInt32, *model);
      auto ptrInt64Split = MakeField<std::int64_t>("int64_split", EColumnType::kSplitInt64, *model);
      auto ptrUInt64Split = MakeField<std::uint64_t>("uint64_split", EColumnType::kSplitUInt64, *model);

      auto ptrInt16Unsplit = MakeField<std::int16_t>("int16_unsplit", EColumnType::kInt16, *model);
      auto ptrInt32Unsplit = MakeField<std::int32_t>("int32_unsplit", EColumnType::kInt32, *model);
      auto ptrUInt32Unsplit = MakeField<std::uint32_t>("uint32_unsplit", EColumnType::kUInt32, *model);
      auto ptrInt64Unsplit = MakeField<std::int64_t>("int64_unsplit", EColumnType::kInt64, *model);
      auto ptrUInt64Unsplit = MakeField<std::uint64_t>("uint64_unsplit", EColumnType::kUInt64, *model);

      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
      *ptrBool = true;
      *ptrChar = 1;
      *ptrInt8 = 2;
      *ptrUInt8 = 3;
      *ptrInt16Split = 4;
      *ptrInt32Split = 5;
      *ptrUInt32Split = 6;
      *ptrInt64Split = 7;
      *ptrUInt64Split = 8;
      *ptrInt16Unsplit = 9;
      *ptrInt32Unsplit = 10;
      *ptrUInt32Unsplit = 11;
      *ptrInt64Unsplit = 12;
      *ptrUInt64Unsplit = 13;
      writer->Fill();
      *ptrBool = false;
      *ptrChar = 127;
      *ptrInt8 = 127;
      *ptrUInt8 = 255;
      *ptrInt16Split = std::numeric_limits<std::int16_t>::max();
      *ptrInt32Split = std::numeric_limits<std::uint16_t>::max();
      *ptrUInt32Split = std::numeric_limits<std::uint16_t>::max() - 1;
      *ptrInt64Split = std::numeric_limits<std::uint16_t>::max() - 2;
      *ptrUInt64Split = std::numeric_limits<std::uint16_t>::max() - 3;
      *ptrInt16Unsplit = std::numeric_limits<std::int16_t>::max();
      *ptrInt32Unsplit = std::numeric_limits<std::uint16_t>::max() - 4;
      *ptrUInt32Unsplit = std::numeric_limits<std::uint16_t>::max() - 5;
      *ptrInt64Unsplit = std::numeric_limits<std::uint16_t>::max() - 6;
      *ptrUInt64Unsplit = std::numeric_limits<std::uint16_t>::max() - 7;
      writer->Fill();
      writer->CommitCluster();
      *ptrInt8 = -1;
      writer->Fill();
   }

   auto castModel = RNTupleModel::Create();
   auto ptrBool = castModel->MakeField<std::uint16_t>("bool");
   auto ptrChar = castModel->MakeField<std::uint16_t>("char");
   auto ptrInt8 = castModel->MakeField<std::uint16_t>("int8");
   auto ptrUInt8 = castModel->MakeField<std::uint16_t>("uint8");
   auto ptrInt16Split = castModel->MakeField<std::uint16_t>("int16_split");
   auto ptrInt32Split = castModel->MakeField<std::uint16_t>("int32_split");
   auto ptrUInt32Split = castModel->MakeField<std::uint16_t>("uint32_split");
   auto ptrInt64Split = castModel->MakeField<std::uint16_t>("int64_split");
   auto ptrUInt64Split = castModel->MakeField<std::uint16_t>("uint64_split");
   auto ptrInt16Unsplit = castModel->MakeField<std::uint16_t>("int16_unsplit");
   auto ptrInt32Unsplit = castModel->MakeField<std::uint16_t>("int32_unsplit");
   auto ptrUInt32Unsplit = castModel->MakeField<std::uint16_t>("uint32_unsplit");
   auto ptrInt64Unsplit = castModel->MakeField<std::uint16_t>("int64_unsplit");
   auto ptrUInt64Unsplit = castModel->MakeField<std::uint16_t>("uint64_unsplit");
   auto reader = RNTupleReader::Open(std::move(castModel), "ntpl", fileGuard.GetPath());
   reader->LoadEntry(0);
   EXPECT_EQ(1, *ptrBool);
   EXPECT_EQ(1, *ptrChar);
   EXPECT_EQ(2, *ptrInt8);
   EXPECT_EQ(3, *ptrUInt8);
   EXPECT_EQ(4, *ptrInt16Split);
   EXPECT_EQ(5, *ptrInt32Split);
   EXPECT_EQ(6, *ptrUInt32Split);
   EXPECT_EQ(7, *ptrInt64Split);
   EXPECT_EQ(8, *ptrUInt64Split);
   EXPECT_EQ(9, *ptrInt16Unsplit);
   EXPECT_EQ(10, *ptrInt32Unsplit);
   EXPECT_EQ(11, *ptrUInt32Unsplit);
   EXPECT_EQ(12, *ptrInt64Unsplit);
   EXPECT_EQ(13, *ptrUInt64Unsplit);
   reader->LoadEntry(1);
   EXPECT_EQ(0, *ptrBool);
   EXPECT_EQ(127, *ptrChar);
   EXPECT_EQ(127, *ptrInt8);
   EXPECT_EQ(255, *ptrUInt8);
   EXPECT_EQ(std::numeric_limits<std::int16_t>::max(), *ptrInt16Split);
   EXPECT_EQ(std::numeric_limits<std::uint16_t>::max(), *ptrInt32Split);
   EXPECT_EQ(std::numeric_limits<std::uint16_t>::max() - 1, *ptrUInt32Split);
   EXPECT_EQ(std::numeric_limits<std::uint16_t>::max() - 2, *ptrInt64Split);
   EXPECT_EQ(std::numeric_limits<std::uint16_t>::max() - 3, *ptrUInt64Split);
   EXPECT_EQ(std::numeric_limits<std::int16_t>::max(), *ptrInt16Unsplit);
   EXPECT_EQ(std::numeric_limits<std::uint16_t>::max() - 4, *ptrInt32Unsplit);
   EXPECT_EQ(std::numeric_limits<std::uint16_t>::max() - 5, *ptrUInt32Unsplit);
   EXPECT_EQ(std::numeric_limits<std::uint16_t>::max() - 6, *ptrInt64Unsplit);
   EXPECT_EQ(std::numeric_limits<std::uint16_t>::max() - 7, *ptrUInt64Unsplit);
   try {
      reader->LoadEntry(2);
      FAIL() << "value out of range should throw";
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("value out of range"));
   }
}

TEST(RNTuple, TypeCastInt32)
{
   FileRaii fileGuard("test_ntuple_type_cast_Int32.root");

   {
      auto model = RNTupleModel::Create();
      auto ptrBool = model->MakeField<bool>("bool");
      auto ptrChar = model->MakeField<char>("char");
      auto ptrInt8 = model->MakeField<std::int8_t>("int8");
      auto ptrUInt8 = model->MakeField<std::uint8_t>("uint8");

      auto ptrInt16Split = MakeField<std::int16_t>("int16_split", EColumnType::kSplitInt16, *model);
      auto ptrUInt16Split = MakeField<std::uint16_t>("uint16_split", EColumnType::kSplitUInt16, *model);
      auto ptrUInt32Split = MakeField<std::uint32_t>("uint32_split", EColumnType::kSplitUInt32, *model);
      auto ptrInt64Split = MakeField<std::int64_t>("int64_split", EColumnType::kSplitInt64, *model);
      auto ptrUInt64Split = MakeField<std::uint64_t>("uint64_split", EColumnType::kSplitUInt64, *model);

      auto ptrInt16Unsplit = MakeField<std::int16_t>("int16_unsplit", EColumnType::kInt16, *model);
      auto ptrUInt16Unsplit = MakeField<std::uint16_t>("uint16_unsplit", EColumnType::kUInt16, *model);
      auto ptrUInt32Unsplit = MakeField<std::uint32_t>("uint32_unsplit", EColumnType::kUInt32, *model);
      auto ptrInt64Unsplit = MakeField<std::int64_t>("int64_unsplit", EColumnType::kInt64, *model);
      auto ptrUInt64Unsplit = MakeField<std::uint64_t>("uint64_unsplit", EColumnType::kUInt64, *model);

      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
      *ptrBool = true;
      *ptrChar = 0;
      *ptrInt8 = 127;
      *ptrUInt8 = 1;
      *ptrInt16Split = -1;
      *ptrUInt16Split = 1;
      *ptrUInt32Split = 2;
      *ptrInt64Split = -2;
      *ptrUInt64Split = 3;
      *ptrInt16Unsplit = -3;
      *ptrUInt16Unsplit = 4;
      *ptrUInt32Unsplit = 5;
      *ptrInt64Unsplit = -4;
      *ptrUInt64Unsplit = 6;
      writer->Fill();
      *ptrBool = false;
      *ptrChar = 127;
      *ptrInt8 = -128;
      *ptrUInt8 = 127;
      *ptrInt16Split = std::numeric_limits<std::int16_t>::min();
      *ptrUInt16Split = std::numeric_limits<std::uint16_t>::max();
      *ptrUInt32Split = std::numeric_limits<std::int32_t>::max();
      *ptrInt64Split = std::numeric_limits<std::int32_t>::min();
      *ptrUInt64Split = std::numeric_limits<std::int32_t>::max() - 1;
      *ptrInt16Unsplit = std::numeric_limits<std::int16_t>::min();
      *ptrUInt16Unsplit = std::numeric_limits<std::uint16_t>::max();
      *ptrUInt32Unsplit = std::numeric_limits<std::int32_t>::max() - 2;
      *ptrInt64Unsplit = std::numeric_limits<std::int32_t>::min() + 1;
      *ptrUInt64Unsplit = std::numeric_limits<std::int32_t>::max() - 3;
      writer->Fill();
      writer->CommitCluster();
      *ptrUInt32Unsplit = std::numeric_limits<std::uint32_t>::max();
      writer->Fill();
   }

   auto castModel = RNTupleModel::Create();
   auto ptrBool = castModel->MakeField<std::int32_t>("bool");
   auto ptrChar = castModel->MakeField<std::int32_t>("char");
   auto ptrInt8 = castModel->MakeField<std::int32_t>("int8");
   auto ptrUInt8 = castModel->MakeField<std::int32_t>("uint8");
   auto ptrInt16Split = castModel->MakeField<std::int32_t>("int16_split");
   auto ptrUInt16Split = castModel->MakeField<std::int32_t>("uint16_split");
   auto ptrUInt32Split = castModel->MakeField<std::int32_t>("uint32_split");
   auto ptrInt64Split = castModel->MakeField<std::int32_t>("int64_split");
   auto ptrUInt64Split = castModel->MakeField<std::int32_t>("uint64_split");
   auto ptrInt16Unsplit = castModel->MakeField<std::int32_t>("int16_unsplit");
   auto ptrUInt16Unsplit = castModel->MakeField<std::int32_t>("uint16_unsplit");
   auto ptrUInt32Unsplit = castModel->MakeField<std::int32_t>("uint32_unsplit");
   auto ptrInt64Unsplit = castModel->MakeField<std::int32_t>("int64_unsplit");
   auto ptrUInt64Unsplit = castModel->MakeField<std::int32_t>("uint64_unsplit");
   auto reader = RNTupleReader::Open(std::move(castModel), "ntpl", fileGuard.GetPath());
   reader->LoadEntry(0);
   EXPECT_EQ(1, *ptrBool);
   EXPECT_EQ(0, *ptrChar);
   EXPECT_EQ(127, *ptrInt8);
   EXPECT_EQ(1, *ptrUInt8);
   EXPECT_EQ(-1, *ptrInt16Split);
   EXPECT_EQ(1, *ptrUInt16Split);
   EXPECT_EQ(2, *ptrUInt32Split);
   EXPECT_EQ(-2, *ptrInt64Split);
   EXPECT_EQ(3, *ptrUInt64Split);
   EXPECT_EQ(-3, *ptrInt16Unsplit);
   EXPECT_EQ(4, *ptrUInt16Unsplit);
   EXPECT_EQ(5, *ptrUInt32Unsplit);
   EXPECT_EQ(-4, *ptrInt64Unsplit);
   EXPECT_EQ(6, *ptrUInt64Unsplit);
   reader->LoadEntry(1);
   EXPECT_EQ(0, *ptrBool);
   EXPECT_EQ(127, *ptrChar);
   EXPECT_EQ(-128, *ptrInt8);
   EXPECT_EQ(127, *ptrUInt8);
   EXPECT_EQ(std::numeric_limits<std::int16_t>::min(), *ptrInt16Split);
   EXPECT_EQ(std::numeric_limits<std::uint16_t>::max(), *ptrUInt16Split);
   EXPECT_EQ(std::numeric_limits<std::int32_t>::max(), *ptrUInt32Split);
   EXPECT_EQ(std::numeric_limits<std::int32_t>::min(), *ptrInt64Split);
   EXPECT_EQ(std::numeric_limits<std::int32_t>::max() - 1, *ptrUInt64Split);
   EXPECT_EQ(std::numeric_limits<std::int16_t>::min(), *ptrInt16Unsplit);
   EXPECT_EQ(std::numeric_limits<std::uint16_t>::max(), *ptrUInt16Unsplit);
   EXPECT_EQ(std::numeric_limits<std::int32_t>::max() - 2, *ptrUInt32Unsplit);
   EXPECT_EQ(std::numeric_limits<std::int32_t>::min() + 1, *ptrInt64Unsplit);
   EXPECT_EQ(std::numeric_limits<std::int32_t>::max() - 3, *ptrUInt64Unsplit);
   try {
      reader->LoadEntry(2);
      FAIL() << "value out of range should throw";
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("value out of range"));
   }
}

TEST(RNTuple, TypeCastUInt32)
{
   FileRaii fileGuard("test_ntuple_type_cast_UInt32.root");

   {
      auto model = RNTupleModel::Create();
      auto ptrBool = model->MakeField<bool>("bool");
      auto ptrChar = model->MakeField<char>("char");
      auto ptrInt8 = model->MakeField<std::int8_t>("int8");
      auto ptrUInt8 = model->MakeField<std::uint8_t>("uint8");

      auto ptrInt16Split = MakeField<std::int16_t>("int16_split", EColumnType::kSplitInt16, *model);
      auto ptrUInt16Split = MakeField<std::uint16_t>("uint16_split", EColumnType::kSplitUInt16, *model);
      auto ptrInt32Split = MakeField<std::int32_t>("int32_split", EColumnType::kSplitInt32, *model);
      auto ptrInt64Split = MakeField<std::int64_t>("int64_split", EColumnType::kSplitInt64, *model);
      auto ptrUInt64Split = MakeField<std::uint64_t>("uint64_split", EColumnType::kSplitUInt64, *model);

      auto ptrInt16Unsplit = MakeField<std::int16_t>("int16_unsplit", EColumnType::kInt16, *model);
      auto ptrUInt16Unsplit = MakeField<std::uint16_t>("uint16_unsplit", EColumnType::kUInt16, *model);
      auto ptrInt32Unsplit = MakeField<std::int32_t>("int32_unsplit", EColumnType::kInt32, *model);
      auto ptrInt64Unsplit = MakeField<std::int64_t>("int64_unsplit", EColumnType::kInt64, *model);
      auto ptrUInt64Unsplit = MakeField<std::uint64_t>("uint64_unsplit", EColumnType::kUInt64, *model);

      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
      *ptrBool = true;
      *ptrChar = 1;
      *ptrInt8 = 2;
      *ptrUInt8 = 3;
      *ptrInt16Split = 4;
      *ptrUInt16Split = 5;
      *ptrInt32Split = 6;
      *ptrInt64Split = 7;
      *ptrUInt64Split = 8;
      *ptrInt16Unsplit = 9;
      *ptrUInt16Unsplit = 10;
      *ptrInt32Unsplit = 11;
      *ptrInt64Unsplit = 12;
      *ptrUInt64Unsplit = 13;
      writer->Fill();
      *ptrBool = false;
      *ptrChar = 127;
      *ptrInt8 = 127;
      *ptrUInt8 = 255;
      *ptrInt16Split = std::numeric_limits<std::int16_t>::max();
      *ptrUInt16Split = std::numeric_limits<std::uint16_t>::max();
      *ptrInt32Split = std::numeric_limits<std::int32_t>::max();
      *ptrInt64Split = std::numeric_limits<std::uint32_t>::max();
      *ptrUInt64Split = std::numeric_limits<std::uint32_t>::max() - 1;
      *ptrInt16Unsplit = std::numeric_limits<std::int16_t>::max();
      *ptrUInt16Unsplit = std::numeric_limits<std::uint16_t>::max();
      *ptrInt32Unsplit = std::numeric_limits<std::int32_t>::max();
      *ptrInt64Unsplit = std::numeric_limits<std::uint32_t>::max() - 2;
      *ptrUInt64Unsplit = std::numeric_limits<std::uint32_t>::max() - 3;
      writer->Fill();
      writer->CommitCluster();
      *ptrInt64Split = std::numeric_limits<std::int64_t>::max();
      writer->Fill();
   }

   auto castModel = RNTupleModel::Create();
   auto ptrBool = castModel->MakeField<std::uint32_t>("bool");
   auto ptrChar = castModel->MakeField<std::uint32_t>("char");
   auto ptrInt8 = castModel->MakeField<std::uint32_t>("int8");
   auto ptrUInt8 = castModel->MakeField<std::uint32_t>("uint8");
   auto ptrInt16Split = castModel->MakeField<std::uint32_t>("int16_split");
   auto ptrUInt16Split = castModel->MakeField<std::uint32_t>("uint16_split");
   auto ptrInt32Split = castModel->MakeField<std::uint32_t>("int32_split");
   auto ptrInt64Split = castModel->MakeField<std::uint32_t>("int64_split");
   auto ptrUInt64Split = castModel->MakeField<std::uint32_t>("uint64_split");
   auto ptrInt16Unsplit = castModel->MakeField<std::uint32_t>("int16_unsplit");
   auto ptrUInt16Unsplit = castModel->MakeField<std::uint32_t>("uint16_unsplit");
   auto ptrInt32Unsplit = castModel->MakeField<std::uint32_t>("int32_unsplit");
   auto ptrInt64Unsplit = castModel->MakeField<std::uint32_t>("int64_unsplit");
   auto ptrUInt64Unsplit = castModel->MakeField<std::uint32_t>("uint64_unsplit");
   auto reader = RNTupleReader::Open(std::move(castModel), "ntpl", fileGuard.GetPath());
   reader->LoadEntry(0);
   EXPECT_EQ(1, *ptrBool);
   EXPECT_EQ(1, *ptrChar);
   EXPECT_EQ(2, *ptrInt8);
   EXPECT_EQ(3, *ptrUInt8);
   EXPECT_EQ(4, *ptrInt16Split);
   EXPECT_EQ(5, *ptrUInt16Split);
   EXPECT_EQ(6, *ptrInt32Split);
   EXPECT_EQ(7, *ptrInt64Split);
   EXPECT_EQ(8, *ptrUInt64Split);
   EXPECT_EQ(9, *ptrInt16Unsplit);
   EXPECT_EQ(10, *ptrUInt16Unsplit);
   EXPECT_EQ(11, *ptrInt32Unsplit);
   EXPECT_EQ(12, *ptrInt64Unsplit);
   EXPECT_EQ(13, *ptrUInt64Unsplit);
   reader->LoadEntry(1);
   EXPECT_EQ(0, *ptrBool);
   EXPECT_EQ(127, *ptrChar);
   EXPECT_EQ(127, *ptrInt8);
   EXPECT_EQ(255, *ptrUInt8);
   EXPECT_EQ(std::numeric_limits<std::int16_t>::max(), *ptrInt16Split);
   EXPECT_EQ(std::numeric_limits<std::uint16_t>::max(), *ptrUInt16Split);
   EXPECT_EQ(std::numeric_limits<std::int32_t>::max(), *ptrInt32Split);
   EXPECT_EQ(std::numeric_limits<std::uint32_t>::max(), *ptrInt64Split);
   EXPECT_EQ(std::numeric_limits<std::uint32_t>::max() - 1, *ptrUInt64Split);
   EXPECT_EQ(std::numeric_limits<std::int16_t>::max(), *ptrInt16Unsplit);
   EXPECT_EQ(std::numeric_limits<std::uint16_t>::max(), *ptrUInt16Unsplit);
   EXPECT_EQ(std::numeric_limits<std::int32_t>::max(), *ptrInt32Unsplit);
   EXPECT_EQ(std::numeric_limits<std::uint32_t>::max() - 2, *ptrInt64Unsplit);
   EXPECT_EQ(std::numeric_limits<std::uint32_t>::max() - 3, *ptrUInt64Unsplit);
   try {
      reader->LoadEntry(2);
      FAIL() << "value out of range should throw";
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("value out of range"));
   }
}

TEST(RNTuple, TypeCastInt64)
{
   FileRaii fileGuard("test_ntuple_type_cast_Int64.root");

   {
      auto model = RNTupleModel::Create();
      auto ptrBool = model->MakeField<bool>("bool");
      auto ptrChar = model->MakeField<char>("char");
      auto ptrInt8 = model->MakeField<std::int8_t>("int8");
      auto ptrUInt8 = model->MakeField<std::uint8_t>("uint8");

      auto ptrInt16Split = MakeField<std::int16_t>("int16_split", EColumnType::kSplitInt16, *model);
      auto ptrUInt16Split = MakeField<std::uint16_t>("uint16_split", EColumnType::kSplitUInt16, *model);
      auto ptrInt32Split = MakeField<std::int32_t>("int32_split", EColumnType::kSplitInt32, *model);
      auto ptrUInt32Split = MakeField<std::uint32_t>("uint32_split", EColumnType::kSplitUInt32, *model);
      auto ptrUInt64Split = MakeField<std::uint64_t>("uint64_split", EColumnType::kSplitUInt64, *model);

      auto ptrInt16Unsplit = MakeField<std::int16_t>("int16_unsplit", EColumnType::kInt16, *model);
      auto ptrUInt16Unsplit = MakeField<std::uint16_t>("uint16_unsplit", EColumnType::kUInt16, *model);
      auto ptrInt32Unsplit = MakeField<std::int32_t>("int32_unsplit", EColumnType::kInt32, *model);
      auto ptrUInt32Unsplit = MakeField<std::uint32_t>("uint32_unsplit", EColumnType::kUInt32, *model);
      auto ptrUInt64Unsplit = MakeField<std::uint64_t>("uint64_unsplit", EColumnType::kUInt64, *model);

      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
      *ptrBool = true;
      *ptrChar = 0;
      *ptrInt8 = 127;
      *ptrUInt8 = 1;
      *ptrInt16Split = -1;
      *ptrUInt16Split = 1;
      *ptrInt32Split = -2;
      *ptrUInt32Split = 2;
      *ptrUInt64Split = 3;
      *ptrInt16Unsplit = -3;
      *ptrUInt16Unsplit = 4;
      *ptrInt32Unsplit = -4;
      *ptrUInt32Unsplit = 5;
      *ptrUInt64Unsplit = 6;
      writer->Fill();
      *ptrBool = false;
      *ptrChar = 127;
      *ptrInt8 = -128;
      *ptrUInt8 = 127;
      *ptrInt16Split = std::numeric_limits<std::int16_t>::min();
      *ptrUInt16Split = std::numeric_limits<std::uint16_t>::max();
      *ptrInt32Split = std::numeric_limits<std::int32_t>::min();
      *ptrUInt32Split = std::numeric_limits<std::uint32_t>::max();
      *ptrUInt64Split = std::numeric_limits<std::int64_t>::max();
      *ptrInt16Unsplit = std::numeric_limits<std::int16_t>::min();
      *ptrUInt16Unsplit = std::numeric_limits<std::uint16_t>::max();
      *ptrInt32Unsplit = std::numeric_limits<std::int32_t>::min();
      *ptrUInt32Unsplit = std::numeric_limits<std::uint32_t>::max();
      *ptrUInt64Unsplit = std::numeric_limits<std::int64_t>::max() - 1;
      writer->Fill();
      writer->CommitCluster();
      *ptrUInt64Unsplit = std::numeric_limits<std::uint64_t>::max();
      writer->Fill();
   }

   auto castModel = RNTupleModel::Create();
   auto ptrBool = castModel->MakeField<std::int64_t>("bool");
   auto ptrChar = castModel->MakeField<std::int64_t>("char");
   auto ptrInt8 = castModel->MakeField<std::int64_t>("int8");
   auto ptrUInt8 = castModel->MakeField<std::int64_t>("uint8");
   auto ptrInt16Split = castModel->MakeField<std::int64_t>("int16_split");
   auto ptrUInt16Split = castModel->MakeField<std::int64_t>("uint16_split");
   auto ptrInt32Split = castModel->MakeField<std::int64_t>("int32_split");
   auto ptrUInt32Split = castModel->MakeField<std::int64_t>("uint32_split");
   auto ptrUInt64Split = castModel->MakeField<std::int64_t>("uint64_split");
   auto ptrInt16Unsplit = castModel->MakeField<std::int64_t>("int16_unsplit");
   auto ptrUInt16Unsplit = castModel->MakeField<std::int64_t>("uint16_unsplit");
   auto ptrInt32Unsplit = castModel->MakeField<std::int64_t>("int32_unsplit");
   auto ptrUInt32Unsplit = castModel->MakeField<std::int64_t>("uint32_unsplit");
   auto ptrUInt64Unsplit = castModel->MakeField<std::int64_t>("uint64_unsplit");
   auto reader = RNTupleReader::Open(std::move(castModel), "ntpl", fileGuard.GetPath());
   reader->LoadEntry(0);
   EXPECT_EQ(1, *ptrBool);
   EXPECT_EQ(0, *ptrChar);
   EXPECT_EQ(127, *ptrInt8);
   EXPECT_EQ(1, *ptrUInt8);
   EXPECT_EQ(-1, *ptrInt16Split);
   EXPECT_EQ(1, *ptrUInt16Split);
   EXPECT_EQ(-2, *ptrInt32Split);
   EXPECT_EQ(2, *ptrUInt32Split);
   EXPECT_EQ(3, *ptrUInt64Split);
   EXPECT_EQ(-3, *ptrInt16Unsplit);
   EXPECT_EQ(4, *ptrUInt16Unsplit);
   EXPECT_EQ(-4, *ptrInt32Unsplit);
   EXPECT_EQ(5, *ptrUInt32Unsplit);
   EXPECT_EQ(6, *ptrUInt64Unsplit);
   reader->LoadEntry(1);
   EXPECT_EQ(0, *ptrBool);
   EXPECT_EQ(127, *ptrChar);
   EXPECT_EQ(-128, *ptrInt8);
   EXPECT_EQ(127, *ptrUInt8);
   EXPECT_EQ(std::numeric_limits<std::int16_t>::min(), *ptrInt16Split);
   EXPECT_EQ(std::numeric_limits<std::uint16_t>::max(), *ptrUInt16Split);
   EXPECT_EQ(std::numeric_limits<std::int32_t>::min(), *ptrInt32Split);
   EXPECT_EQ(std::numeric_limits<std::uint32_t>::max(), *ptrUInt32Split);
   EXPECT_EQ(std::numeric_limits<std::int64_t>::max(), *ptrUInt64Split);
   EXPECT_EQ(std::numeric_limits<std::int16_t>::min(), *ptrInt16Unsplit);
   EXPECT_EQ(std::numeric_limits<std::uint16_t>::max(), *ptrUInt16Unsplit);
   EXPECT_EQ(std::numeric_limits<std::int32_t>::min(), *ptrInt32Unsplit);
   EXPECT_EQ(std::numeric_limits<std::uint32_t>::max(), *ptrUInt32Unsplit);
   EXPECT_EQ(std::numeric_limits<std::int64_t>::max() - 1, *ptrUInt64Unsplit);
   try {
      reader->LoadEntry(2);
      FAIL() << "value out of range should throw";
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("value out of range"));
   }
}

TEST(RNTuple, TypeCastUInt64)
{
   FileRaii fileGuard("test_ntuple_type_cast_UInt64.root");

   {
      auto model = RNTupleModel::Create();
      auto ptrBool = model->MakeField<bool>("bool");
      auto ptrChar = model->MakeField<char>("char");
      auto ptrInt8 = model->MakeField<std::int8_t>("int8");
      auto ptrUInt8 = model->MakeField<std::uint8_t>("uint8");

      auto ptrInt16Split = MakeField<std::int16_t>("int16_split", EColumnType::kSplitInt16, *model);
      auto ptrUInt16Split = MakeField<std::uint16_t>("uint16_split", EColumnType::kSplitUInt16, *model);
      auto ptrInt32Split = MakeField<std::int32_t>("int32_split", EColumnType::kSplitInt32, *model);
      auto ptrUInt32Split = MakeField<std::uint32_t>("uint32_split", EColumnType::kSplitUInt32, *model);
      auto ptrInt64Split = MakeField<std::int64_t>("int64_split", EColumnType::kSplitInt64, *model);

      auto ptrInt16Unsplit = MakeField<std::int16_t>("int16_unsplit", EColumnType::kInt16, *model);
      auto ptrUInt16Unsplit = MakeField<std::uint16_t>("uint16_unsplit", EColumnType::kUInt16, *model);
      auto ptrInt32Unsplit = MakeField<std::int32_t>("int32_unsplit", EColumnType::kInt32, *model);
      auto ptrUInt32Unsplit = MakeField<std::uint32_t>("uint32_unsplit", EColumnType::kUInt32, *model);
      auto ptrInt64Unsplit = MakeField<std::int64_t>("int64_unsplit", EColumnType::kInt64, *model);

      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
      *ptrBool = true;
      *ptrChar = 1;
      *ptrInt8 = 2;
      *ptrUInt8 = 3;
      *ptrInt16Split = 4;
      *ptrUInt16Split = 5;
      *ptrInt32Split = 6;
      *ptrUInt32Split = 7;
      *ptrInt64Split = 8;
      *ptrInt16Unsplit = 9;
      *ptrUInt16Unsplit = 10;
      *ptrInt32Unsplit = 11;
      *ptrUInt32Unsplit = 12;
      *ptrInt64Unsplit = 13;
      writer->Fill();
      *ptrBool = false;
      *ptrChar = 127;
      *ptrInt8 = 127;
      *ptrUInt8 = 255;
      *ptrInt16Split = std::numeric_limits<std::int16_t>::max();
      *ptrUInt16Split = std::numeric_limits<std::uint16_t>::max();
      *ptrInt32Split = std::numeric_limits<std::int32_t>::max();
      *ptrUInt32Split = std::numeric_limits<std::uint32_t>::max();
      *ptrInt64Split = std::numeric_limits<std::int64_t>::max();
      *ptrInt16Unsplit = std::numeric_limits<std::int16_t>::max();
      *ptrUInt16Unsplit = std::numeric_limits<std::uint16_t>::max();
      *ptrInt32Unsplit = std::numeric_limits<std::int32_t>::max();
      *ptrUInt32Unsplit = std::numeric_limits<std::uint32_t>::max();
      *ptrInt64Unsplit = std::numeric_limits<std::int64_t>::max();
      writer->Fill();
      writer->CommitCluster();
      *ptrInt8 = -1;
      writer->Fill();
   }

   auto castModel = RNTupleModel::Create();
   auto ptrBool = castModel->MakeField<std::uint64_t>("bool");
   auto ptrChar = castModel->MakeField<std::uint64_t>("char");
   auto ptrInt8 = castModel->MakeField<std::uint64_t>("int8");
   auto ptrUInt8 = castModel->MakeField<std::uint64_t>("uint8");
   auto ptrInt16Split = castModel->MakeField<std::uint64_t>("int16_split");
   auto ptrUInt16Split = castModel->MakeField<std::uint64_t>("uint16_split");
   auto ptrInt32Split = castModel->MakeField<std::uint64_t>("int32_split");
   auto ptrUInt32Split = castModel->MakeField<std::uint64_t>("uint32_split");
   auto ptrInt64Split = castModel->MakeField<std::uint64_t>("int64_split");
   auto ptrInt16Unsplit = castModel->MakeField<std::uint64_t>("int16_unsplit");
   auto ptrUInt16Unsplit = castModel->MakeField<std::uint64_t>("uint16_unsplit");
   auto ptrInt32Unsplit = castModel->MakeField<std::uint64_t>("int32_unsplit");
   auto ptrUInt32Unsplit = castModel->MakeField<std::uint64_t>("uint32_unsplit");
   auto ptrInt64Unsplit = castModel->MakeField<std::uint64_t>("int64_unsplit");
   auto reader = RNTupleReader::Open(std::move(castModel), "ntpl", fileGuard.GetPath());
   reader->LoadEntry(0);
   EXPECT_EQ(1, *ptrBool);
   EXPECT_EQ(1, *ptrChar);
   EXPECT_EQ(2, *ptrInt8);
   EXPECT_EQ(3, *ptrUInt8);
   EXPECT_EQ(4, *ptrInt16Split);
   EXPECT_EQ(5, *ptrUInt16Split);
   EXPECT_EQ(6, *ptrInt32Split);
   EXPECT_EQ(7, *ptrUInt32Split);
   EXPECT_EQ(8, *ptrInt64Split);
   EXPECT_EQ(9, *ptrInt16Unsplit);
   EXPECT_EQ(10, *ptrUInt16Unsplit);
   EXPECT_EQ(11, *ptrInt32Unsplit);
   EXPECT_EQ(12, *ptrUInt32Unsplit);
   EXPECT_EQ(13, *ptrInt64Unsplit);
   reader->LoadEntry(1);
   EXPECT_EQ(0, *ptrBool);
   EXPECT_EQ(127, *ptrChar);
   EXPECT_EQ(127, *ptrInt8);
   EXPECT_EQ(255, *ptrUInt8);
   EXPECT_EQ(std::numeric_limits<std::int16_t>::max(), *ptrInt16Split);
   EXPECT_EQ(std::numeric_limits<std::uint16_t>::max(), *ptrUInt16Split);
   EXPECT_EQ(std::numeric_limits<std::int32_t>::max(), *ptrInt32Split);
   EXPECT_EQ(std::numeric_limits<std::uint32_t>::max(), *ptrUInt32Split);
   EXPECT_EQ(std::numeric_limits<std::int64_t>::max(), *ptrInt64Split);
   EXPECT_EQ(std::numeric_limits<std::int16_t>::max(), *ptrInt16Unsplit);
   EXPECT_EQ(std::numeric_limits<std::uint16_t>::max(), *ptrUInt16Unsplit);
   EXPECT_EQ(std::numeric_limits<std::int32_t>::max(), *ptrInt32Unsplit);
   EXPECT_EQ(std::numeric_limits<std::uint32_t>::max(), *ptrUInt32Unsplit);
   EXPECT_EQ(std::numeric_limits<std::int64_t>::max(), *ptrInt64Unsplit);
   try {
      reader->LoadEntry(2);
      FAIL() << "value out of range should throw";
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("value out of range"));
   }
}
