#include "ntuple_test.hxx"

#include <cstdint>
#include <utility>

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
   EXPECT_THROW(RNTupleReader::Open(std::move(castModelInvalid), "ntpl", fileGuard.GetPath()), RException);

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

      auto fieldInt16Unsplit = std::make_unique<RField<std::int16_t>>("int16_unsplit");
      fieldInt16Unsplit->SetColumnRepresentatives({{EColumnType::kInt16}});
      model->AddField(std::move(fieldInt16Unsplit));
      auto ptrInt16Unsplit = model->GetDefaultEntry().GetPtr<std::int16_t>("int16_unsplit");

      auto fieldUInt16Unsplit = std::make_unique<RField<std::uint16_t>>("uint16_unsplit");
      fieldUInt16Unsplit->SetColumnRepresentatives({{EColumnType::kUInt16}});
      model->AddField(std::move(fieldUInt16Unsplit));
      auto ptrUInt16Unsplit = model->GetDefaultEntry().GetPtr<std::uint16_t>("uint16_unsplit");

      auto fieldInt32Unsplit = std::make_unique<RField<std::int32_t>>("int32_unsplit");
      fieldInt32Unsplit->SetColumnRepresentatives({{EColumnType::kInt32}});
      model->AddField(std::move(fieldInt32Unsplit));
      auto ptrInt32Unsplit = model->GetDefaultEntry().GetPtr<std::int32_t>("int32_unsplit");

      auto fieldUInt32Unsplit = std::make_unique<RField<std::uint32_t>>("uint32_unsplit");
      fieldUInt32Unsplit->SetColumnRepresentatives({{EColumnType::kUInt32}});
      model->AddField(std::move(fieldUInt32Unsplit));
      auto ptrUInt32Unsplit = model->GetDefaultEntry().GetPtr<std::uint32_t>("uint32_unsplit");

      auto fieldInt64Unsplit = std::make_unique<RField<std::int64_t>>("int64_unsplit");
      fieldInt64Unsplit->SetColumnRepresentatives({{EColumnType::kInt64}});
      model->AddField(std::move(fieldInt64Unsplit));
      auto ptrInt64Unsplit = model->GetDefaultEntry().GetPtr<std::int64_t>("int64_unsplit");

      auto fieldUInt64Unsplit = std::make_unique<RField<std::uint64_t>>("uint64_unsplit");
      fieldUInt64Unsplit->SetColumnRepresentatives({{EColumnType::kUInt64}});
      model->AddField(std::move(fieldUInt64Unsplit));
      auto ptrUInt64Unsplit = model->GetDefaultEntry().GetPtr<std::uint64_t>("uint64_unsplit");

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
