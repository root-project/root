#include "ntuple_test.hxx"

#include <TDictAttributeMap.h>
#include <TVirtualStreamerInfo.h>

#include "Unsplit.hxx"
#include "UnsplitXML.h"

TEST(RField, UnsplitDirect)
{
   FileRaii fileGuard("test_ntuple_rfield_unsplit_direct.root");
   {
      auto model = RNTupleModel::Create();
      model->AddField(std::make_unique<ROOT::Experimental::RUnsplitField>("pt", "std::vector<float>"));
      auto ptrPt = model->GetDefaultEntry().GetPtr<std::vector<float>>("pt");
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
      ptrPt->push_back(1.0);
      writer->Fill();
   }

   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
   auto ptrPt = reader->GetModel().GetDefaultEntry().GetPtr<std::vector<float>>("pt");

   ASSERT_EQ(1U, reader->GetNEntries());
   reader->LoadEntry(0);
   EXPECT_EQ(1u, ptrPt->size());
   EXPECT_FLOAT_EQ(1.0, ptrPt->at(0));
}

TEST(RField, UnsplitMember)
{
   auto cl = TClass::GetClass("CyclicMember");
   cl->CreateAttributeMap();
   cl->GetAttributeMap()->AddProperty("rntuple.split", "false");

   FileRaii fileGuard("test_ntuple_rfield_unsplit_member.root");
   {
      auto model = RNTupleModel::Create();
      auto ptrClassWithUnsplitMember = model->MakeField<ClassWithUnsplitMember>("event");
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
      ptrClassWithUnsplitMember->fA = 1.0;
      CyclicMember inner;
      inner.fB = 3.0;
      ptrClassWithUnsplitMember->fUnsplit.fB = 2.0;
      ptrClassWithUnsplitMember->fUnsplit.fV.push_back(inner);
      writer->Fill();
   }

   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
   auto ptrClassWithUnsplitMember = reader->GetModel().GetDefaultEntry().GetPtr<ClassWithUnsplitMember>("event");

   ASSERT_EQ(1U, reader->GetNEntries());
   reader->LoadEntry(0);
   EXPECT_FLOAT_EQ(1.0, ptrClassWithUnsplitMember->fA);
   EXPECT_FLOAT_EQ(2.0, ptrClassWithUnsplitMember->fUnsplit.fB);
   EXPECT_EQ(1u, ptrClassWithUnsplitMember->fUnsplit.fV.size());
   EXPECT_FLOAT_EQ(3.0, ptrClassWithUnsplitMember->fUnsplit.fV.at(0).fB);
   EXPECT_EQ(0u, ptrClassWithUnsplitMember->fUnsplit.fV.at(0).fV.size());
}

TEST(RField, ForceSplitMode)
{
   auto cl = TClass::GetClass("CustomStreamer");
   ASSERT_TRUE(cl != nullptr);

   EXPECT_FALSE(cl->CanSplit());
   EXPECT_THROW(RFieldBase::Create("f", "CustomStreamer").Unwrap(), RException);

   cl->CreateAttributeMap();
   cl->GetAttributeMap()->AddProperty("rntuple.split", "true");

   // No exception
   RFieldBase::Create("f", "CustomStreamer").Unwrap();

   // "Force Split" attribute set by Linkdef
   cl = TClass::GetClass("CustomStreamerForceSplit");
   ASSERT_TRUE(cl != nullptr);
   EXPECT_FALSE(cl->CanSplit());
   // No exception
   RFieldBase::Create("f", "CustomStreamerForceSplit");

   // "Force Split" attribute set by selection XML
   cl = TClass::GetClass("ForceSplitXML");
   ASSERT_TRUE(cl != nullptr);
   EXPECT_FALSE(cl->CanSplit());
   // No exception
   RFieldBase::Create("f", "ForceSplitXML");

   // "Force Unsplit" attribute set by Linkdef
   cl = TClass::GetClass("CustomStreamerForceUnsplit");
   ASSERT_TRUE(cl != nullptr);
   EXPECT_TRUE(cl->CanSplit());
   auto f = RFieldBase::Create("f", "CustomStreamerForceUnsplit").Unwrap();
   EXPECT_TRUE(dynamic_cast<ROOT::Experimental::RUnsplitField *>(f.get()) != nullptr);

   // "Force Unsplit" attribute set by selection XML
   cl = TClass::GetClass("ForceUnsplitXML");
   ASSERT_TRUE(cl != nullptr);
   EXPECT_TRUE(cl->CanSplit());
   f = RFieldBase::Create("f", "ForceUnsplitXML").Unwrap();
   EXPECT_TRUE(dynamic_cast<ROOT::Experimental::RUnsplitField *>(f.get()) != nullptr);
}

TEST(RField, IgnoreUnsplitComment)
{
   auto fieldClass = RFieldBase::Create("f", "IgnoreUnsplitComment").Unwrap();

   // Only one member, so we know that it is first sub field
   const auto fieldMember = fieldClass->GetSubFields()[0];
   EXPECT_EQ(std::string("v"), fieldMember->GetFieldName());
   EXPECT_EQ(nullptr, dynamic_cast<const ROOT::Experimental::RUnsplitField *>(fieldMember));
}

TEST(RField, UnsupportedUnsplit)
{
   using ROOT::Experimental::RUnsplitField;
   auto success = std::make_unique<RUnsplitField>("name", "std::vector<int>");
   EXPECT_THROW(std::make_unique<RUnsplitField>("name", "int"), RException); // no TClass of fundamental types

   // Unsplit types cannot be added through MakeField<T> but only through RFieldBase::CreateField()
   auto model = RNTupleModel::Create();
   EXPECT_THROW(model->MakeField<CustomStreamerForceUnsplit>("f"), RException);
}

TEST(RField, UnsplitPoly)
{
   FileRaii fileGuard("test_ntuple_rfield_unsplit_poly.root");
   {
      auto model = RNTupleModel::Create();
      model->AddField(RFieldBase::Create("p", "PolyContainer").Unwrap());
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
      auto ptrPoly = writer->GetModel().GetDefaultEntry().GetPtr<PolyContainer>("p");
      ptrPoly->fPoly = std::make_unique<PolyBase>();
      ptrPoly->fPoly->x = 0;
      writer->Fill();
      ptrPoly->fPoly = std::make_unique<PolyA>();
      ptrPoly->fPoly->x = 1;
      dynamic_cast<PolyA *>(ptrPoly->fPoly.get())->a = 100;
      writer->Fill();
      ptrPoly->fPoly = std::make_unique<PolyB>();
      ptrPoly->fPoly->x = 2;
      dynamic_cast<PolyB *>(ptrPoly->fPoly.get())->b = 200;
      writer->Fill();
   }

   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
   ASSERT_EQ(3U, reader->GetNEntries());

   auto ptrPoly = reader->GetModel().GetDefaultEntry().GetPtr<PolyContainer>("p");

   reader->LoadEntry(0);
   EXPECT_EQ(0, ptrPoly->fPoly->x);
   reader->LoadEntry(1);
   EXPECT_EQ(1, ptrPoly->fPoly->x);
   EXPECT_EQ(100, dynamic_cast<PolyA *>(ptrPoly->fPoly.get())->a);
   reader->LoadEntry(2);
   EXPECT_EQ(2, ptrPoly->fPoly->x);
   EXPECT_EQ(200, dynamic_cast<PolyB *>(ptrPoly->fPoly.get())->b);
}

TEST(RField, UnsplitMerge)
{
   FileRaii fileGuard1("test_ntuple_merge_unsplit_1.root");
   {
      auto model = RNTupleModel::Create();
      model->AddField(RFieldBase::Create("p", "PolyContainer").Unwrap());
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard1.GetPath());
      auto ptrPoly = writer->GetModel().GetDefaultEntry().GetPtr<PolyContainer>("p");
      ptrPoly->fPoly = std::make_unique<PolyA>();
      ptrPoly->fPoly->x = 1;
      dynamic_cast<PolyA *>(ptrPoly->fPoly.get())->a = 100;
      writer->Fill();
   }

   FileRaii fileGuard2("test_ntuple_merge_unsplit_2.root");
   {
      auto model = RNTupleModel::Create();
      model->AddField(RFieldBase::Create("p", "PolyContainer").Unwrap());
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard2.GetPath());
      auto ptrPoly = writer->GetModel().GetDefaultEntry().GetPtr<PolyContainer>("p");
      ptrPoly->fPoly = std::make_unique<PolyB>();
      ptrPoly->fPoly->x = 2;
      dynamic_cast<PolyB *>(ptrPoly->fPoly.get())->b = 200;
      writer->Fill();
   }

   // Now merge the inputs
   FileRaii fileGuard3("test_ntuple_merge_unsplit_out.root");
   {
      std::vector<std::unique_ptr<RPageSource>> sources;
      sources.push_back(RPageSource::Create("ntpl", fileGuard1.GetPath(), RNTupleReadOptions()));
      sources.push_back(RPageSource::Create("ntpl", fileGuard2.GetPath(), RNTupleReadOptions()));
      std::vector<RPageSource *> sourcePtrs{sources[0].get(), sources[1].get()};
      auto destination = std::make_unique<RPageSinkFile>("ntpl", fileGuard3.GetPath(), RNTupleWriteOptions());

      RNTupleMerger merger;
      EXPECT_NO_THROW(merger.Merge(sourcePtrs, *destination));
   }

   auto reader = RNTupleReader::Open("ntpl", fileGuard3.GetPath());
   EXPECT_EQ(2u, reader->GetNEntries());
   auto ptrPoly = reader->GetModel().GetDefaultEntry().GetPtr<PolyContainer>("p");
   reader->LoadEntry(0);
   EXPECT_EQ(1, ptrPoly->fPoly->x);
   EXPECT_EQ(100, dynamic_cast<PolyA *>(ptrPoly->fPoly.get())->a);
   reader->LoadEntry(1);
   EXPECT_EQ(2, ptrPoly->fPoly->x);
   EXPECT_EQ(200, dynamic_cast<PolyB *>(ptrPoly->fPoly.get())->b);

   const auto &desc = reader->GetDescriptor();
   EXPECT_EQ(1u, desc.GetNExtraTypeInfos());
   const auto &typeInfo = *desc.GetExtraTypeInfoIterable().begin();
   EXPECT_EQ(ROOT::Experimental::EExtraTypeInfoIds::kStreamerInfo, typeInfo.GetContentId());
   auto streamerInfoMap = RNTupleSerializer::DeserializeStreamerInfos(typeInfo.GetContent()).Unwrap();
   EXPECT_EQ(4u, streamerInfoMap.size());
   std::array<bool, 4> seenStreamerInfos{false, false, false, false};
   for (const auto [_, streamerInfo] : streamerInfoMap) {
      if (strcmp(streamerInfo->GetName(), "PolyContainer") == 0)
         seenStreamerInfos[0] = true;
      else if (strcmp(streamerInfo->GetName(), "PolyBase") == 0)
         seenStreamerInfos[1] = true;
      else if (strcmp(streamerInfo->GetName(), "PolyA") == 0)
         seenStreamerInfos[2] = true;
      else if (strcmp(streamerInfo->GetName(), "PolyB") == 0)
         seenStreamerInfos[3] = true;
   }
   EXPECT_TRUE(seenStreamerInfos[0]);
   EXPECT_TRUE(seenStreamerInfos[1]);
   EXPECT_TRUE(seenStreamerInfos[2]);
   EXPECT_TRUE(seenStreamerInfos[3]);
}
