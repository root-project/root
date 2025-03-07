#include "ntuple_test.hxx"

#include <TDictAttributeMap.h>
#include <TVirtualStreamerInfo.h>
#include <TFileMerger.h>

#include "StreamerField.hxx"
#include "StreamerFieldXML.h"

TEST(RField, StreamerDirect)
{
   FileRaii fileGuard("test_ntuple_rfield_streamer_direct.root");
   {
      auto model = RNTupleModel::Create();
      model->AddField(std::make_unique<ROOT::Experimental::RStreamerField>("pt", "std::vector<float>"));
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

TEST(RField, StreamedMember)
{
   auto cl = TClass::GetClass("CyclicMember");
   cl->CreateAttributeMap();
   cl->GetAttributeMap()->AddProperty("rntuple.streamerMode", "true");

   FileRaii fileGuard("test_ntuple_rfield_streamed_member.root");
   {
      auto model = RNTupleModel::Create();
      auto ptrClassWithStreamedMember = model->MakeField<ClassWithStreamedMember>("event");
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
      ptrClassWithStreamedMember->fA = 1.0;
      CyclicMember inner;
      inner.fB = 3.0;
      ptrClassWithStreamedMember->fStreamed.fB = 2.0;
      ptrClassWithStreamedMember->fStreamed.fV.push_back(inner);
      writer->Fill();
   }

   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
   auto ptrClassWithStreamedMember = reader->GetModel().GetDefaultEntry().GetPtr<ClassWithStreamedMember>("event");

   ASSERT_EQ(1U, reader->GetNEntries());
   reader->LoadEntry(0);
   EXPECT_FLOAT_EQ(1.0, ptrClassWithStreamedMember->fA);
   EXPECT_FLOAT_EQ(2.0, ptrClassWithStreamedMember->fStreamed.fB);
   EXPECT_EQ(1u, ptrClassWithStreamedMember->fStreamed.fV.size());
   EXPECT_FLOAT_EQ(3.0, ptrClassWithStreamedMember->fStreamed.fV.at(0).fB);
   EXPECT_EQ(0u, ptrClassWithStreamedMember->fStreamed.fV.at(0).fV.size());
}

TEST(RField, ForceNativeMode)
{
   auto cl = TClass::GetClass("CustomStreamer");
   ASSERT_TRUE(cl != nullptr);

   EXPECT_FALSE(cl->CanSplit());
   EXPECT_THROW(RFieldBase::Create("f", "CustomStreamer").Unwrap(), ROOT::RException);

   cl->CreateAttributeMap();
   cl->GetAttributeMap()->AddProperty("rntuple.streamerMode", "false");

   // No exception
   RFieldBase::Create("f", "CustomStreamer").Unwrap();

   // "Force Native" attribute set by Linkdef
   cl = TClass::GetClass("CustomStreamerForceNative");
   ASSERT_TRUE(cl != nullptr);
   EXPECT_FALSE(cl->CanSplit());
   // No exception
   RFieldBase::Create("f", "CustomStreamerForceNative");

   // "Force Native" attribute set by selection XML
   cl = TClass::GetClass("ForceNativeXML");
   ASSERT_TRUE(cl != nullptr);
   EXPECT_FALSE(cl->CanSplit());
   // No exception
   RFieldBase::Create("f", "ForceNativeXML");

   // "Force Streamed" attribute set by Linkdef
   cl = TClass::GetClass("CustomStreamerForceStreamed");
   ASSERT_TRUE(cl != nullptr);
   EXPECT_TRUE(cl->CanSplit());
   auto f = RFieldBase::Create("f", "CustomStreamerForceStreamed").Unwrap();
   EXPECT_TRUE(dynamic_cast<ROOT::Experimental::RStreamerField *>(f.get()) != nullptr);

   // "Force Streamed" attribute set by selection XML
   cl = TClass::GetClass("ForceStreamedXML");
   ASSERT_TRUE(cl != nullptr);
   EXPECT_TRUE(cl->CanSplit());
   f = RFieldBase::Create("f", "ForceStreamedXML").Unwrap();
   EXPECT_TRUE(dynamic_cast<ROOT::Experimental::RStreamerField *>(f.get()) != nullptr);
}

TEST(RField, IgnoreUnsplitComment)
{
   auto fieldClass = RFieldBase::Create("f", "IgnoreUnsplitComment").Unwrap();

   // Only one member, so we know that it is first sub field
   const auto fieldMember = fieldClass->GetConstSubfields()[0];
   EXPECT_EQ(std::string("v"), fieldMember->GetFieldName());
   EXPECT_EQ(nullptr, dynamic_cast<const ROOT::Experimental::RStreamerField *>(fieldMember));
}

TEST(RField, UnsupportedStreamed)
{
   using ROOT::Experimental::RStreamerField;
   auto success = std::make_unique<RStreamerField>("name", "std::vector<int>");
   EXPECT_THROW(RStreamerField("name", "int"), ROOT::RException); // no TClass of fundamental types

   // Streamer fields cannot be added through MakeField<T> but only through RFieldBase::CreateField()
   auto model = RNTupleModel::Create();
   EXPECT_THROW(model->MakeField<CustomStreamerForceStreamed>("f"), ROOT::RException);
}

TEST(RField, StreamerPoly)
{
   FileRaii fileGuard("test_ntuple_rfield_streamer_poly.root");
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

TEST(RField, StreamerMerge)
{
   FileRaii fileGuard1("test_ntuple_merge_streamer_1.root");
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

   FileRaii fileGuard2("test_ntuple_merge_streamer_2.root");
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
   FileRaii fileGuard3("test_ntuple_merge_streamed_out.root");
   {
      std::vector<std::unique_ptr<RPageSource>> sources;
      sources.push_back(RPageSource::Create("ntpl", fileGuard1.GetPath(), RNTupleReadOptions()));
      sources.push_back(RPageSource::Create("ntpl", fileGuard2.GetPath(), RNTupleReadOptions()));
      std::vector<RPageSource *> sourcePtrs{sources[0].get(), sources[1].get()};
      auto destination = std::make_unique<RPageSinkFile>("ntpl", fileGuard3.GetPath(), RNTupleWriteOptions());

      RNTupleMerger merger{std::move(destination)};
      EXPECT_NO_THROW(merger.Merge(sourcePtrs));
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
   for (const auto &[_, streamerInfo] : streamerInfoMap) {
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

TEST(RField, StreamerMergeIncremental)
{
   FileRaii fileGuard1("test_ntuple_merge_streamer_incr_1.root");
   {
      auto model = RNTupleModel::Create();
      model->AddField(RFieldBase::Create("p", "PolyContainer").Unwrap());
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard1.GetPath());
      auto ptrPoly = writer->GetModel().GetDefaultEntry().GetPtr<PolyContainer>("p");
      ptrPoly->fPoly = std::make_unique<PolyA>();
      ptrPoly->fPoly->x = 1;
      static_cast<PolyA &>(*ptrPoly->fPoly).a = 100;
      writer->Fill();
   }

   FileRaii fileGuard2("test_ntuple_merge_streamer_incr_2.root");
   {
      auto model = RNTupleModel::Create();
      model->AddField(RFieldBase::Create("p", "PolyContainer").Unwrap());
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard2.GetPath());
      auto ptrPoly = writer->GetModel().GetDefaultEntry().GetPtr<PolyContainer>("p");
      ptrPoly->fPoly = std::make_unique<PolyB>();
      ptrPoly->fPoly->x = 2;
      static_cast<PolyB &>(*ptrPoly->fPoly).b = 200;
      writer->Fill();
   }

   // Now merge the inputs
   FileRaii fileGuard3("test_ntuple_merge_streamer_incr_out.root");

   TFileMerger merger(kFALSE, kFALSE);
   merger.OutputFile(fileGuard3.GetPath().c_str(), "RECREATE", 505);

   const FileRaii inputFiles[] = {std::move(fileGuard1), std::move(fileGuard2)};

   for (auto i = 0u; i < std::size(inputFiles); ++i) {
      auto tfile = std::unique_ptr<TFile>(TFile::Open(inputFiles[i].GetPath().c_str(), "READ"));
      merger.AddFile(tfile.get());
      bool result =
         merger.PartialMerge(TFileMerger::kIncremental | TFileMerger::kNonResetable | TFileMerger::kKeepCompression);
      ASSERT_TRUE(result);
   }

   auto reader = RNTupleReader::Open("ntpl", fileGuard3.GetPath());
   EXPECT_EQ(2u, reader->GetNEntries());
   auto ptrPoly = reader->GetModel().GetDefaultEntry().GetPtr<PolyContainer>("p");
   reader->LoadEntry(0);
   EXPECT_EQ(1, ptrPoly->fPoly->x);
   EXPECT_EQ(100, static_cast<PolyA &>(*ptrPoly->fPoly).a);
   reader->LoadEntry(1);
   EXPECT_EQ(2, ptrPoly->fPoly->x);
   EXPECT_EQ(200, static_cast<PolyB &>(*ptrPoly->fPoly).b);

   const auto &desc = reader->GetDescriptor();
   EXPECT_EQ(1u, desc.GetNExtraTypeInfos());
   const auto &typeInfo = *desc.GetExtraTypeInfoIterable().begin();
   EXPECT_EQ(ROOT::Experimental::EExtraTypeInfoIds::kStreamerInfo, typeInfo.GetContentId());
   auto streamerInfoMap = RNTupleSerializer::DeserializeStreamerInfos(typeInfo.GetContent()).Unwrap();
   EXPECT_EQ(4u, streamerInfoMap.size());
   std::array<bool, 4> seenStreamerInfos{false, false, false, false};
   for (const auto &[_, streamerInfo] : streamerInfoMap) {
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
