#include "ntuple_test.hxx"
#include <ROOT/RNTupleAttrWriting.hxx>

TEST(RNTupleAttributes, CreateWriter)
{
   FileRaii fileGuard("ntuple_attr_create_writer.root");

   ROOT::TestSupport::CheckDiagsRAII diagsRaii;
   diagsRaii.requiredDiag(kWarning, "ROOT.NTuple", "RNTuple Attributes are experimental", false);

   auto file = std::unique_ptr<TFile>(TFile::Open(fileGuard.GetPath().c_str(), "RECREATE"));
   auto model = RNTupleModel::Create();

   auto writer = RNTupleWriter::Append(std::move(model), "ntuple", *file);

   auto attrModel = RNTupleModel::Create();
   auto attrSetWriter = writer->CreateAttributeSet(std::move(attrModel), "AttrSet1");

   // both bare and non-bare models work.
   auto attrModel2 = RNTupleModel::CreateBare();
   auto attrSetWriter2 = writer->CreateAttributeSet(std::move(attrModel2), "AttrSet2");

   // Should fail to create an attribute set with a reserved name (ones starting with `__`)
   try {
      writer->CreateAttributeSet(RNTupleModel::Create(), "__ROOT");
      FAIL() << "creating an attribute set with a reserved name should fail.";
   } catch (const ROOT::RException &ex) {
      EXPECT_THAT(ex.what(), testing::HasSubstr("reserved for internal use"));
   }

   // Should fail to create an attribute set with an empty name
   try {
      writer->CreateAttributeSet(RNTupleModel::Create(), "");
      FAIL() << "creating an attribute set with an empty name should fail.";
   } catch (const ROOT::RException &ex) {
      EXPECT_THAT(ex.what(), testing::HasSubstr("Attribute Set with an empty name"));
   }

   // (This is not a reserved name).
   EXPECT_NO_THROW(writer->CreateAttributeSet(RNTupleModel::Create(), "ROOT__"));
}

TEST(RNTupleAttributes, AttributeSetDuplicateName)
{
   FileRaii fileGuard("test_ntuple_attrs_duplicate_name.root");

   ROOT::TestSupport::CheckDiagsRAII diagsRaii;
   diagsRaii.requiredDiag(kWarning, "ROOT.NTuple", "RNTuple Attributes are experimental", false);

   // Create a RNTuple
   auto model = RNTupleModel::Create();
   auto pInt = model->MakeField<int>("int");
   auto file = std::unique_ptr<TFile>(TFile::Open(fileGuard.GetPath().c_str(), "RECREATE"));
   auto writer = RNTupleWriter::Append(std::move(model), "ntpl", *file);

   auto attrModel = RNTupleModel::Create();
   writer->CreateAttributeSet(attrModel->Clone(), "MyAttrSet");
   try {
      writer->CreateAttributeSet(std::move(attrModel), "MyAttrSet");
      FAIL() << "Trying to create duplicate attribute sets should fail";
   } catch (const ROOT::RException &ex) {
      EXPECT_THAT(ex.what(), testing::HasSubstr("already exists"));
   }
}

TEST(RNTupleAttributes, BasicWriting)
{
   FileRaii fileGuard("ntuple_attr_basic_writing.root");

   ROOT::TestSupport::CheckDiagsRAII diagsRaii;
   diagsRaii.requiredDiag(kWarning, "ROOT.NTuple", "RNTuple Attributes are experimental", false);

   auto file = std::unique_ptr<TFile>(TFile::Open(fileGuard.GetPath().c_str(), "RECREATE"));
   auto model = RNTupleModel::Create();
   auto pInt = model->MakeField<int>("int");
   auto writer = RNTupleWriter::Append(std::move(model), "ntuple", *file);

   auto attrModel = RNTupleModel::Create();
   auto pAttr = attrModel->MakeField<std::string>("attr");
   auto attrSetWriter = writer->CreateAttributeSet(std::move(attrModel), "AttrSet1");

   auto attrRange = attrSetWriter->BeginRange();
   *pAttr = "My Attribute";
   for (int i = 0; i < 100; ++i) {
      *pInt = i;
      writer->Fill();
   }
   attrSetWriter->CommitRange(std::move(attrRange));
   writer.reset();

   // Cannot create new ranges after closing the main writer
   EXPECT_THROW((attrRange = attrSetWriter->BeginRange()), ROOT::RException);

   // Cannot directly fetch the attribute RNTuple from the TFile
   {
      auto tfile = std::unique_ptr<TFile>(TFile::Open(fileGuard.GetPath().c_str()));
      auto ntuple = tfile->Get<ROOT::RNTuple>("AttrSet1");
      EXPECT_EQ(ntuple, nullptr);
   }

   auto reader = RNTupleReader::Open("ntuple", fileGuard.GetPath());
   EXPECT_EQ(reader->GetDescriptor().GetNAttributeSets(), 1);
   for (const auto &attrSetIt : reader->GetDescriptor().GetAttrSetIterable()) {
      EXPECT_EQ(attrSetIt.GetName(), "AttrSet1");
   }
}

TEST(RNTupleAttributes, BasicWritingWithExplicitEntry)
{
   FileRaii fileGuard("ntuple_attr_basic_writing_entry.root");

   ROOT::TestSupport::CheckDiagsRAII diagsRaii;
   diagsRaii.requiredDiag(kWarning, "ROOT.NTuple", "RNTuple Attributes are experimental", false);

   auto file = std::unique_ptr<TFile>(TFile::Open(fileGuard.GetPath().c_str(), "RECREATE"));
   auto model = RNTupleModel::Create();
   auto pInt = model->MakeField<int>("int");
   auto writer = RNTupleWriter::Append(std::move(model), "ntuple", *file);

   auto attrModel = RNTupleModel::CreateBare();
   attrModel->MakeField<std::string>("attr");
   auto attrSetWriter = writer->CreateAttributeSet(std::move(attrModel), "AttrSet1");
   auto attrEntry = attrSetWriter->CreateEntry();
   auto pAttr = attrEntry->GetPtr<std::string>("attr");

   auto attrRange = attrSetWriter->BeginRange();
   *pAttr = "My Attribute";
   for (int i = 0; i < 100; ++i) {
      *pInt = i;
      writer->Fill();
   }
   attrSetWriter->CommitRange(std::move(attrRange), *attrEntry);
   writer.reset();

   // Cannot create new ranges after closing the main writer
   EXPECT_THROW((attrRange = attrSetWriter->BeginRange()), ROOT::RException);

   // Cannot directly fetch the attribute RNTuple from the TFile
   {
      auto tfile = std::unique_ptr<TFile>(TFile::Open(fileGuard.GetPath().c_str()));
      auto ntuple = tfile->Get<ROOT::RNTuple>("AttrSet1");
      EXPECT_EQ(ntuple, nullptr);
   }

   auto reader = RNTupleReader::Open("ntuple", fileGuard.GetPath());
   EXPECT_EQ(reader->GetDescriptor().GetNAttributeSets(), 1);
   for (const auto &attrSetIt : reader->GetDescriptor().GetAttrSetIterable()) {
      EXPECT_EQ(attrSetIt.GetName(), "AttrSet1");
   }
}

TEST(RNTupleAttributes, NoCommitRange)
{
   FileRaii fileGuard("ntuple_attr_no_commit_range.root");

   ROOT::TestSupport::CheckDiagsRAII diagsRaii;
   diagsRaii.requiredDiag(kWarning, "ROOT.NTuple", "RNTuple Attributes are experimental", false);

   auto file = std::unique_ptr<TFile>(TFile::Open(fileGuard.GetPath().c_str(), "RECREATE"));
   auto model = RNTupleModel::Create();
   auto pInt = model->MakeField<int>("int");
   auto writer = RNTupleWriter::Append(std::move(model), "ntuple", *file);

   auto attrModel = RNTupleModel::Create();
   auto pAttr = attrModel->MakeField<std::string>("attr");
   auto attrSetWriter = writer->CreateAttributeSet(std::move(attrModel), "AttrSet1");

   diagsRaii.requiredDiag(kWarning, "ROOT.NTuple", "A pending attribute range was not committed", false);

   auto attrRange = attrSetWriter->BeginRange();
   *pAttr = "My Attribute";
   for (int i = 0; i < 100; ++i) {
      *pInt = i;
      writer->Fill();
   }
   // Forgot to commit the range!
}

TEST(RNTupleAttributes, MultipleSets)
{
   // Create multiple sets and interleave attribute ranges

   FileRaii fileGuard("test_ntuple_attrs_multiplesets.root");

   ROOT::TestSupport::CheckDiagsRAII diagsRaii;
   diagsRaii.requiredDiag(kWarning, "ROOT.NTuple", "RNTuple Attributes are experimental", false);

   {
      auto model = RNTupleModel::Create();
      auto pInt = model->MakeField<int>("int");
      auto file = std::unique_ptr<TFile>(TFile::Open(fileGuard.GetPath().c_str(), "RECREATE"));
      auto writer = RNTupleWriter::Append(std::move(model), "ntpl", *file);

      auto attrModel1 = RNTupleModel::Create();
      auto pInt1 = attrModel1->MakeField<int>("int");
      auto attrSet1 = writer->CreateAttributeSet(attrModel1->Clone(), "MyAttrSet1");

      auto attrModel2 = RNTupleModel::Create();
      auto pString2 = attrModel2->MakeField<std::string>("string");
      auto attrOpts2 = ROOT::RNTupleWriteOptions();
      attrOpts2.SetCompression(404);
      auto attrSet2 = writer->CreateAttributeSet(attrModel2->Clone(), "MyAttrSet2", &attrOpts2);

      auto attrRange2 = attrSet2->BeginRange();
      for (int i = 0; i < 100; ++i) {
         auto attrRange1 = attrSet1->BeginRange();
         *pInt1 = i;
         *pInt = i;
         writer->Fill();
         attrSet1->CommitRange(std::move(attrRange1));
      }
      *pString2 = "Run 1";
      attrSet2->CommitRange(std::move(attrRange2));
   }

   auto tfile = std::unique_ptr<TFile>(TFile::Open(fileGuard.GetPath().c_str()));
   auto ntpl = tfile->Get<ROOT::RNTuple>("ntpl");
   auto reader = RNTupleReader::Open(*ntpl);
   EXPECT_EQ(reader->GetDescriptor().GetNAttributeSets(), 2);
   int n = 1;
   for (const auto &attrSetIt : reader->GetDescriptor().GetAttrSetIterable()) {
      EXPECT_EQ(attrSetIt.GetName(), "MyAttrSet" + std::to_string(n));
      ++n;
   }

   // Verify compression
   auto tkeys = tfile->WalkTKeys();
   int nHeader = 0;
   for (const auto &key : tkeys) {
      if (key.fType != ROOT::Detail::TKeyMapNode::kKey || key.fClassName != "RBlob")
         continue;

      // The first 3 RBlobs we're gonna find are, in order: the main RNTuple header, MyAttrSet1's header
      // and MyAttrSet2's header.
      const auto headerSeek = key.fSeekKey + key.fKeyLen;

      // Extract the header's compression
      tfile->Seek(headerSeek);
      unsigned char zipHeader[9];
      bool ok = tfile->ReadBuffer(reinterpret_cast<char *>(zipHeader), sizeof(zipHeader));
      ASSERT_FALSE(ok);

      const int expectedCompression = nHeader < 2 ? ROOT::RCompressionSetting::EAlgorithm::kZSTD : 4;
      const auto realCompression = R__getCompressionAlgorithm(zipHeader, sizeof(zipHeader));
      ASSERT_EQ(realCompression, expectedCompression);

      if (++nHeader == 3)
         break;
   }
}

TEST(RNTupleAttributes, AttributeInvalidModel)
{
   FileRaii fileGuard("test_ntuple_attrs_invalid_model.root");

   auto model = RNTupleModel::Create();
   auto pInt = model->MakeField<int>("int");
   auto file = std::unique_ptr<TFile>(TFile::Open(fileGuard.GetPath().c_str(), "RECREATE"));
   auto writer = RNTupleWriter::Append(std::move(model), "ntpl", *file);

   // Projected fields are forbidden in attribute models
   {
      auto attrModel = RNTupleModel::Create();
      attrModel->MakeField<int>("foo");
      auto projField = std::make_unique<ROOT::RField<int>>("proj");
      attrModel->AddProjectedField(std::move(projField), [](const auto &) { return "foo"; });
      try {
         writer->CreateAttributeSet(attrModel->Clone(), "MyAttrSet");
         FAIL() << "Trying to create an attribute model with projected fields should fail";
      } catch (const ROOT::RException &ex) {
         EXPECT_THAT(ex.what(), testing::HasSubstr("cannot contain projected fields"));
      }
   }

   // Streamer fields are forbidden in attribute models
   {
      auto attrModel = RNTupleModel::Create();
      attrModel->AddField(std::make_unique<ROOT::RStreamerField>("foo", "CustomStruct"));
      try {
         writer->CreateAttributeSet(attrModel->Clone(), "MyAttrSet");
         FAIL() << "Trying to create an attribute model with streamer fields should fail";
      } catch (const ROOT::RException &ex) {
         EXPECT_THAT(ex.what(), testing::HasSubstr("cannot contain Streamer field"));
      }
   }
}

TEST(RNTupleAttributes, AttributeInvalidCommitRange)
{
   // Same as AttributeMultipleSets but try to pass the wrong range to a Set and verify it fails.

   FileRaii fileGuard("test_ntuple_attrs_invalid_endrange.root");

   ROOT::TestSupport::CheckDiagsRAII diagsRaii;
   diagsRaii.requiredDiag(kWarning, "ROOT.NTuple", "RNTuple Attributes are experimental", false);

   // Create a RNTuple
   auto model = RNTupleModel::Create();
   auto pInt = model->MakeField<int>("int");
   auto file = std::unique_ptr<TFile>(TFile::Open(fileGuard.GetPath().c_str(), "RECREATE"));
   auto writer = RNTupleWriter::Append(std::move(model), "ntpl", *file);

   auto attrModel1 = RNTupleModel::Create();
   auto pInt1 = attrModel1->MakeField<int>("int");
   auto attrSet1 = writer->CreateAttributeSet(attrModel1->Clone(), "MyAttrSet1");

   auto attrModel2 = RNTupleModel::Create();
   auto pString2 = attrModel2->MakeField<std::string>("string");
   auto attrSet2 = writer->CreateAttributeSet(attrModel2->Clone(), "MyAttrSet2");

   [[maybe_unused]] auto attrRange2 = attrSet2->BeginRange();

   for (int i = 0; i < 100; ++i) {
      auto attrRange1 = attrSet1->BeginRange();
      *pInt1 = i;
      *pInt = i;
      writer->Fill();
      attrSet1->CommitRange(std::move(attrRange1));
   }
   *pString2 = "Run 1";

   // Oops! Calling CommitRange on the wrong set!
   EXPECT_THROW(attrSet1->CommitRange(std::move(attrRange2)), ROOT::RException);
}

TEST(RNTupleAttributes, InvalidPendingRange)
{
   FileRaii fileGuard("test_ntuple_attr_invalid_pendingrange.root");

   ROOT::TestSupport::CheckDiagsRAII diagsRaii;
   diagsRaii.requiredDiag(kWarning, "ROOT.NTuple", "RNTuple Attributes are experimental", false);

   auto model = RNTupleModel::Create();
   auto file = std::unique_ptr<TFile>(TFile::Open(fileGuard.GetPath().c_str(), "RECREATE"));
   auto writer = RNTupleWriter::Append(std::move(model), "ntpl", *file);

   auto attrModel = RNTupleModel::Create();
   auto attrSet = writer->CreateAttributeSet(std::move(attrModel), "MyAttributes");

   ROOT::Experimental::RNTupleAttrPendingRange attrRange;
   EXPECT_FALSE(attrRange);
   try {
      attrSet->CommitRange(std::move(attrRange));
      FAIL() << "committing an invalid pending range should fail.";
   } catch (const ROOT::RException &ex) {
      EXPECT_THAT(ex.what(), testing::HasSubstr("was not created by it or was already committed"));
   }
}

TEST(RNTupleAttributes, ReassignPendingRange)
{
   // verify that reassigning a pending range and not committing it properly triggers the warning.
   FileRaii fileGuard("test_ntuple_attr_reassign_range.root");

   ROOT::TestSupport::CheckDiagsRAII diagsRaii;
   diagsRaii.requiredDiag(kWarning, "ROOT.NTuple", "RNTuple Attributes are experimental", false);

   auto file = std::unique_ptr<TFile>(TFile::Open(fileGuard.GetPath().c_str(), "RECREATE"));
   auto writer = RNTupleWriter::Append(RNTupleModel::Create(), "ntpl", *file);
   auto attrSet = writer->CreateAttributeSet(RNTupleModel::Create(), "MyAttributes");

   auto attrRange = attrSet->BeginRange();
   attrSet->CommitRange(std::move(attrRange));
   // reassign the range but never commit it.
   attrRange = attrSet->BeginRange();

   diagsRaii.requiredDiag(kWarning, "ROOT.NTuple", "A pending attribute range was not committed", false);
}
