#include "ntuple_test.hxx"
#include <ROOT/RNTupleAttrWriting.hxx>
#include <ROOT/RNTupleAttrReading.hxx>
#include <TKey.h>

static std::size_t Count(ROOT::Experimental::RNTupleAttrEntryIterable iterable)
{
   std::size_t n = 0;
   for ([[maybe_unused]] auto _ : iterable)
      ++n;
   return n;
}

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

TEST(RNTupleAttributes, BasicReadingWriting)
{
   FileRaii fileGuard("ntuple_attr_basic_readwriting.root");

   ROOT::TestSupport::CheckDiagsRAII diagsRaii;
   diagsRaii.requiredDiag(kWarning, "ROOT.NTuple", "RNTuple Attributes are experimental", false);

   /// Writing
   {
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
   }

   // Cannot directly fetch the attribute RNTuple from the TFile
   {
      auto tfile = std::unique_ptr<TFile>(TFile::Open(fileGuard.GetPath().c_str()));
      auto ntuple = tfile->Get<ROOT::RNTuple>("AttrSet1");
      EXPECT_EQ(ntuple, nullptr);
   }

   /// Reading
   auto reader = RNTupleReader::Open("ntuple", fileGuard.GetPath());
   EXPECT_EQ(reader->GetDescriptor().GetNAttributeSets(), 1);
   for (const auto &attrSetIt : reader->GetDescriptor().GetAttrSetIterable()) {
      EXPECT_EQ(attrSetIt.GetName(), "AttrSet1");
   }

   auto attrSetReader = reader->OpenAttributeSet("AttrSet1");
   EXPECT_EQ(attrSetReader->GetNEntries(), 1);
   auto pAttr = attrSetReader->GetModel().GetDefaultEntry().GetPtr<std::string>("attr");
   {
      int nAttrs = 0;
      // iterate all attributes
      for (auto idx : attrSetReader->GetAttributes()) {
         attrSetReader->LoadEntry(idx);
         EXPECT_EQ(*pAttr, "My Attribute");
         nAttrs += 1;
      }
      EXPECT_EQ(nAttrs, 1);
   }
   {
      int nAttrs = 0;
      // attributes containing entry 99
      for (auto idx : attrSetReader->GetAttributes(99)) {
         attrSetReader->LoadEntry(idx);
         EXPECT_EQ(*pAttr, "My Attribute");
         nAttrs += 1;
      }
      EXPECT_EQ(nAttrs, 1);
   }
   {
      // attributes containing entry 100 (no entry)
      auto iter = attrSetReader->GetAttributes(100);
      EXPECT_EQ(iter.begin(), iter.end());
   }
   {
      // attributes contained in entry range 50-200 (no entry)
      auto iter = attrSetReader->GetAttributesInRange(50, 200);
      EXPECT_EQ(iter.begin(), iter.end());
   }
   {
      int nAttrs = 0;
      // attributes contained in entry range 0-1000
      for (auto idx : attrSetReader->GetAttributesInRange(0, 1000)) {
         attrSetReader->LoadEntry(idx);
         EXPECT_EQ(*pAttr, "My Attribute");
         nAttrs += 1;
      }
      EXPECT_EQ(nAttrs, 1);
   }
   {
      // attributes containing entry range 200-300 (no entry)
      auto iter = attrSetReader->GetAttributesContainingRange(200, 300);
      EXPECT_EQ(iter.begin(), iter.end());
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

   auto attrSetReader = reader->OpenAttributeSet("AttrSet1");
   EXPECT_EQ(attrSetReader->GetNEntries(), 1);
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

TEST(RNTupleAttributes, MultipleCommitRange)
{
   // Calling CommitRange multiple times on the same handle is an error (technically it cannot be on the "same handle"
   // since CommitRange requires you to move the handle - meaning the second time you are passing an invalid handle.)

   FileRaii fileGuard("test_ntuple_attrs_multipleend.root");

   ROOT::TestSupport::CheckDiagsRAII diagsRaii;
   diagsRaii.requiredDiag(kWarning, "ROOT.NTuple", "RNTuple Attributes are experimental", false);

   auto model = RNTupleModel::Create();
   auto pInt = model->MakeField<int>("int");
   auto file = std::unique_ptr<TFile>(TFile::Open(fileGuard.GetPath().c_str(), "RECREATE"));
   auto writer = RNTupleWriter::Append(std::move(model), "ntpl", *file);

   auto attrModel = RNTupleModel::Create();
   attrModel->MakeField<std::string>("string");
   auto attrSet = writer->CreateAttributeSet(attrModel->Clone(), "MyAttrSet");

   auto &wModel = writer->GetModel();

   auto attrRange = attrSet->BeginRange();
   auto pMyAttr = attrSet->GetModel().GetDefaultEntry().GetPtr<std::string>("string");
   *pMyAttr = "Run 1";
   for (int i = 0; i < 100; ++i) {
      auto entry = wModel.CreateEntry();
      *pInt = i;
      writer->Fill(*entry);
   }
   attrSet->CommitRange(std::move(attrRange));
   try {
      attrSet->CommitRange(std::move(attrRange));
      FAIL() << "committing the same range twice should fail.";
   } catch (const ROOT::RException &ex) {
      EXPECT_THAT(ex.what(), testing::HasSubstr("was not created by it or was already committed"));
   }
}

TEST(RNTupleAttributes, AccessPastCommitRange)
{
   FileRaii fileGuard("test_ntuple_attrs_pastcommitrange.root");

   ROOT::TestSupport::CheckDiagsRAII diagsRaii;
   diagsRaii.requiredDiag(kWarning, "ROOT.NTuple", "RNTuple Attributes are experimental", false);

   auto model = RNTupleModel::Create();
   auto pInt = model->MakeField<int>("int");
   auto file = std::unique_ptr<TFile>(TFile::Open(fileGuard.GetPath().c_str(), "RECREATE"));
   auto writer = RNTupleWriter::Append(std::move(model), "ntpl", *file);

   auto attrModel = RNTupleModel::Create();
   auto pMyAttr = attrModel->MakeField<std::string>("string");
   auto attrSet = writer->CreateAttributeSet(attrModel->Clone(), "MyAttrSet");

   auto &wModel = writer->GetModel();

   auto attrRange = attrSet->BeginRange();
   *pMyAttr = "Run 1";
   for (int i = 0; i < 100; ++i) {
      auto entry = wModel.CreateEntry();
      *pInt = i;
      writer->Fill(*entry);
   }
   attrSet->CommitRange(std::move(attrRange));
   // Cannot access attrRange after CommitRange()
   EXPECT_THROW(attrRange.GetStart(), ROOT::RException);
}

TEST(RNTupleAttributes, NoImplicitCommitRange)
{
   // CommitRange doesn't get called automatically when a AttributeRangeHandle goes out of scope:
   // forgetting to call CommitRange will cause the attribute range not to be saved.

   FileRaii fileGuard("test_ntuple_attrs_auto_end_range.root");
   ROOT::TestSupport::CheckDiagsRAII diagsRaii;
   diagsRaii.requiredDiag(kWarning, "ROOT.NTuple", "RNTuple Attributes are experimental", false);

   {
      auto model = RNTupleModel::Create();
      auto pInt = model->MakeField<int>("int");
      auto file = std::unique_ptr<TFile>(TFile::Open(fileGuard.GetPath().c_str(), "RECREATE"));
      auto writer = RNTupleWriter::Append(std::move(model), "ntpl", *file);

      auto attrModel = RNTupleModel::Create();
      auto pMyAttr = attrModel->MakeField<std::string>("string");
      auto attrSet = writer->CreateAttributeSet(attrModel->Clone(), "MyAttrSet");

      ROOT::TestSupport::CheckDiagsRAII diags;
      diags.requiredDiag(kWarning, "NTuple", "range was not committed", false);

      [[maybe_unused]] auto attrRange = attrSet->BeginRange();
      *pMyAttr = "Run 1";
      for (int i = 0; i < 10; ++i) {
         *pInt = i;
         writer->Fill();
      }

      // Not calling CommitRange, so the attributes are not written.
   }

   // Read back the attributes
   {
      auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
      // Fetch a specific attribute set
      auto attrSet = reader->OpenAttributeSet("MyAttrSet");
      EXPECT_EQ(Count(attrSet->GetAttributes()), 0);
   }
}

TEST(RNTupleAttributes, MultipleSets)
{
   // Create multiple sets and interleave attribute ranges

   FileRaii fileGuard("test_ntuple_attrs_multiplesets.root");
   ROOT::TestSupport::CheckDiagsRAII diagsRaii;
   diagsRaii.requiredDiag(kWarning, "ROOT.NTuple", "RNTuple Attributes are experimental", false);

   /// Writing
   {
      auto model = RNTupleModel::Create();
      auto pInt = model->MakeField<int>("int");
      auto file = std::unique_ptr<TFile>(TFile::Open(fileGuard.GetPath().c_str(), "RECREATE"));
      auto writer = RNTupleWriter::Append(std::move(model), "ntpl", *file);

      auto attrModel1 = RNTupleModel::Create();
      auto pInt1 = attrModel1->MakeField<int>("int");
      auto attrSet1 = writer->CreateAttributeSet(std::move(attrModel1), "MyAttrSet1");

      auto attrModel2 = RNTupleModel::Create();
      auto pString2 = attrModel2->MakeField<std::string>("string");
      auto attrOpts2 = ROOT::RNTupleWriteOptions();
      attrOpts2.SetCompression(404);
      auto attrSet2 = writer->CreateAttributeSet(std::move(attrModel2), "MyAttrSet2", &attrOpts2);

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

   /// Reading
   auto tfile = std::unique_ptr<TFile>(TFile::Open(fileGuard.GetPath().c_str()));
   auto ntpl = tfile->Get<ROOT::RNTuple>("ntpl");
   auto reader = RNTupleReader::Open(*ntpl);
   EXPECT_EQ(reader->GetDescriptor().GetNAttributeSets(), 2);
   int n = 1;
   for (const auto &attrSetIt : reader->GetDescriptor().GetAttrSetIterable()) {
      EXPECT_EQ(attrSetIt.GetName(), "MyAttrSet" + std::to_string(n));
      ++n;
   }

   auto sets = reader->GetDescriptor().GetAttrSetIterable();
   // NOTE: there is no guaranteed order in which the attribute sets appear in the iterable
   EXPECT_NE(std::find_if(sets.begin(), sets.end(), [](auto &&s) { return s.GetName() == "MyAttrSet1"; }), sets.end());
   EXPECT_NE(std::find_if(sets.begin(), sets.end(), [](auto &&s) { return s.GetName() == "MyAttrSet2"; }), sets.end());

   auto attrSetReader1 = reader->OpenAttributeSet("MyAttrSet1");
   EXPECT_EQ(attrSetReader1->GetNEntries(), 100);
   auto attrSetReader2 = reader->OpenAttributeSet("MyAttrSet2");
   EXPECT_EQ(attrSetReader2->GetNEntries(), 1);

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

   // check attribute ranges
   auto attrEntry1 = attrSetReader1->CreateEntry();
   auto pAttrInt = attrEntry1->GetPtr<int>("int");
   auto attrEntry2 = attrSetReader2->CreateEntry();
   auto pAttrString = attrEntry2->GetPtr<std::string>("string");
   {
      int nAttrs = 0;
      for (auto idx : attrSetReader1->GetAttributesInRange(0, 1000)) {
         auto range = attrSetReader1->LoadEntry(idx, *attrEntry1);
         EXPECT_EQ(*pAttrInt, idx);
         EXPECT_EQ(range.GetStart(), idx);
         EXPECT_EQ(range.GetLength(), 1);
         nAttrs += 1;
      }
      EXPECT_EQ(nAttrs, 100);
   }
   {
      int nAttrs = 0;
      for (auto idx : attrSetReader1->GetAttributes(42)) {
         auto range = attrSetReader1->LoadEntry(idx, *attrEntry1);
         EXPECT_EQ(*pAttrInt, 42);
         EXPECT_EQ(range.GetStart(), 42);
         EXPECT_EQ(range.GetLength(), 1);
         nAttrs += 1;
      }
      EXPECT_EQ(nAttrs, 1);
   }
   {
      int nAttrs = 0;
      for (auto idx : attrSetReader2->GetAttributes()) {
         auto range = attrSetReader2->LoadEntry(idx, *attrEntry2);
         EXPECT_EQ(*pAttrString, "Run 1");
         EXPECT_EQ(range.GetStart(), 0);
         EXPECT_EQ(range.GetLength(), 100);
         nAttrs += 1;
      }
      EXPECT_EQ(nAttrs, 1);
   }
   {
      for (auto idx : attrSetReader2->GetAttributes()) {
         // Reading into the wrong entry
         try {
            attrSetReader2->LoadEntry(idx, *attrEntry1);
            FAIL() << "reading into an unrelated entry should fail";
         } catch (const ROOT::RException &ex) {
            EXPECT_THAT(ex.what(), testing::HasSubstr("mismatch between entry and model"));
         }
      }
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

TEST(RNTupleAttributes, InterleavingRanges)
{
   // Calling BeginRange multiple times without calling CommitRange in between is valid:
   // the user gets separate AttributeEntries that they can commit at their will.

   FileRaii fileGuard("test_ntuple_attrs_multiplebegin.root");

   ROOT::TestSupport::CheckDiagsRAII diagsRaii;
   diagsRaii.requiredDiag(kWarning, "ROOT.NTuple", "RNTuple Attributes are experimental", false);

   {
      auto model = RNTupleModel::Create();
      auto pInt = model->MakeField<int>("int");
      auto file = std::unique_ptr<TFile>(TFile::Open(fileGuard.GetPath().c_str(), "RECREATE"));
      auto wopts = RNTupleWriteOptions();
      wopts.SetCompression(0);
      auto writer = RNTupleWriter::Append(std::move(model), "ntpl", *file, wopts);

      auto attrModel = RNTupleModel::Create();
      attrModel->MakeField<int>("attrInt");
      auto attrSet = writer->CreateAttributeSet(attrModel->Clone(), "MyAttrSet");

      auto attrEntry1 = attrSet->CreateEntry();
      auto attrRange1 = attrSet->BeginRange();
      auto attrEntry2 = attrSet->CreateEntry();
      auto attrRange2 = attrSet->BeginRange();
      int i1 = 0, i2 = 0;
      *attrEntry1->GetPtr<int>("attrInt") = i1++;
      *attrEntry2->GetPtr<int>("attrInt") = i2++;
      for (int i = 0; i < 30; ++i) {
         if (i > 0 && (i % 5) == 0) {
            attrSet->CommitRange(std::move(attrRange1), *attrEntry1);
            attrRange1 = attrSet->BeginRange();
            *attrEntry1->GetPtr<int>("attrInt") = i1++;
         }
         if (i > 0 && (i % 11) == 0) {
            attrSet->CommitRange(std::move(attrRange2), *attrEntry2);
            attrRange2 = attrSet->BeginRange();
            *attrEntry2->GetPtr<int>("attrInt") = i2++;
         }
         *pInt = i;
         writer->Fill();
      }
      attrSet->CommitRange(std::move(attrRange1), *attrEntry1);
      attrSet->CommitRange(std::move(attrRange2), *attrEntry2);
   }

   // read back
   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
   auto attrSet = reader->OpenAttributeSet("MyAttrSet");
   auto attrEntry = attrSet->CreateEntry();
   for (auto i : reader->GetEntryRange()) {
      auto attrs = attrSet->GetAttributes(i);
      const auto nAttrs = Count(attrs);
      EXPECT_EQ(nAttrs, 2);
      int totVal = 0;
      for (auto idx : attrs) {
         attrSet->LoadEntry(idx, *attrEntry);
         totVal += *attrEntry->GetPtr<int>("attrInt");
      }
      int expected = (i / 5) + (i / 11);
      EXPECT_EQ(totVal, expected);
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

TEST(RNTupleAttributes, AssignMetadataAfterData)
{
   // Assigning the attribute range's value to the pointer can be done either before or after filling
   // the corresponding rows - provided it's done between a BeginRange() and an CommitRange().

   FileRaii fileGuard("test_ntuple_attrs_assign_after.root");
   ROOT::TestSupport::CheckDiagsRAII diagsRaii;
   diagsRaii.requiredDiag(kWarning, "ROOT.NTuple", "RNTuple Attributes are experimental", false);

   {
      auto model = RNTupleModel::Create();
      auto pInt = model->MakeField<int>("int");
      auto file = std::unique_ptr<TFile>(TFile::Open(fileGuard.GetPath().c_str(), "RECREATE"));
      auto writer = RNTupleWriter::Append(std::move(model), "ntpl", *file);

      auto attrModel = RNTupleModel::Create();
      auto pAttrString = attrModel->MakeField<std::string>("string");
      auto pAttrInt = attrModel->MakeField<int>("int");
      auto attrSet = writer->CreateAttributeSet(std::move(attrModel), "MyAttrSet");

      // First attribute entry
      {
         auto attrRange = attrSet->BeginRange();
         *pAttrString = "Run 1";
         *pAttrInt = 1;
         for (int i = 0; i < 10; ++i) {
            *pInt = i;
            writer->Fill();
         }
         attrSet->CommitRange(std::move(attrRange));
      }

      // Second attribute entry
      {
         auto attrRange = attrSet->BeginRange();
         for (int i = 0; i < 10; ++i) {
            *pInt = i;
            writer->Fill();
         }
         *pAttrString = "Run 2";
         *pAttrInt = 2;
         attrSet->CommitRange(std::move(attrRange));
      }
   }

   // Read back the attributes
   {
      auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
      // Fetch a specific attribute set
      auto attrSet = reader->OpenAttributeSet("MyAttrSet");
      auto nAttrs = 0;
      auto attrEntry = attrSet->CreateEntry();
      for (const auto idx : attrSet->GetAttributes()) {
         const auto range = attrSet->LoadEntry(idx, *attrEntry);
         auto pAttrStr = attrEntry->GetPtr<std::string>("string");
         auto pAttrInt = attrEntry->GetPtr<int>("int");
         EXPECT_EQ(range.GetStart(), nAttrs * 10);
         EXPECT_EQ(range.GetEnd(), (1 + nAttrs) * 10);
         EXPECT_EQ(*pAttrStr, nAttrs == 0 ? "Run 1" : "Run 2");
         EXPECT_EQ(*pAttrInt, nAttrs == 0 ? 1 : 2);
         ++nAttrs;
      }
      EXPECT_EQ(nAttrs, 2);
   }
}

TEST(RNTupleAttributes, ReadRanges)
{
   // Testing reading attribute ranges with a [startIdx, endIdx].

   FileRaii fileGuard("test_ntuple_attrs_readranges.root");
   ROOT::TestSupport::CheckDiagsRAII diagsRaii;
   diagsRaii.requiredDiag(kWarning, "ROOT.NTuple", "RNTuple Attributes are experimental", false);

   {
      auto model = RNTupleModel::Create();
      auto pInt = model->MakeField<int>("int");
      auto file = std::unique_ptr<TFile>(TFile::Open(fileGuard.GetPath().c_str(), "RECREATE"));
      auto writer = RNTupleWriter::Append(std::move(model), "ntpl", *file);

      auto attrModelRuns = RNTupleModel::Create();
      auto pRun = attrModelRuns->MakeField<std::string>("run");
      auto attrSetRuns = writer->CreateAttributeSet(std::move(attrModelRuns), "Attr_Runs");
      auto attrModelEpochs = RNTupleModel::Create();
      auto pEpoch = attrModelEpochs->MakeField<std::string>("epoch");
      auto attrSetEpochs = writer->CreateAttributeSet(std::move(attrModelEpochs), "Attr_Epochs");

      {
         auto attrRangeEpoch = attrSetEpochs->BeginRange();
         *pEpoch = "Epoch 1";

         {
            auto attrRange = attrSetRuns->BeginRange();
            *pRun = "Run 1";
            for (int i = 0; i < 10; ++i) {
               *pInt = i;
               writer->Fill();
            }
            attrSetRuns->CommitRange(std::move(attrRange));
         }

         // Second attribute entry
         {
            auto attrRange = attrSetRuns->BeginRange();
            for (int i = 0; i < 10; ++i) {
               *pInt = i;
               writer->Fill();
            }
            *pRun = "Run 2";
            attrSetRuns->CommitRange(std::move(attrRange));
         }
         attrSetEpochs->CommitRange(std::move(attrRangeEpoch));
      }
      {
         auto attrRangeEpoch = attrSetEpochs->BeginRange();
         *pEpoch = "Epoch 2";

         {
            auto attrRange = attrSetRuns->BeginRange();
            *pRun = "Run 3";
            for (int i = 0; i < 10; ++i) {
               *pInt = i;
               writer->Fill();
            }
            attrSetRuns->CommitRange(std::move(attrRange));
         }
         attrSetEpochs->CommitRange(std::move(attrRangeEpoch));
      }
   }

   // Read back the attributes
   {
      auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
      // Fetch a specific attribute set
      auto attrSetRuns = reader->OpenAttributeSet("Attr_Runs");
      EXPECT_EQ(Count(attrSetRuns->GetAttributes()), 3);
      EXPECT_EQ(Count(attrSetRuns->GetAttributes(4)), 1);
      EXPECT_EQ(Count(attrSetRuns->GetAttributesContainingRange(4, 5)), 1);
      EXPECT_EQ(Count(attrSetRuns->GetAttributesContainingRange(0, 10)), 1);
      EXPECT_EQ(Count(attrSetRuns->GetAttributesContainingRange(0, 11)), 0);
      EXPECT_EQ(Count(attrSetRuns->GetAttributesContainingRange(0, 100)), 0);
      EXPECT_EQ(Count(attrSetRuns->GetAttributesInRange(4, 5)), 0);
      EXPECT_EQ(Count(attrSetRuns->GetAttributesInRange(0, 100)), 3);
      EXPECT_EQ(Count(attrSetRuns->GetAttributes(12)), 1);
      EXPECT_EQ(Count(attrSetRuns->GetAttributes(120)), 0);

      ROOT_EXPECT_WARNING_PARTIAL(EXPECT_EQ(Count(attrSetRuns->GetAttributesInRange(4, 3)), 0), "ROOT.NTuple",
                                  "empty range");
      ROOT_EXPECT_WARNING_PARTIAL(EXPECT_EQ(Count(attrSetRuns->GetAttributesContainingRange(4, 3)), 0), "ROOT.NTuple",
                                  "empty range");
      ROOT_EXPECT_WARNING_PARTIAL(EXPECT_EQ(Count(attrSetRuns->GetAttributesInRange(4, 4)), 0), "ROOT.NTuple",
                                  "empty range");
      ROOT_EXPECT_WARNING_PARTIAL(EXPECT_EQ(Count(attrSetRuns->GetAttributesContainingRange(4, 4)), 0), "ROOT.NTuple",
                                  "empty range");

      auto attrSetEpochs = reader->OpenAttributeSet("Attr_Epochs");
      EXPECT_EQ(Count(attrSetEpochs->GetAttributes()), 2);
   }
}

TEST(RNTupleAttributes, EmptyAttrRange)
{
   FileRaii fileGuard("test_ntuple_attrs_empty.root");
   ROOT::TestSupport::CheckDiagsRAII diagsRaii;
   diagsRaii.requiredDiag(kWarning, "ROOT.NTuple", "RNTuple Attributes are experimental", false);

   {
      auto model = RNTupleModel::Create();
      auto pInt = model->MakeField<int>("int");
      auto file = std::unique_ptr<TFile>(TFile::Open(fileGuard.GetPath().c_str(), "RECREATE"));
      auto writer = RNTupleWriter::Append(std::move(model), "ntpl", *file);

      auto attrModel = RNTupleModel::Create();
      auto pAttr = attrModel->MakeField<std::string>("myAttr");
      auto attrSet = writer->CreateAttributeSet(std::move(attrModel), "MyAttrSet");

      auto attrRange = attrSet->BeginRange();
      *pAttr = "This is range is empty.";
      // No values written to the main RNTuple...
      attrSet->CommitRange(std::move(attrRange));
   }

   {
      auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());

      // Read back the list of available attribute sets
      auto attrSets = reader->GetDescriptor().GetAttrSetIterable();
      int n = 0;
      for (const auto &attrSet : attrSets) {
         ++n;
         EXPECT_EQ(attrSet.GetName(), "MyAttrSet");
      }
      EXPECT_EQ(n, 1);

      // Fetch a specific attribute set
      auto attrSet = reader->OpenAttributeSet("MyAttrSet");
      EXPECT_EQ(Count(attrSet->GetAttributes()), 1);
      EXPECT_EQ(Count(attrSet->GetAttributes(0)), 0);
      ROOT_EXPECT_WARNING_PARTIAL(EXPECT_EQ(Count(attrSet->GetAttributesInRange(0, reader->GetNEntries())), 0),
                                  "ROOT.NTuple", "empty range");
      ROOT_EXPECT_WARNING_PARTIAL(EXPECT_EQ(Count(attrSet->GetAttributesContainingRange(0, reader->GetNEntries())), 0),
                                  "ROOT.NTuple", "empty range");
   }
}

TEST(RNTupleAttributes, AccessAttrSetReaderAfterClosingMainReader)
{
   FileRaii fileGuard("test_ntuple_attrs_readerafterclosing.root");
   ROOT::TestSupport::CheckDiagsRAII diagsRaii;
   diagsRaii.requiredDiag(kWarning, "ROOT.NTuple", "RNTuple Attributes are experimental", false);

   {
      auto model = RNTupleModel::Create();
      auto pInt = model->MakeField<int>("int");
      auto file = std::unique_ptr<TFile>(TFile::Open(fileGuard.GetPath().c_str(), "RECREATE"));
      auto writer = RNTupleWriter::Append(std::move(model), "ntpl", *file);

      auto attrModel = RNTupleModel::Create();
      auto pAttr = attrModel->MakeField<std::string>("myAttr");
      auto attrSet = writer->CreateAttributeSet(std::move(attrModel), "MyAttrSet");

      auto attrRange = attrSet->BeginRange();
      *pAttr = "This is an attribute";

      for (int i = 0; i < 10; ++i) {
         *pInt = i;
         writer->Fill();
      }
      attrSet->CommitRange(std::move(attrRange));
   }

   {
      auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());

      // Fetch an attribute set
      auto attrSet = reader->OpenAttributeSet("MyAttrSet");

      const auto nEntries = reader->GetNEntries();

      // Close the main reader
      reader.reset();

      // Access the attribute set reader
      EXPECT_EQ(Count(attrSet->GetAttributes()), 1);
      EXPECT_EQ(Count(attrSet->GetAttributes(0)), 1);
      EXPECT_EQ(Count(attrSet->GetAttributesInRange(0, nEntries)), 1);
      EXPECT_EQ(Count(attrSet->GetAttributesContainingRange(0, nEntries)), 1);
   }
}

TEST(RNTupleAttributes, AccessAttrSetWriterAfterClosingMainWriter)
{
   FileRaii fileGuard("test_ntuple_attrs_writerafterclosing.root");
   ROOT::TestSupport::CheckDiagsRAII diagsRaii;
   diagsRaii.requiredDiag(kWarning, "ROOT.NTuple", "RNTuple Attributes are experimental", false);

   {
      auto model = RNTupleModel::Create();
      auto pInt = model->MakeField<int>("int");
      auto file = std::unique_ptr<TFile>(TFile::Open(fileGuard.GetPath().c_str(), "RECREATE"));
      auto writer = RNTupleWriter::Append(std::move(model), "ntpl", *file);

      auto attrModel = RNTupleModel::Create();
      auto pMyAttr = attrModel->MakeField<std::string>("myAttr");
      auto attrSet = writer->CreateAttributeSet(std::move(attrModel), "MyAttrSet");

      diagsRaii.requiredDiag(kWarning, "ROOT.NTuple", "range was not committed", false);

      auto attrEntry = attrSet->BeginRange();

      *pMyAttr = "This is an attribute";
      for (int i = 0; i < 10; ++i) {
         *pInt = i;
         writer->Fill();
      }
      writer.reset();
      EXPECT_THROW(attrSet->CommitRange(std::move(attrEntry)), ROOT::RException);
   }
}

TEST(RNTupleAttributes, ReadAttributesUnknownMajor)
{
   // Try reading an attribute set with an unknown Major schema version and verify it fails.
   FileRaii fileGuard("test_ntuple_attrs_futuremajorschema.root");

   ROOT::TestSupport::CheckDiagsRAII diagsRaii;
   diagsRaii.requiredDiag(kWarning, "ROOT.NTuple", "RNTuple Attributes are experimental", false);

   // Write a regular RNTuple with attributes
   {
      auto file = std::unique_ptr<TFile>(TFile::Open(fileGuard.GetPath().c_str(), "RECREATE"));
      auto opts = RNTupleWriteOptions();
      opts.SetCompression(0);
      auto writer = RNTupleWriter::Append(RNTupleModel::Create(), "ntpl", *file, opts);
      auto attrSet = writer->CreateAttributeSet(RNTupleModel::Create(), "MyAttrSet");

      auto attrRange = attrSet->BeginRange();
      for (int i = 0; i < 10; ++i) {
         writer->Fill();
      }
      attrSet->CommitRange(std::move(attrRange));
   }

   // Get metadata about the main footer that we're gonna patch.
   std::uint64_t footerSeek = 0, footerNBytes = 0;
   {
      auto file = std::unique_ptr<TFile>(TFile::Open(fileGuard.GetPath().c_str()));
      const ROOT::RNTuple *mainAnchor = file->Get<ROOT::RNTuple>("ntpl");
      ASSERT_NE(mainAnchor, nullptr);

      footerSeek = mainAnchor->GetSeekFooter();
      footerNBytes = mainAnchor->GetNBytesFooter() - 8;
   }

   // Patch the attribute schema version (and update the footer checksum)
   const std::uint64_t majorOff = 0xA0;
   const std::byte newMajorVersion{0x99};
   PatchRNTupleSection(fileGuard.GetPath(), footerSeek, footerNBytes, majorOff, &newMajorVersion,
                       sizeof(newMajorVersion), EEndianness::LE);

   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());

   // Fetch an attribute set
   try {
      auto attrSet = reader->OpenAttributeSet("MyAttrSet");
      FAIL() << "opening an attribute set with an unknown major version should fail.";
   } catch (const ROOT::RException &ex) {
      EXPECT_THAT(ex.what(), testing::HasSubstr("unsupported attribute schema version"));
   }
}

TEST(RNTupleAttributes, ReadAttributesUnknownMinor)
{
   // Try reading an attribute set with an unknown Minor schema version and verify it works.
   FileRaii fileGuard("test_ntuple_attrs_futureminorschema.root");

   ROOT::TestSupport::CheckDiagsRAII diagsRaii;
   diagsRaii.requiredDiag(kWarning, "ROOT.NTuple", "RNTuple Attributes are experimental", false);

   // Write a regular RNTuple with attributes
   {
      auto file = std::unique_ptr<TFile>(TFile::Open(fileGuard.GetPath().c_str(), "RECREATE"));
      auto opts = RNTupleWriteOptions();
      opts.SetCompression(0);
      auto writer = RNTupleWriter::Append(RNTupleModel::Create(), "ntpl", *file, opts);
      auto attrSet = writer->CreateAttributeSet(RNTupleModel::Create(), "MyAttrSet");

      auto attrRange = attrSet->BeginRange();
      for (int i = 0; i < 10; ++i) {
         writer->Fill();
      }
      attrSet->CommitRange(std::move(attrRange));
   }

   // Get metadata about the main footer that we're gonna patch.
   std::uint64_t footerSeek = 0, footerNBytes = 0;
   {
      auto file = std::unique_ptr<TFile>(TFile::Open(fileGuard.GetPath().c_str()));
      const ROOT::RNTuple *mainAnchor = file->Get<ROOT::RNTuple>("ntpl");
      ASSERT_NE(mainAnchor, nullptr);

      footerSeek = mainAnchor->GetSeekFooter();
      footerNBytes = mainAnchor->GetNBytesFooter() - 8;
   }

   // Patch the attribute schema version (and update the footer checksum)
   const std::uint64_t majorOff = 0xA2;
   const std::byte newMinorVersion{0x99};
   PatchRNTupleSection(fileGuard.GetPath(), footerSeek, footerNBytes, majorOff, &newMinorVersion,
                       sizeof(newMinorVersion), EEndianness::LE);

   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
   EXPECT_NO_THROW(reader->OpenAttributeSet("MyAttrSet"));
}
