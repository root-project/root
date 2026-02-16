#include "ntuple_test.hxx"
#include <ROOT/RNTupleAttributes.hxx>

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

TEST(RNTupleAttributes, AttributeInvalidModel)
{
   FileRaii fileGuard("test_ntuple_attrs_invalid_model.root");

   // Create a RNTuple
   auto model = RNTupleModel::Create();
   auto pInt = model->MakeField<int>("int");
   auto file = std::unique_ptr<TFile>(TFile::Open(fileGuard.GetPath().c_str(), "RECREATE"));
   auto writer = RNTupleWriter::Append(std::move(model), "ntpl", *file);

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
