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

   fileGuard.PreserveFile();
}
