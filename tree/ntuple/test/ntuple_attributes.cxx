#include "ntuple_test.hxx"
#include <ROOT/RNTupleAttributes.hxx>

TEST(RNTupleAttributes, CreateWriter)
{
   FileRaii fileGuard("ntuple_attr_create_writer.root");

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
