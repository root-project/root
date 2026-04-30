#include <ROOT/RError.hxx>
#include <ROOT/RField.hxx>
#include <ROOT/RFieldUtils.hxx>
#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleReader.hxx>
#include <ROOT/RNTupleView.hxx>
#include <ROOT/RNTupleWriter.hxx>
#include <ROOT/TestSupport.hxx>

#include "SoAField.hxx"
#include "SoAFieldXML.h"

#include <TClass.h>

#include <memory>
#include <utility>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

using ROOT::Experimental::RSoAField;

TEST(RNTuple, SoADict)
{
   auto cl = TClass::GetClass("Record");
   ASSERT_NE(cl, nullptr);
   EXPECT_TRUE(ROOT::Internal::GetRNTupleSoARecord(cl).empty());

   cl = TClass::GetClass("SoA");
   ASSERT_NE(cl, nullptr);
   EXPECT_EQ("Record", ROOT::Internal::GetRNTupleSoARecord(cl));

   cl = TClass::GetClass("RecordXML");
   ASSERT_NE(cl, nullptr);
   EXPECT_TRUE(ROOT::Internal::GetRNTupleSoARecord(cl).empty());

   cl = TClass::GetClass("SoAXML");
   ASSERT_NE(cl, nullptr);
   EXPECT_EQ("RecordXML", ROOT::Internal::GetRNTupleSoARecord(cl));
}

TEST(RNTuple, SoACheck)
{
   ROOT::TestSupport::CheckDiagsRAII diagRAII;
   diagRAII.requiredDiag(kWarning, "[ROOT.NTuple]", "The SoA field is experimental and still under development.",
                         true /* matchFullMessage */);

   try {
      auto f = std::make_unique<RSoAField>("f", "Record");
      FAIL() << "creating SoA field for untagged class should fail";
   } catch (const ROOT::RException &e) {
      EXPECT_THAT(e.what(), ::testing::HasSubstr("not marked with the rntupleSoARecord dictionary option"));
   }

   try {
      auto f = std::make_unique<RSoAField>("f", "SoAUnknownRecord");
      FAIL() << "creating SoA field with unknown underlying record type should fail";
   } catch (const ROOT::RException &e) {
      EXPECT_THAT(e.what(), ::testing::HasSubstr("invalid record type of SoA field SoAUnknownRecord"));
   }

   try {
      auto f = std::make_unique<RSoAField>("f", "SoAVersionMismatch");
      FAIL() << "creating SoA field with a class version different from the underlying record type's should fail";
   } catch (const ROOT::RException &e) {
      EXPECT_THAT(e.what(), ::testing::HasSubstr("version mismatch between SoA type and underlying record type"));
   }

   try {
      auto f = std::make_unique<RSoAField>("f", "SoAOnDerivedRecord");
      FAIL() << "creating SoA field on derived record should fail";
   } catch (const ROOT::RException &e) {
      EXPECT_THAT(e.what(), ::testing::HasSubstr("SoA fields with inheritance are currently unsupported"));
   }
   try {
      auto f = std::make_unique<RSoAField>("f", "SoADerivedOnBaseRecord");
      FAIL() << "creating a derived SoA field should fail";
   } catch (const ROOT::RException &e) {
      EXPECT_THAT(e.what(), ::testing::HasSubstr("SoA fields with inheritance are currently unsupported"));
   }
   {
      EXPECT_NO_THROW(auto f = std::make_unique<RSoAField>("f", "SoABase"));
   }

   try {
      auto f = std::make_unique<RSoAField>("f", "SoASimpleBadArray");
      FAIL() << "creating SoA field with arrays fail";
   } catch (const ROOT::RException &e) {
      EXPECT_THAT(e.what(), ::testing::HasSubstr("unsupported array type in SoA class: fY"));
   }
   try {
      auto f = std::make_unique<RSoAField>("f", "SoASimpleBadType");
      FAIL() << "creating SoA field with vectors instead of RVecs should fail";
   } catch (const ROOT::RException &e) {
      EXPECT_THAT(e.what(), ::testing::HasSubstr("invalid field type in SoA class: std::vector<float>"));
   }
   try {
      auto f = std::make_unique<RSoAField>("f", "SoASimpleUnexpectedMember");
      FAIL() << "creating SoA field with an unexpected member should fail";
   } catch (const ROOT::RException &e) {
      EXPECT_THAT(e.what(), ::testing::HasSubstr("unexpected SoA member: fZ"));
   }
   try {
      auto f = std::make_unique<RSoAField>("f", "SoASimpleMissingMember");
      FAIL() << "creating SoA field with a missing member should fail";
   } catch (const ROOT::RException &e) {
      EXPECT_THAT(e.what(), ::testing::HasSubstr("missing SoA members"));
   }
   try {
      auto f = std::make_unique<RSoAField>("f", "SoASimpleWrongMember");
      FAIL() << "creating SoA field with a member of wrong type should fail";
   } catch (const ROOT::RException &e) {
      EXPECT_THAT(e.what(), ::testing::HasSubstr("SoA member type mismatch: fY (double [Double32_t] vs. float)"));
   }

   {
      EXPECT_NO_THROW(auto f = std::make_unique<RSoAField>("f", "SoASimple"));
   }

   try {
      auto f = std::make_unique<ROOT::RField<SoA>>("f");
      FAIL() << "creating an RClassField on a SoA type should fail";
   } catch (const ROOT::RException &e) {
      EXPECT_THAT(e.what(), ::testing::HasSubstr("SoA is a SoA field and connot be used through RClassField"));
   }
}

TEST(RNTuple, SoADescriptor)
{
   ROOT::TestSupport::FileRaii fileGuard("test_rntuple_soa_descriptor.root");

   {
      auto model = ROOT::RNTupleModel::Create();
      auto f1 = std::make_unique<RSoAField>("f1", "SoA");
      EXPECT_TRUE(f1->GetTraits() & ROOT::RFieldBase::kTraitSoACollection);
      model->AddField(std::move(f1));
      auto f2 = ROOT::RFieldBase::Create("f2", "SoA").Unwrap();
      EXPECT_TRUE(f2->GetTraits() & ROOT::RFieldBase::kTraitSoACollection);
      model->AddField(std::move(f2));
      model->MakeField<std::vector<Record>>("f3");
      auto writer = ROOT::RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
   }

   auto reader = ROOT::RNTupleReader::Open("ntpl", fileGuard.GetPath());
   const auto &desc = reader->GetDescriptor();
   const auto &f1Desc = desc.GetFieldDescriptor(desc.FindFieldId("f1"));
   EXPECT_TRUE(f1Desc.IsSoACollection());
   EXPECT_EQ(ROOT::ENTupleStructure::kCollection, f1Desc.GetStructure());
   EXPECT_EQ(TClass::GetClass("SoA")->GetClassVersion(), f1Desc.GetTypeVersion());
   EXPECT_TRUE(f1Desc.GetTypeChecksum());
   EXPECT_EQ(TClass::GetClass("SoA")->GetCheckSum(), *f1Desc.GetTypeChecksum());
   const auto &f2Desc = desc.GetFieldDescriptor(desc.FindFieldId("f2"));
   EXPECT_TRUE(f2Desc.IsSoACollection());
   EXPECT_EQ(TClass::GetClass("SoA")->GetClassVersion(), f2Desc.GetTypeVersion());
   EXPECT_EQ(ROOT::ENTupleStructure::kCollection, f2Desc.GetStructure());
   EXPECT_TRUE(f2Desc.GetTypeChecksum());
   EXPECT_EQ(TClass::GetClass("SoA")->GetCheckSum(), *f2Desc.GetTypeChecksum());
   const auto &f3Desc = desc.GetFieldDescriptor(desc.FindFieldId("f3"));
   EXPECT_FALSE(f3Desc.IsSoACollection());
   EXPECT_EQ(0u, f3Desc.GetTypeVersion());
   EXPECT_EQ(ROOT::ENTupleStructure::kCollection, f3Desc.GetStructure());
   EXPECT_FALSE(f3Desc.GetTypeChecksum());
}

TEST(RNTuple, SoAEmpty)
{
   ROOT::TestSupport::FileRaii fileGuard("test_rntuple_soa_empty.root");

   {
      auto model = ROOT::RNTupleModel::Create();
      model->AddField(std::make_unique<RSoAField>("f", "SoA"));
      auto writer = ROOT::RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
      writer->Fill();
      writer->CommitCluster();
      writer->Fill();
      writer->Fill();
   }

   auto reader = ROOT::RNTupleReader::Open("ntpl", fileGuard.GetPath());
   EXPECT_EQ(3u, reader->GetNEntries());
   auto v = reader->GetView<std::vector<Record>>("f");
   EXPECT_EQ(0u, v(0).size());
   EXPECT_EQ(0u, v(1).size());
   EXPECT_EQ(0u, v(2).size());
}

TEST(RNTuple, SoASimple)
{
   ROOT::TestSupport::FileRaii fileGuard("test_rntuple_soa_simple.root");

   {
      auto model = ROOT::RNTupleModel::Create();
      model->AddField(std::make_unique<RSoAField>("f", "SoASimple"));
      auto writer = ROOT::RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
      auto soa = writer->GetModel().GetDefaultEntry().GetPtr<SoASimple>("f");
      soa->fX.push_back(1.0);
      soa->fY.push_back(2.0);
      writer->Fill();
      soa->fX.clear();
      // Filling in general is not exception-safe, but filling a single SoA field is
      EXPECT_THROW(writer->Fill(), ROOT::RException);
      soa->fY.clear();
      writer->Fill();
      writer->CommitCluster();
      soa->fX.push_back(3.0);
      soa->fY.push_back(4.0);
      soa->fX.push_back(5.0);
      soa->fY.push_back(6.0);
      writer->Fill();
   }

   auto reader = ROOT::RNTupleReader::Open("ntpl", fileGuard.GetPath());
   EXPECT_EQ(3u, reader->GetNEntries());
   auto v = reader->GetView<std::vector<RecordSimple>>("f");
   EXPECT_EQ(1u, v(0).size());
   EXPECT_FLOAT_EQ(1.0, v(0).at(0).fX);
   EXPECT_FLOAT_EQ(2.0, v(0).at(0).fY);
   EXPECT_EQ(0u, v(1).size());
   EXPECT_EQ(2u, v(2).size());
   EXPECT_FLOAT_EQ(3.0, v(2).at(0).fX);
   EXPECT_FLOAT_EQ(4.0, v(2).at(0).fY);
   EXPECT_FLOAT_EQ(5.0, v(2).at(1).fX);
   EXPECT_FLOAT_EQ(6.0, v(2).at(1).fY);

   auto card = reader->GetView<ROOT::RNTupleCardinality<std::uint64_t>>("f");
   EXPECT_EQ(1u, card(0));
   EXPECT_EQ(0u, card(1));
   EXPECT_EQ(2u, card(2));
}

TEST(RNTuple, SoASimpleSwapped)
{
   ROOT::TestSupport::FileRaii fileGuard("test_rntuple_soa_simple_swapped.root");

   {
      auto model = ROOT::RNTupleModel::Create();
      model->AddField(std::make_unique<RSoAField>("f", "SoASimpleSwapped"));
      auto writer = ROOT::RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
      auto soa = writer->GetModel().GetDefaultEntry().GetPtr<SoASimpleSwapped>("f");
      soa->fX.push_back(1.0);
      soa->fY.push_back(2.0);
      writer->Fill();
   }

   auto reader = ROOT::RNTupleReader::Open("ntpl", fileGuard.GetPath());
   EXPECT_EQ(1u, reader->GetNEntries());
   auto v = reader->GetView<std::vector<RecordSimple>>("f");
   EXPECT_EQ(1u, v(0).size());
   EXPECT_FLOAT_EQ(1.0, v(0).at(0).fX);
   EXPECT_FLOAT_EQ(2.0, v(0).at(0).fY);
}
