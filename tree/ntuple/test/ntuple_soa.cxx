#include <ROOT/RError.hxx>
#include <ROOT/RField.hxx>
#include <ROOT/RFieldUtils.hxx>
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
      FAIL() << "creating SoA field with missing record typedef should fail";
   } catch (const ROOT::RException &e) {
      EXPECT_THAT(e.what(), ::testing::HasSubstr("invalid record type of SoA field SoAUnknownRecord"));
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
   EXPECT_NO_THROW(std::make_unique<RSoAField>("f", "SoABase"));

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

   EXPECT_NO_THROW(std::make_unique<RSoAField>("f", "SoASimple"));

   try {
      auto f = std::make_unique<ROOT::RField<SoA>>("f");
      FAIL() << "creating an RClassField on a SoA type should fail";
   } catch (const ROOT::RException &e) {
      EXPECT_THAT(e.what(), ::testing::HasSubstr("SoA is a SoA field and connot be used through RClassField"));
   }
}
