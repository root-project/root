#include <ROOT/RFieldUtils.hxx>
#include "SoAField.hxx"
#include "SoAFieldXML.h"

#include <TClass.h>

#include "gtest/gtest.h"

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
