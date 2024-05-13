#include "TClass.h"
#include "TDataMember.h"

#include "gtest/gtest.h"

TEST(TClingDataMemberInfo, issue8553)
{
   TClass *clBox = nullptr;
   ASSERT_TRUE(clBox = TClass::GetClass("ROOT::Experimental::RBox"));
   TDataMember *dmAttrBorder = nullptr;
   ASSERT_TRUE(dmAttrBorder = clBox->GetDataMember("border"));
   EXPECT_NE(dmAttrBorder->GetOffsetCint(), 0);
}
