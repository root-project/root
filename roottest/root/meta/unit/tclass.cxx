#include "gtest/gtest.h"

#include "TClass.h"
#include "TInterpreter.h"

TEST(TClassTests, DeclFileName) {
  gInterpreter->Declare("class TEST_TClassUnitTest {};");

  ASSERT_NE(nullptr, TClass::GetClass("TEST_TClassUnitTest"));
  TClass* cl = TClass::GetClass("TEST_TClassUnitTest");

  EXPECT_EQ(-1, cl->GetDeclFileLine());
  EXPECT_STREQ("", cl->GetDeclFileName()); // ROOT-7526

  EXPECT_EQ(-1, cl->GetImplFileLine());
  EXPECT_STREQ("", cl->GetImplFileName());
}

