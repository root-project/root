#include "gtest/gtest.h"

#include "TClass.h"
#include "TInterpreter.h"
#include "TList.h"
#include "TMethod.h"

TEST(TemplateFuncTests, ROOT8542) {
  gInterpreter->Declare("#include <RStringView.h>");

  ASSERT_NE(nullptr, TClass::GetClass("std::string_view"));
  TClass* cl = TClass::GetClass("std::string_view");

  TCollection* svFuncs = cl->GetListOfMethods();
  ASSERT_NE(nullptr, svFuncs);
  EXPECT_LT(5, svFuncs->GetSize());
  ASSERT_NE(nullptr, svFuncs->FindObject("to_string")); // ROOT-8542
  TMethod* mToStr = (TMethod*)svFuncs->FindObject("to_string");
  EXPECT_STREQ("string", mToStr->GetReturnTypeNormalizedName().c_str());
}

