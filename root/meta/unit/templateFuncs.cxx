#include "gtest/gtest.h"

#include "TClass.h"
#include "TInterpreter.h"
#include "TList.h"
#include "TMethod.h"

TEST(TemplateFuncTests, ROOT8542) {
  gInterpreter->Declare("template <int I, class T = float> class ROOT8542{ template <class X = T> X func(); };");

  ASSERT_NE(nullptr, TClass::GetClass("ROOT8542<42>"));
  TClass* cl = TClass::GetClass("ROOT8542<42>");

  TCollection* funcs = cl->GetListOfMethods();
  ASSERT_NE(nullptr, funcs);
  /*
int ROOT8542<42>::func<float>()
ROOT8542<42> ROOT8542<42>::ROOT8542<42>()
ROOT8542<42> ROOT8542<42>::ROOT8542<42>(const ROOT8542<42>&)
ROOT8542<42>& ROOT8542<42>::operator=(const ROOT8542<42>&)
ROOT8542<42> ROOT8542<42>::ROOT8542<42>(ROOT8542<42>&&)
ROOT8542<42>& ROOT8542<42>::operator=(ROOT8542<42>&&)
void ROOT8542<42>::~ROOT8542<42>()
  */
  EXPECT_EQ(7, funcs->GetSize());
  ASSERT_NE(nullptr, funcs->FindObject("func")); // ROOT-8542
  TMethod* func = (TMethod*)funcs->FindObject("func");
  EXPECT_STREQ("float", func->GetReturnTypeNormalizedName().c_str());
}

