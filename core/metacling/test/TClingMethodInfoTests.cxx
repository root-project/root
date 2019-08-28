#include "TClass.h"
#include "TList.h"
#include "TMethod.h"

#include "gtest/gtest.h"

TEST(TClingMethodInfo, Prototype)
{
  TClass *cl = TClass::GetClass("TObject");
  ASSERT_NE(cl, nullptr);
  TMethod *meth = (TMethod*)cl->GetListOfMethods()->FindObject("SysError");
  ASSERT_NE(meth, nullptr);
  EXPECT_STREQ(meth->GetPrototype(), "void TObject::SysError(const char* method, const char* msgfmt,...) const");
}
