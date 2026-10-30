#include "gtest/gtest.h"

#include "TClass.h"
#include "TDataMember.h"
#include "TInterpreter.h"
#include "TList.h"

TEST(TDataMemberTests, ROOT8499) {
  gInterpreter->Declare(
" \
template <typename T> \
class ROOT8499 \
{ \
public: \
  static int initHelper()  { return 0;  }; \
  static const int s_info; \
}; \
\
template <class T> \
const int ROOT8499<T>::s_info = ROOT8499<T>::initHelper();");

  ASSERT_NE(nullptr, TClass::GetClass("ROOT8499<int>"));
  TClass* cl = TClass::GetClass("ROOT8499<int>");
  ASSERT_NE(nullptr, cl->GetListOfDataMembers()->At(0));
  TDataMember* dm = (TDataMember*) cl->GetListOfDataMembers()->At(0);

  // As used in bindings/pyroot/src/Cppyy.cxx:
  EXPECT_NE(0, dm->GetOffsetCint());
}

TEST(TDataMemberTests, Issue23777)
{
   // Test that calling GetOffset() does not corrupt or interfere with GetOffsetCint()
   TClass *cl = TClass::GetClass("TAttImage");
   ASSERT_NE(nullptr, cl);
   TObject *obj = cl->GetListOfDataMembers()->FindObject("kImgBest");
   ASSERT_NE(nullptr, obj);
   TDataMember *dm = dynamic_cast<TDataMember *>(obj);
   ASSERT_NE(nullptr, dm);

   EXPECT_STREQ("TAttImage::EImageQuality", dm->GetTypeName());
   // Static and enum members do not have an instance offset; GetOffset() must return -1
   EXPECT_EQ(-1, dm->GetOffset());
   // GetOffsetCint() must return a non-zero address/offset from Cling even after GetOffset() was called
   EXPECT_NE(0, dm->GetOffsetCint());
   // GetOffset() must still return -1 even after GetOffsetCint() cached the address
   EXPECT_EQ(-1, dm->GetOffset());
}
