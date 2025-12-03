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

