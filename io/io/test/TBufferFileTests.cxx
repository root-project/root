#include "gtest/gtest.h"

#include "TBufferFile.h"
#include "TMemFile.h"
#include "TClass.h"
#include "TInterpreter.h"
#include <vector>
#include <iostream>

// Tests ROOT-8367
TEST(TBufferFile, ROOT_8367)
{
   std::vector<int> v{1, 2, 3, 4, 5, 6, 7};

   TClass *cl1 = TClass::GetClass("vector<int>");
   TClass *cl2 = TClass::GetClass("vector<float>");

   TBufferFile *m_buff = new TBufferFile(TBuffer::kWrite);
   m_buff->WriteObjectAny(&v, cl1);
   m_buff->SetReadMode();
   m_buff->Reset();
   void *obj = m_buff->ReadObjectAny(cl2);
   std::vector<float> &v2 = *reinterpret_cast<std::vector<float> *>(obj);
   EXPECT_FLOAT_EQ(v2[0], 1.);
   EXPECT_FLOAT_EQ(v2[1], 2.);
   EXPECT_FLOAT_EQ(v2[2], 3.);
   EXPECT_FLOAT_EQ(v2[5], 6.);
   EXPECT_FLOAT_EQ(v2[6], 7.);
   EXPECT_EQ(v2.size(), 7);
}

// https://github.com/root-project/root/issues/19371
#define MYSTRUCT0 struct MyS0 { static constexpr short Class_Version() { return 0; } };
#define MYSTRUCT1 struct MyS1 { static constexpr short Class_Version() { return 1; } };
MYSTRUCT0
MYSTRUCT1
#define TO_LITERAL(string) _QUOTE_(string)
TEST(TBufferFile, ForeignZeroVersionClass)
{
   gInterpreter->Declare(TO_LITERAL(MYSTRUCT0));
   gInterpreter->Declare(TO_LITERAL(MYSTRUCT1));
   MyS0 s0;
   MyS1 s1;
   TMemFile f("mem19371.root", "RECREATE");
   f.WriteObject(&s0, "s0");
   f.WriteObject(&s1, "s1");
   EXPECT_NE(f.Get<MyS0>("s0"), nullptr); // Before the fix, even if return was already not nullptr, this line was raising an Error thus test would fail with unexpected diagnostic
   EXPECT_NE(f.Get<MyS1>("s1"), nullptr);
}
