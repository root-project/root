#include "TStatusBitsChecker.h"

#include "TClass.h"
#include "TInterpreter.h"
#include "ROOT/TestSupport.hxx"

#include "gtest/gtest.h"

const char* gCode = R"CODE(
   struct ClassWithOverlap : public TObject {
      enum EStatusBits {
         kOverlappingBit = BIT(13)
      };
   };
)CODE";

void MakeClassWithOverlap() {
   gInterpreter->Declare(gCode);
}

TEST(StatusBitsChecker,NoOverlap)
{
   EXPECT_TRUE(ROOT::Detail::TStatusBitsChecker::Check("TObject"));
   EXPECT_TRUE(ROOT::Detail::TStatusBitsChecker::Check("TNamed"));
   EXPECT_TRUE(ROOT::Detail::TStatusBitsChecker::Check("TStreamerElement"));
}

TEST(StatusBitsChecker,Overlap)
{
   ROOT::TestSupport::CheckDiagsRAII diags;
   diags.requiredDiag(kError, "TStatusBitsChecker", "ClassWithOverlap", /*matchFullMessage=*/false);
   diags.requiredDiag(kError, "TStatusBitsChecker", "13 used in",       /*matchFullMessage=*/false);

   MakeClassWithOverlap();
   EXPECT_NE(nullptr,TClass::GetClass("ClassWithOverlap"));
   EXPECT_FALSE(ROOT::Detail::TStatusBitsChecker::Check("ClassWithOverlap"));
}
