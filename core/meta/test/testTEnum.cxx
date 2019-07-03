#include "TEnum.h"
#include "TInterpreter.h"

#include "gtest/gtest.h"
#include "gmock/gmock.h"

TEST(TEnum, UnderlyingType)
{
   gInterpreter->Declare(R"CODE(
enum E0 { kE0One };
enum E1 { kE1One = LONG_MAX };
enum E2 { kE2One = ULONG_MAX };
enum E3: char { kE3One };

enum Eb: bool { kEbOne };
enum Euc: unsigned char { kEucOne };
enum Esc: signed char { kEscOne };
enum Eus: unsigned short { kEusOne };
enum Ess: signed short { kEssOne };
enum Eui: unsigned int { kEuiOne };
enum Esi: signed int { kEsiOne };
enum Eul: unsigned long { kEulOne };
enum Esl: signed long { kEslOne };
enum Eull: unsigned long long { kEullOne };
enum Esll: signed long long { kEsllOne };

enum Ecl: short;

enum class ECb: bool { kOne };
enum class ECuc: unsigned char { kOne };
enum class ECsc: signed char { kOne };
enum class ECus: unsigned short { kOne };
enum class ECss: signed short { kOne };
enum class ECui: unsigned int { kOne };
enum class ECsi: signed int { kOne };
enum class ECul: unsigned long { kOne };
enum class ECsl: signed long { kOne };
enum class ECull: unsigned long long { kOne };
enum class ECsll: signed long long { kOne };

enum class ECcl: short;
)CODE"
			);

   EXPECT_EQ(TEnum::GetEnum("E0")->GetUnderlyingType(), kUInt_t);
   EXPECT_EQ(TEnum::GetEnum("E1")->GetUnderlyingType(), kULong_t);
   EXPECT_EQ(TEnum::GetEnum("E2")->GetUnderlyingType(), kULong_t);
   EXPECT_EQ(TEnum::GetEnum("E3")->GetUnderlyingType(), kChar_t);

   EXPECT_EQ(TEnum::GetEnum("Eb")->GetUnderlyingType(), kBool_t);
   EXPECT_EQ(TEnum::GetEnum("Euc")->GetUnderlyingType(), kUChar_t);
   EXPECT_EQ(TEnum::GetEnum("Esc")->GetUnderlyingType(), kChar_t);
   EXPECT_EQ(TEnum::GetEnum("Eus")->GetUnderlyingType(), kUShort_t);
   EXPECT_EQ(TEnum::GetEnum("Ess")->GetUnderlyingType(), kShort_t);
   EXPECT_EQ(TEnum::GetEnum("Eui")->GetUnderlyingType(), kUInt_t);
   EXPECT_EQ(TEnum::GetEnum("Esi")->GetUnderlyingType(), kInt_t);
   EXPECT_EQ(TEnum::GetEnum("Eul")->GetUnderlyingType(), kULong_t);
   EXPECT_EQ(TEnum::GetEnum("Esl")->GetUnderlyingType(), kLong_t);
   EXPECT_EQ(TEnum::GetEnum("Eull")->GetUnderlyingType(), kULong64_t);
   EXPECT_EQ(TEnum::GetEnum("Esll")->GetUnderlyingType(), kLong64_t);
   EXPECT_EQ(TEnum::GetEnum("Ecl")->GetUnderlyingType(), kShort_t);
}


TEST(TEnum, Scoped)
{
   gInterpreter->Declare(R"CODE(
enum class EC { kOne };
enum class EC1: long { kOne };
enum ED { kEDOne };
)CODE"
			);
   EXPECT_EQ(TEnum::GetEnum("EC")->Property() & kIsScopedEnum, kIsScopedEnum);
   EXPECT_EQ(TEnum::GetEnum("EC1")->Property() & kIsScopedEnum, kIsScopedEnum);
   EXPECT_EQ(TEnum::GetEnum("ED")->Property() & kIsScopedEnum, 0);
}
