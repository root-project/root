#include "TDataType.h"
#include "TInterpreter.h"

#include "gtest/gtest.h"

TEST(TDataType, GetType)
{
   auto interpreted = gInterpreter->Calc("TDataType::GetType(typeid(int))");
   auto compiled = TDataType::GetType(typeid(int));

   EXPECT_EQ(kInt_t, interpreted);
   EXPECT_EQ(kInt_t, compiled);
}
