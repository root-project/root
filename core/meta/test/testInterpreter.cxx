#include "TInterpreter.h"
#include "TSystem.h"

#include "gtest/gtest.h"
#include "gmock/gmock.h"

TEST(TInterpreter, ErrnoValue)
{
	gInterpreter->ProcessLine("errno");
	EXPECT_EQ(0, TSystem::GetErrno());
}