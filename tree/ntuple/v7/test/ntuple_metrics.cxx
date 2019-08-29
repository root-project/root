#include "gtest/gtest.h"

#include <ROOT/RNTupleMetrics.hxx>

#include <chrono>
#include <thread>

using RNTuplePlainCounter = ROOT::Experimental::Detail::RNTuplePlainCounter;
using RNTupleAtomicCounter = ROOT::Experimental::Detail::RNTupleAtomicCounter;
using RNTuplePlainTimer = ROOT::Experimental::Detail::RNTuplePlainTimer;
using RNTupleAtomicTimer = ROOT::Experimental::Detail::RNTupleAtomicTimer;
using RNTupleMetrics = ROOT::Experimental::Detail::RNTupleMetrics;

TEST(Metrics, Counters)
{
	RNTupleMetrics metrics("test");
	EXPECT_FALSE(metrics.IsEnabled());

	RNTuplePlainCounter *ctrOne = nullptr;
	RNTupleAtomicCounter *ctrTwo = nullptr;
	metrics.MakeCounter("plain", "s", "example 1", ctrOne);
	metrics.MakeCounter("atomic", "s", "example 2", ctrTwo);
	ASSERT_NE(nullptr, ctrOne);
	ASSERT_NE(nullptr, ctrTwo);
	EXPECT_TRUE(ctrOne->IsEnabled());
	EXPECT_FALSE(ctrTwo->IsEnabled());

	EXPECT_EQ(0, ctrOne->GetValue());
	ctrOne->Inc();
	ctrTwo->Inc();
	EXPECT_EQ(1, ctrOne->GetValue());
	EXPECT_EQ(0, ctrTwo->GetValue());
	metrics.Enable();
	EXPECT_TRUE(metrics.IsEnabled());
	ctrTwo->Inc();
	EXPECT_EQ(1, ctrTwo->XAdd(5));
	EXPECT_EQ(1, ctrOne->GetValue());
	EXPECT_EQ(6, ctrTwo->GetValue());
}

TEST(Metrics, Timer)
{
	RNTupleAtomicCounter ctrWallTime("wall time", "ns", "");
	ROOT::Experimental::Detail::RNTupleTickCounter<RNTupleAtomicCounter> ctrCpuTicks("cpu time", "ns", "");
	{
		RNTupleAtomicTimer timer(ctrWallTime, ctrCpuTicks);
		std::this_thread::sleep_for(std::chrono::milliseconds(10));
	}
	EXPECT_EQ(0U, ctrWallTime.GetValue());
	EXPECT_EQ(0U, ctrCpuTicks.GetValue());
	ctrWallTime.Enable();
	ctrCpuTicks.Enable();
	{
		RNTupleAtomicTimer timer(ctrWallTime, ctrCpuTicks);
		std::this_thread::sleep_for(std::chrono::milliseconds(10));
	}
	EXPECT_GT(ctrWallTime.GetValue(), 0U);
}
