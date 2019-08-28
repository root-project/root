#include "gtest/gtest.h"

#include <ROOT/RNTupleMetrics.hxx>

#include <chrono>
#include <thread>

using RNTuplePerfCounter = ROOT::Experimental::Detail::RNTuplePerfCounter;
using RNTupleTickCounter = ROOT::Experimental::Detail::RNTupleTickCounter;
using RNTupleMetrics = ROOT::Experimental::Detail::RNTupleMetrics;
using RNTupleTimer = ROOT::Experimental::Detail::RNTupleTimer;

TEST(Metrics, Counters)
{
	RNTupleMetrics metrics("test");
	EXPECT_FALSE(metrics.IsActive());

	auto ctrOne = metrics.Generate("one", "s", "example 1");
	auto ctrTwo = metrics.Generate("two", "m", "example 2");

	EXPECT_EQ(0, ctrOne->GetValue());
	ctrOne->Inc();
	EXPECT_EQ(0, ctrOne->GetValue());
	metrics.Activate();
	EXPECT_TRUE(metrics.IsActive());
	ctrOne->Inc();
	EXPECT_EQ(0, ctrTwo->XAdd(5));
	EXPECT_EQ(1, ctrOne->GetValue());
	EXPECT_EQ(5, ctrTwo->GetValue());
}

TEST(Metrics, Timer)
{
	RNTuplePerfCounter ctrWallTime;
	RNTupleTickCounter ctrCpuTicks;
	{
		RNTupleTimer timer(ctrWallTime, ctrCpuTicks);
		std::this_thread::sleep_for(std::chrono::milliseconds(10));
	}
	EXPECT_EQ(0U, ctrWallTime.GetValue());
	EXPECT_EQ(0U, ctrCpuTicks.GetValue());
	ctrWallTime.Activate();
	ctrCpuTicks.Activate();
	{
		RNTupleTimer timer(ctrWallTime, ctrCpuTicks);
		std::this_thread::sleep_for(std::chrono::milliseconds(10));
	}
	EXPECT_GT(ctrWallTime.GetValue(), 0U);
}
