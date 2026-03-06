// Test that ROOT_LOG env var correctly configures RLogger channel verbosity.
// ROOT_LOG is parsed once at RLogManager construction (process startup),
// so each test case is a separate executable launched with the env var set.

#include "ROOT/RLogger.hxx"
#include "gtest/gtest.h"

// Declare a test channel the same way ROOT modules do
ROOT::RLogChannel &TestChannel()
{
   static ROOT::RLogChannel channel("ROOT.TestChannel");
   return channel;
}

// Test: channel verbosity set via ROOT_LOG is reflected in GetEnvVerbosity
TEST(RLoggerEnvVar, EnvVerbosityIsStored)
{
   // ROOT_LOG was set to 'ROOT.TestChannel=Error' before process start
   // (set in CMakeLists via set_tests_properties)
   auto level = ROOT::RLogManager::Get().GetEnvVerbosity("ROOT.TestChannel");
   EXPECT_EQ(level, ROOT::ELogLevel::kError);
}

// Test: unknown channel returns kUnset
TEST(RLoggerEnvVar, UnknownChannelReturnsUnset)
{
   auto level = ROOT::RLogManager::Get().GetEnvVerbosity("ROOT.DoesNotExist");
   EXPECT_EQ(level, ROOT::ELogLevel::kUnset);
}

// Test: channel effective verbosity uses env var when channel has no explicit level
TEST(RLoggerEnvVar, EffectiveVerbosityUsesEnvVar)
{
   // Channel has no explicit verbosity set, so should use env var value
   auto effective = TestChannel().GetEffectiveVerbosity(ROOT::RLogManager::Get());
   EXPECT_EQ(effective, ROOT::ELogLevel::kError);
}
// Test: explicitly set verbosity on a channel takes precedence over ROOT_LOG env var.
// ROOT_LOG sets ROOT.TestChannel=Error, but we explicitly set it to kInfo here.
// The explicit setting should win.
TEST(RLoggerEnvVar, ExplicitVerbosityTakesPrecedenceOverEnvVar)
{
   // ROOT_LOG set ROOT.TestChannel=Error via environment
   // Now explicitly override it to kInfo
   TestChannel().SetVerbosity(ROOT::ELogLevel::kInfo);

   // Explicit verbosity should win over env var
   EXPECT_EQ(TestChannel().GetEffectiveVerbosity(ROOT::RLogManager::Get()), ROOT::ELogLevel::kInfo);

   // Reset back to kUnset so other tests are not affected
   TestChannel().SetVerbosity(ROOT::ELogLevel::kUnset);
}