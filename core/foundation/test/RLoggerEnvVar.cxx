// Test that ROOT_LOG env var correctly configures RLogger channel verbosity.
// ROOT_LOG is parsed once at RLogManager construction (process startup),
// so the env var is set via ENVIRONMENT in CMakeLists.txt.

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
   auto effective = TestChannel().GetEffectiveVerbosity(ROOT::RLogManager::Get());
   EXPECT_EQ(effective, ROOT::ELogLevel::kError);
}

// Test: explicitly set verbosity on a channel takes precedence over ROOT_LOG env var.
TEST(RLoggerEnvVar, ExplicitVerbosityTakesPrecedenceOverEnvVar)
{
   TestChannel().SetVerbosity(ROOT::ELogLevel::kInfo);
   EXPECT_EQ(TestChannel().GetEffectiveVerbosity(ROOT::RLogManager::Get()), ROOT::ELogLevel::kInfo);
   // Reset back to kUnset so other tests are not affected
   TestChannel().SetVerbosity(ROOT::ELogLevel::kUnset);
}