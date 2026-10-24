#include "ROOT/RLogger.hxx"

#include <TError.h>

#include "gtest/gtest.h"

#include <memory>
#include <vector>

struct TestLogger {
   class Handler : public ROOT::RLogHandler {
      TestLogger *fLogger;

   public:
      Handler(TestLogger &logger) : fLogger(&logger) {}

      bool Emit(const ROOT::RLogEntry &entry) override
      {
         fLogger->fLogEntries.emplace_back(entry);
         // Stop emission.
         return false;
      }
   };

   std::vector<ROOT::RLogEntry> fLogEntries;
   Handler *fHandler;

   TestLogger()
   {
      auto uptr = std::make_unique<Handler>(*this);
      fHandler = uptr.get();
      ROOT::RLogManager::Get().PushFront(std::move(uptr));
   }

   ~TestLogger() { ROOT::RLogManager::Get().Remove(fHandler); }

   size_t size() const { return fLogEntries.size(); }
   bool empty() const { return fLogEntries.empty(); }
};

TEST(Logger, EmittedEntry)
{
   TestLogger testLogger;
   auto prevErrors = ROOT::RLogManager::Get().GetNumErrors();
   auto prevWarnings = ROOT::RLogManager::Get().GetNumWarnings();

   ROOT::RLogChannel channel("TheChannel");

   // clang-format off
   R__LOG_ERROR(channel) << "The text"; auto logLine = __LINE__;
   // clang-format on

   // Check manager's emission
   EXPECT_EQ(testLogger.size(), 1);
   EXPECT_EQ(ROOT::RLogManager::Get().GetNumErrors(), prevErrors + 1);
   EXPECT_EQ(ROOT::RLogManager::Get().GetNumWarnings(), prevWarnings);

   // Check emitted RLogEntry
   EXPECT_EQ(testLogger.fLogEntries[0].fChannel->GetName(), "TheChannel");
   EXPECT_NE(testLogger.fLogEntries[0].fLocation.fFile.find("testLogger.cxx"), std::string::npos);
   EXPECT_NE(testLogger.fLogEntries[0].fLocation.fFuncName.find("EmittedEntry"), std::string::npos);
   EXPECT_EQ(testLogger.fLogEntries[0].fLocation.fLine, logLine);
   EXPECT_EQ(testLogger.fLogEntries[0].fLevel, ROOT::ELogLevel::kError);
   EXPECT_EQ(testLogger.fLogEntries[0].fMessage, "The text");
   EXPECT_TRUE(testLogger.fLogEntries[0].IsError());
   EXPECT_FALSE(testLogger.fLogEntries[0].IsWarning());
}

TEST(Logger, RLogManagerCounts)
{
   TestLogger testLogger;
   // Check diag counts of RLogManager.
   auto initialWarnings = ROOT::RLogManager::Get().GetNumWarnings();
   auto initialErrors = ROOT::RLogManager::Get().GetNumErrors();
   R__LOG_ERROR() << "emitted";
   EXPECT_EQ(ROOT::RLogManager::Get().GetNumWarnings(), initialWarnings);
   EXPECT_EQ(ROOT::RLogManager::Get().GetNumErrors(), initialErrors + 1);
}

TEST(Logger, RLogDiagCounter)
{
   TestLogger testLogger;
   R__LOG_ERROR() << "emitted"; // before RAII, should not be counted.
   R__LOG_ERROR() << "emitted"; // before RAII, should not be counted.
   // Check counter seeing what was emitted during its lifetime.
   ROOT::RLogScopedDiagCount counter;
   EXPECT_EQ(counter.GetAccumulatedWarnings(), 0);
   EXPECT_EQ(counter.GetAccumulatedErrors(), 0);
   EXPECT_FALSE(counter.HasWarningOccurred());
   EXPECT_FALSE(counter.HasErrorOccurred());
   EXPECT_FALSE(counter.HasErrorOrWarningOccurred());

   R__LOG_ERROR() << "emitted";

   EXPECT_EQ(counter.GetAccumulatedWarnings(), 0);
   EXPECT_EQ(counter.GetAccumulatedErrors(), 1);
   EXPECT_FALSE(counter.HasWarningOccurred());
   EXPECT_TRUE(counter.HasErrorOccurred());
   EXPECT_TRUE(counter.HasErrorOrWarningOccurred());
}

TEST(Logger, RLogScopedVerbositySuppress)
{
   ROOT::RLogChannel channel("ABC");

   auto initialLogLevel = ROOT::RLogManager::Get().GetVerbosity();
   auto initialLogLevelABC = channel.GetVerbosity();
   EXPECT_EQ(initialLogLevel, ROOT::ELogLevel::kWarning);
   EXPECT_EQ(initialLogLevelABC, ROOT::ELogLevel::kUnset);

   {
      // Test simple suppression.
      TestLogger testLogger;
      R__LOG_WARNING(channel) << "emitted";
      EXPECT_EQ(testLogger.size(), 1);
      ROOT::RLogScopedVerbosity suppress(ROOT::ELogLevel::kError);
      R__LOG_WARNING(channel) << "suppressed";
      EXPECT_EQ(testLogger.size(), 1);
   }
   {
      // Test channel specific suppression given global higher verbosity.
      TestLogger testLogger;
      ROOT::RLogScopedVerbosity suppressGlobal(ROOT::ELogLevel::kInfo);
      ROOT::RLogScopedVerbosity suppress(channel, ROOT::ELogLevel::kError);
      R__LOG_WARNING(channel) << "suppressed";
      R__LOG_INFO(channel) << "suppressed, too";
      EXPECT_TRUE(testLogger.empty());
      R__LOG_ERROR(channel) << "emitted";
      EXPECT_EQ(testLogger.size(), 1);

      R__LOG_ERROR() << "emitted";
      R__LOG_WARNING() << "emitted, second";
      R__LOG_INFO() << "emitted, third";
      EXPECT_EQ(testLogger.size(), 4);
   }
   {
      // Check unrelated channel.
      TestLogger testLogger;
      ROOT::RLogChannel channel123("123");
      ROOT::RLogScopedVerbosity suppress(channel123, ROOT::ELogLevel::kFatal);
      R__LOG_ERROR(channel) << "emitted";
      EXPECT_EQ(testLogger.size(), 1);
   }
   {
      // Check global vs specific suppression.
      TestLogger testLogger;
      ROOT::RLogScopedVerbosity suppressGlobal(ROOT::ELogLevel::kDebug);
      ROOT::RLogScopedVerbosity suppress(channel, ROOT::ELogLevel::kError);
      R__LOG_WARNING(channel) << "suppressed";
      EXPECT_TRUE(testLogger.empty());
      R__LOG_WARNING() << "emitted";
      EXPECT_EQ(testLogger.size(), 1);
   }

   // Check that levels have returned to values before RAII.
   EXPECT_EQ(ROOT::RLogManager::Get().GetVerbosity(), initialLogLevel);
   EXPECT_EQ(channel.GetVerbosity(), initialLogLevelABC);
}

TEST(Logger, RLogScopedVerbosityVerbose)
{
   ROOT::RLogChannel channel("ABC");

   auto initialLogLevel = ROOT::RLogManager::Get().GetVerbosity();
   auto initialLogLevelABC = channel.GetVerbosity();
   EXPECT_EQ(initialLogLevel, ROOT::ELogLevel::kWarning);
   EXPECT_EQ(initialLogLevelABC, ROOT::ELogLevel::kUnset);

   {
      // Test same diag level as verbosity, in channel and global, before and after RAII.
      TestLogger testLogger;
      R__LOG_DEBUG(0, channel) << "suppressed";
      EXPECT_TRUE(testLogger.empty());
      R__LOG_DEBUG(0) << "suppressed";
      EXPECT_TRUE(testLogger.empty());
      ROOT::RLogScopedVerbosity verbose(ROOT::ELogLevel::kDebug);
      R__LOG_DEBUG(0, channel) << "emitted";
      EXPECT_EQ(testLogger.size(), 1);
      R__LOG_DEBUG(0) << "emitted";
      EXPECT_EQ(testLogger.size(), 2);
   }
   {
      // Test different diag levels, in channel and global, before and after RAII.
      TestLogger testLogger;
      R__LOG_DEBUG(0, channel) << "suppressed";
      EXPECT_TRUE(testLogger.empty());
      R__LOG_WARNING(channel) << "emitted";
      EXPECT_EQ(testLogger.size(), 1);
      R__LOG_DEBUG(0) << "suppressed";
      EXPECT_EQ(testLogger.size(), 1);
      R__LOG_WARNING() << "emitted";
      EXPECT_EQ(testLogger.size(), 2);
      ROOT::RLogScopedVerbosity verbose(channel, ROOT::ELogLevel::kDebug);
      R__LOG_DEBUG(0, channel) << "emitted";
      EXPECT_EQ(testLogger.size(), 3);
      R__LOG_WARNING(channel) << "emitted, too";
      EXPECT_EQ(testLogger.size(), 4);
      R__LOG_DEBUG(0) << "suppressed";
      EXPECT_EQ(testLogger.size(), 4);
      R__LOG_WARNING() << "emitted";
      EXPECT_EQ(testLogger.size(), 5);
   }
   {
      // Test Info level verbosity.
      TestLogger testLogger;
      R__LOG_INFO(channel) << "suppressed";
      EXPECT_TRUE(testLogger.empty());
      R__LOG_DEBUG(0, channel) << "suppressed, second";
      EXPECT_TRUE(testLogger.empty());
      ROOT::RLogScopedVerbosity verbose(channel, ROOT::ELogLevel::kInfo);
      R__LOG_INFO(channel) << "emitted";
      EXPECT_EQ(testLogger.size(), 1);
      R__LOG_DEBUG(0, channel) << "suppressed, third";
      EXPECT_EQ(testLogger.size(), 1);
   }
   {
      // Test verbosity change on other channel not influcing this one.
      TestLogger testLogger;
      ROOT::RLogChannel otherChannel("123");
      ROOT::RLogScopedVerbosity verbose(otherChannel, ROOT::ELogLevel::kDebug);
      R__LOG_DEBUG(0, channel) << "suppressed";
      EXPECT_TRUE(testLogger.empty());
      R__LOG_DEBUG(0, otherChannel) << "emitted";
      EXPECT_EQ(testLogger.size(), 1);
   }

   // Check that levels have returned to values before RAII.
   EXPECT_EQ(ROOT::RLogManager::Get().GetVerbosity(), initialLogLevel);
   EXPECT_EQ(channel.GetVerbosity(), initialLogLevelABC);
}

TEST(Logger, ExtraVerbosityLevels)
{
   TestLogger testLogger;
   ROOT::RLogChannel channel("channel");
   ROOT::RLogScopedVerbosity verbose(channel, ROOT::ELogLevel::kDebug + 50);

   R__LOG_DEBUG(0, channel) << "emitted";
   EXPECT_EQ(testLogger.size(), 1);
   R__LOG_DEBUG(0, channel) << "emitted";
   EXPECT_EQ(testLogger.size(), 2);
   R__LOG_DEBUG(0, channel) << "emitted";
   EXPECT_EQ(testLogger.size(), 3);
   R__LOG_DEBUG(50, channel) << "emitted";
   EXPECT_EQ(testLogger.size(), 4);
   R__LOG_DEBUG(51, channel) << "suppressed";
   EXPECT_EQ(testLogger.size(), 4);
   R__LOG_DEBUG(50) << "suppressed";
   EXPECT_EQ(testLogger.size(), 4);
}

TEST(Logger, SuppressStreamEval)
{
   TestLogger testLogger;
   ROOT::RLogChannel channel("channel");
   bool wasEvaluated = false;
   R__LOG_DEBUG(0, channel) << "It's debug, this should not be called!" << [&]() -> int {
      wasEvaluated = true;
      return 0;
   }();
   EXPECT_FALSE(wasEvaluated);
}

namespace {
struct TestErrorHandler_t {
   int fLevel;
   Bool_t fAbort;
   std::string fLocation;
   std::string fMsg;
};

TestErrorHandler_t testErrorHandlerVal;
void testErrorHandler(int level, Bool_t abort, const char *location, const char *msg)
{
   testErrorHandlerVal.fLevel = level;
   testErrorHandlerVal.fAbort = abort;
   testErrorHandlerVal.fLocation = location;
   testErrorHandlerVal.fMsg = msg;
}
} // namespace

TEST(Logger, ROOTErrorHandlerDiagString)
{
   auto prevErrorHandler = GetErrorHandler();
   SetErrorHandler(testErrorHandler);
   ROOT::RLogChannel channel("ROOT.checkChannelName");

   // clang-format off
   R__LOG_ERROR(channel) << "check message " << 42; auto lineNumber = __LINE__;
   // clang-format on

   EXPECT_EQ(testErrorHandlerVal.fLevel, kError);
   EXPECT_EQ(testErrorHandlerVal.fAbort, false);
   EXPECT_NE(testErrorHandlerVal.fLocation.find("[ROOT.checkChannelName]"), std::string::npos)
      << "testErrorHandlerVal.fLocation is " << testErrorHandlerVal.fLocation;
   EXPECT_NE(testErrorHandlerVal.fLocation.find(" Error "), std::string::npos)
      << "testErrorHandlerVal.fLocation is " << testErrorHandlerVal.fLocation;
   EXPECT_NE(testErrorHandlerVal.fLocation.find("testLogger.cxx:" + std::to_string(lineNumber) + " "),
             std::string::npos)
      << "testErrorHandlerVal.fLocation is " << testErrorHandlerVal.fLocation;
   EXPECT_NE(testErrorHandlerVal.fLocation.find("ROOTErrorHandlerDiagString"), std::string::npos)
      << "testErrorHandlerVal.fLocation is " << testErrorHandlerVal.fLocation;

   EXPECT_EQ(testErrorHandlerVal.fMsg, "check message 42");

   SetErrorHandler(prevErrorHandler);
}

TEST(Logger, ParseLogLevel)
{
   EXPECT_EQ(ROOT::RLogManager::ParseLogLevel("Debug(3)"), ROOT::ELogLevel::kDebug + 3);
   EXPECT_EQ(ROOT::RLogManager::ParseLogLevel("Debug+5"), ROOT::ELogLevel::kDebug + 5);
   EXPECT_EQ(ROOT::RLogManager::ParseLogLevel("Debug:2"), ROOT::ELogLevel::kDebug + 2);
   EXPECT_EQ(ROOT::RLogManager::ParseLogLevel("Debug"), ROOT::ELogLevel::kDebug);
   EXPECT_EQ(ROOT::RLogManager::ParseLogLevel("kDebug"), ROOT::ELogLevel::kDebug);
   EXPECT_EQ(ROOT::RLogManager::ParseLogLevel("Error"), ROOT::ELogLevel::kError);
   EXPECT_EQ(ROOT::RLogManager::ParseLogLevel("kError"), ROOT::ELogLevel::kError);
   EXPECT_EQ(ROOT::RLogManager::ParseLogLevel("Warning"), ROOT::ELogLevel::kWarning);
   EXPECT_EQ(ROOT::RLogManager::ParseLogLevel("warn"), ROOT::ELogLevel::kWarning);
   EXPECT_EQ(ROOT::RLogManager::ParseLogLevel("Info"), ROOT::ELogLevel::kInfo);
   EXPECT_EQ(ROOT::RLogManager::ParseLogLevel("FATAL"), ROOT::ELogLevel::kFatal);
   EXPECT_EQ(ROOT::RLogManager::ParseLogLevel("Unset"), ROOT::ELogLevel::kUnset);
   EXPECT_EQ(ROOT::RLogManager::ParseLogLevel("  kInfo  "), ROOT::ELogLevel::kInfo);
   EXPECT_EQ(ROOT::RLogManager::ParseLogLevel(""), ROOT::ELogLevel::kUnset);
   EXPECT_EQ(ROOT::RLogManager::ParseLogLevel("invalid_level"), ROOT::ELogLevel::kUnset);
}

TEST(Logger, ConfigureFromString)
{
   auto &mgr = ROOT::RLogManager::Get();
   auto initialGlobal = mgr.GetVerbosity();

   mgr.ConfigureFromString("ROOT.InterpreterPerf=Debug(3), ROOT.RBrowser=Error, default=Info");

   ROOT::RLogChannel perf("ROOT.InterpreterPerf");
   ROOT::RLogChannel browser("ROOT.RBrowser");
   ROOT::RLogChannel other("ROOT.UnspecifiedChannel");

   EXPECT_EQ(perf.GetEffectiveVerbosity(mgr), ROOT::ELogLevel::kDebug + 3);
   EXPECT_EQ(browser.GetEffectiveVerbosity(mgr), ROOT::ELogLevel::kError);
   EXPECT_EQ(other.GetEffectiveVerbosity(mgr), ROOT::ELogLevel::kInfo);

   // Clean up
   mgr.ResetAllChannelVerbosities();
   mgr.SetVerbosity(initialGlobal);
}

TEST(Logger, ConfigureFromEnv)
{
   auto &mgr = ROOT::RLogManager::Get();
   auto initialGlobal = mgr.GetVerbosity();

   const char *prevEnv = std::getenv("ROOT_LOG");
   std::string savedEnv = prevEnv ? prevEnv : "";

#if defined(_WIN32)
   _putenv("ROOT_LOG=ROOT.InterpreterPerf=Debug(3),ROOT.RBrowser=Error");
#else
   setenv("ROOT_LOG", "ROOT.InterpreterPerf=Debug(3),ROOT.RBrowser=Error", 1);
#endif

   mgr.ConfigureFromEnv();

   ROOT::RLogChannel perf("ROOT.InterpreterPerf");
   ROOT::RLogChannel browser("ROOT.RBrowser");

   EXPECT_EQ(perf.GetEffectiveVerbosity(mgr), ROOT::ELogLevel::kDebug + 3);
   EXPECT_EQ(browser.GetEffectiveVerbosity(mgr), ROOT::ELogLevel::kError);

   {
      TestLogger testLogger;

      // Debug(2) should be emitted for perf (effective level is Debug(3))
      R__LOG_DEBUG(2, perf) << "perf debug 2 emitted";
      EXPECT_EQ(testLogger.size(), 1);

      // Debug(4) should be suppressed for perf
      R__LOG_DEBUG(4, perf) << "perf debug 4 suppressed";
      EXPECT_EQ(testLogger.size(), 1);

      // Warning should be suppressed for browser (effective level is Error)
      R__LOG_WARNING(browser) << "browser warning suppressed";
      EXPECT_EQ(testLogger.size(), 1);

      // Error should be emitted for browser
      R__LOG_ERROR(browser) << "browser error emitted";
      EXPECT_EQ(testLogger.size(), 2);
   }

   // Restore environment and manager state
#if defined(_WIN32)
   if (prevEnv)
      _putenv(("ROOT_LOG=" + savedEnv).c_str());
   else
      _putenv("ROOT_LOG=");
#else
   if (prevEnv)
      setenv("ROOT_LOG", savedEnv.c_str(), 1);
   else
      unsetenv("ROOT_LOG");
#endif

   mgr.ResetAllChannelVerbosities();
   mgr.SetVerbosity(initialGlobal);
}
