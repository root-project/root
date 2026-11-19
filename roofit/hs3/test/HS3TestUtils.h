// Helpers for the tests of RooFitHS3
// Authors: Carsten D. Burgard, DESY/ATLAS, 12/2021
//          Jonas Rembser, CERN 11/2026

#ifndef RooFitHS3_test_HS3TestUtils_h
#define RooFitHS3_test_HS3TestUtils_h

#include <RooFitHS3/RooJSONFactoryWSTool.h>
#include <RooFit/Detail/JSONInterface.h>

#include <RooFit/ModelConfig.h>
#include <RooHelpers.h>
#include <RooMsgService.h>
#include <RooWorkspace.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace HS3Test {

// ---------------------------------------------------------------------------------------------------------------
// Messages

/// The tests print warnings and errors only, unless they change the level themselves. Register it with
/// `testing::AddGlobalTestEnvironment(new HS3Test::QuietEnvironment)`.
class QuietEnvironment : public testing::Environment {
public:
   void SetUp() override { _level = std::make_unique<RooHelpers::LocalChangeMsgLevel>(RooFit::WARNING); }
   void TearDown() override { _level.reset(); }

private:
   std::unique_ptr<RooHelpers::LocalChangeMsgLevel> _level;
};

/// Runs `action` and returns everything that was logged at the given message level and topic while it ran.
template <class Action>
std::string captureMessages(RooFit::MsgLevel level, RooFit::MsgTopic topic, Action &&action)
{
   RooHelpers::HijackMessageStream stream{level, topic};
   action();
   return stream.str();
}

/// The warnings that the RooJSONFactoryWSTool gives while `action` runs.
template <class Action>
std::string warningsOf(Action &&action)
{
   return captureMessages(RooFit::WARNING, RooFit::ObjectHandling, action);
}

/// Like warningsOf(), but the warnings are logged again afterwards: they are still printed, and they still reach a
/// captureMessages() or warningsOf() around the call.
template <class Action>
std::string recordWarningsOf(Action &&action)
{
   std::string warnings = warningsOf(action);
   if (!warnings.empty()) {
      RooMsgService::instance().log(nullptr, RooFit::WARNING, RooFit::ObjectHandling, /*forceSkipPrefix=*/true)
         << warnings << std::flush;
   }
   return warnings;
}

/// Counts the number of (non-overlapping) occurrences of `needle` in `haystack`.
inline std::size_t countOccurrences(std::string_view haystack, std::string_view needle)
{
   std::size_t result = 0;
   for (std::size_t pos = 0; (pos = haystack.find(needle, pos)) != std::string_view::npos; pos += needle.size()) {
      ++result;
   }
   return result;
}

// ---------------------------------------------------------------------------------------------------------------
// Export and import

/// A flag of RooJSONFactoryWSTool::config(), with the value that an export or import is done with.
using ConfigFlag = std::pair<bool RooJSONFactoryWSTool::Config::*, bool>;

// The flags by name, e.g. `toJSON(ws, {exportArrayDist(false), exportConstraintData(true)})`.
inline ConfigFlag exportArrayDist(bool value)
{
   return {&RooJSONFactoryWSTool::Config::exportArrayDist, value};
}
inline ConfigFlag exportConstraintData(bool value)
{
   return {&RooJSONFactoryWSTool::Config::exportConstraintData, value};
}
inline ConfigFlag renameCombinedObservables(bool value)
{
   return {&RooJSONFactoryWSTool::Config::renameCombinedObservables, value};
}
inline ConfigFlag allowExportSharedVariates(bool value)
{
   return {&RooJSONFactoryWSTool::Config::allowExportSharedVariates, value};
}
inline ConfigFlag identifyLegacyBinWidthFactors(bool value)
{
   return {&RooJSONFactoryWSTool::Config::identifyLegacyBinWidthFactors, value};
}
inline ConfigFlag importConstraintsAsChannels(bool value)
{
   return {&RooJSONFactoryWSTool::Config::importConstraintsAsChannels, value};
}

/// Sets flags of RooJSONFactoryWSTool::config() for its lifetime.
class ScopedConfig {
public:
   explicit ScopedConfig(std::vector<ConfigFlag> const &flags) : _old{RooJSONFactoryWSTool::config()}
   {
      for (auto const &flag : flags) {
         RooJSONFactoryWSTool::config().*flag.first = flag.second;
      }
   }
   ~ScopedConfig() { RooJSONFactoryWSTool::config() = _old; }
   ScopedConfig(ScopedConfig const &) = delete;
   ScopedConfig &operator=(ScopedConfig const &) = delete;

private:
   RooJSONFactoryWSTool::Config _old;
};

/// The HS3 document that the workspace is exported to.
inline std::string toJSON(RooWorkspace &ws, std::vector<ConfigFlag> const &flags = {})
{
   ScopedConfig config{flags};
   return RooJSONFactoryWSTool{ws}.exportJSONtoString();
}

/// Imports an HS3 document into the workspace, and returns the warnings of the import.
inline std::string importInto(RooWorkspace &ws, std::string const &json, std::vector<ConfigFlag> const &flags = {})
{
   ScopedConfig config{flags};
   return recordWarningsOf([&] { RooJSONFactoryWSTool{ws}.importJSONfromString(json); });
}

/// The workspace that an HS3 document is imported to, with the warnings of the import.
struct Imported : RooWorkspace {
   explicit Imported(std::string const &json, std::vector<ConfigFlag> const &flags = {})
      : warnings{importInto(*this, json, flags)}
   {
   }
   std::string warnings;
};

/// The document that RooFit writes for the workspace that it reads from `json`.
inline std::string reexport(std::string const &json, std::vector<ConfigFlag> const &flags = {})
{
   Imported ws{json};
   return toJSON(ws, flags);
}

/// `text` with the first occurrence of `from`, which must exist, replaced by `to`.
inline std::string replaced(std::string text, std::string const &from, std::string const &to)
{
   const auto pos = text.find(from);
   if (pos == std::string::npos) {
      throw std::runtime_error("replaced(): no '" + from + "' in " + text);
   }
   return text.replace(pos, from.size(), to);
}

/// Asserts that importing `json` throws and logs an error message containing `expectedReason`.
inline void expectImportThrowsWithError(std::string const &json, std::string const &expectedReason)
{
   RooWorkspace ws;
   bool threw = false;
   const std::string errors = captureMessages(RooFit::ERROR, RooFit::IO, [&] {
      try {
         RooJSONFactoryWSTool{ws}.importJSONfromString(json);
      } catch (const std::runtime_error &) {
         threw = true;
      }
   });

   EXPECT_TRUE(threw);
   EXPECT_THAT(errors, testing::HasSubstr(expectedReason));
}

/// Asserts that exporting `ws` throws and logs an error message containing `expectedReason`.
inline void expectExportThrowsWithError(RooWorkspace &ws, std::string const &expectedReason)
{
   bool threw = false;
   std::string exported;
   const std::string errors = captureMessages(RooFit::ERROR, RooFit::IO, [&] {
      try {
         exported = toJSON(ws);
      } catch (const std::runtime_error &) {
         threw = true;
      }
   });

   EXPECT_TRUE(threw);
   EXPECT_TRUE(exported.empty());
   EXPECT_THAT(errors, testing::HasSubstr(expectedReason));
}

/// Imports a ModelConfig for the pdf into the workspace. The roles are comma-separated lists of names.
inline void addModelConfig(RooWorkspace &ws, const char *name, const char *pdf, const char *observables,
                           const char *poi, const char *nuisance = nullptr, const char *globals = nullptr)
{
   RooFit::ModelConfig mc{name, &ws};
   mc.SetPdf(pdf);
   mc.SetObservables(observables);
   mc.SetParametersOfInterest(poi);
   if (nuisance) {
      mc.SetNuisanceParameters(nuisance);
   }
   if (globals) {
      mc.SetGlobalObservables(globals);
   }
   ws.import(mc);
}

} // namespace HS3Test

#endif
