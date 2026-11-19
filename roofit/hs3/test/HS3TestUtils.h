// Helpers for the tests of RooFitHS3
// Authors: Carsten D. Burgard, DESY/ATLAS, 12/2021
//          Jonas Rembser, CERN 11/2026

#ifndef RooFitHS3_test_HS3TestUtils_h
#define RooFitHS3_test_HS3TestUtils_h

#include <RooFitHS3/RooJSONFactoryWSTool.h>
#include <RooFit/Detail/JSONInterface.h>

#include <RooAbsData.h>
#include <RooAbsPdf.h>
#include <RooArgSet.h>
#include <RooCategory.h>
#include <RooFit/ModelConfig.h>
#include <RooHelpers.h>
#include <RooMsgService.h>
#include <RooRealVar.h>
#include <RooWorkspace.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <map>
#include <memory>
#include <ostream>
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

// ---------------------------------------------------------------------------------------------------------------
// Looking into documents

/// An HS3 document with access to its nodes by path, like "distributions/model/samples/0/modifiers/lumi/data": the
/// steps are the keys of maps and, in lists, the position of an element or its "name".
class Doc {
public:
   Doc(std::string json) : _json{std::move(json)} {}
   Doc(const char *json) : _json{json} {}
   /// The document that the workspace is exported to, with the warnings of the export.
   explicit Doc(RooWorkspace &ws, std::vector<ConfigFlag> const &flags = {})
      : _warnings{recordWarningsOf([&] { _json = toJSON(ws, flags); })}
   {
   }

   std::string const &json() const { return _json; }
   operator std::string const &() const { return _json; }
   std::string const &warnings() const { return _warnings; }

   RooFit::Detail::JSONNode const &root() const
   {
      if (!_tree) {
         _tree = RooFit::Detail::JSONTree::create(_json);
      }
      return _tree->rootnode();
   }

   /// The node at the path, or null if there is none.
   RooFit::Detail::JSONNode const *find(std::string const &path) const
   {
      RooFit::Detail::JSONNode const *node = &root();
      std::stringstream steps{path};
      for (std::string step; node && std::getline(steps, step, '/');) {
         node = child(*node, step);
      }
      return node;
   }

   /// The node at the path. Throws if there is none, which fails the test.
   RooFit::Detail::JSONNode const &operator[](std::string const &path) const
   {
      if (auto const *node = find(path)) {
         return *node;
      }
      throw std::runtime_error("no node '" + path + "' in " + _json);
   }

   /// Whether the node at the path exists. The node that should contain it must exist.
   bool has(std::string const &path) const
   {
      const auto pos = path.rfind('/');
      auto const &parent = pos == std::string::npos ? root() : (*this)[path.substr(0, pos)];
      return child(parent, path.substr(pos == std::string::npos ? 0 : pos + 1)) != nullptr;
   }

   /// The values of the list at the path, with "null" for null.
   std::vector<std::string> strings(std::string const &path) const
   {
      std::vector<std::string> out;
      for (auto const &item : (*this)[path].children()) {
         out.push_back(item.is_null() ? "null" : item.val());
      }
      return out;
   }

   /// The names of the objects in the list at the path.
   std::vector<std::string> names(std::string const &path) const
   {
      std::vector<std::string> out;
      for (auto const &item : (*this)[path].children()) {
         out.push_back(item["name"].val());
      }
      return out;
   }

   /// The node at the path as JSON.
   std::string dump(std::string const &path) const
   {
      std::stringstream ss;
      (*this)[path].writeJSON(ss);
      return ss.str();
   }

private:
   static RooFit::Detail::JSONNode const *child(RooFit::Detail::JSONNode const &node, std::string const &step)
   {
      if (node.is_map()) {
         return node.find(step);
      }
      if (!node.is_seq()) {
         return nullptr;
      }
      if (!step.empty() && step.find_first_not_of("0123456789") == std::string::npos) {
         const std::size_t pos = std::stoul(step);
         return pos < node.num_children() ? &node.child(pos) : nullptr;
      }
      return RooJSONFactoryWSTool::findNamedChild(node, step);
   }

   std::string _json;
   std::string _warnings;
   mutable std::unique_ptr<RooFit::Detail::JSONTree> _tree;
};

namespace Detail {

inline std::string dump(RooFit::Detail::JSONNode const &node)
{
   std::stringstream ss;
   node.writeJSON(ss);
   return ss.str();
}

// Describes in `why` the first difference between a node and the expectation, see matches().
inline bool matchNode(RooFit::Detail::JSONNode const &actual, RooFit::Detail::JSONNode const &expected,
                      std::string const &path, std::string &why)
{
   auto mismatch = [&](std::string const &what) {
      why = "at '" + path + "': expected " + what + ", found " + dump(actual);
      return false;
   };
   if (expected.is_map()) {
      if (!actual.is_map()) {
         return mismatch("a map");
      }
      for (auto const &item : expected.children()) {
         const std::string key = item.key();
         const bool absent = !item.is_container() && !item.is_null() && item.val() == "<absent>";
         if (absent ? actual.has_child(key) : !actual.has_child(key)) {
            return mismatch(absent ? "no \"" + key + "\"" : "\"" + key + "\"");
         }
         if (!absent && !matchNode(actual[key], item, path + "/" + key, why)) {
            return false;
         }
      }
      return true;
   }
   if (expected.is_seq()) {
      if (!actual.is_seq() || actual.num_children() != expected.num_children()) {
         return mismatch("a list of " + std::to_string(expected.num_children()) + " elements");
      }
      for (std::size_t i = 0; i < expected.num_children(); ++i) {
         if (!matchNode(actual.child(i), expected.child(i), path + "/" + std::to_string(i), why)) {
            return false;
         }
      }
      return true;
   }
   bool equal = false;
   if (expected.is_null() || actual.is_null() || actual.is_container()) {
      equal = expected.is_null() && actual.is_null();
   } else if (expected.is_number() || actual.is_number()) {
      equal = expected.is_number() && actual.is_number() &&
              testing::Matches(testing::DoubleEq(expected.val_double()))(actual.val_double());
   } else {
      equal = expected.val() == actual.val();
   }
   return equal || mismatch(dump(expected));
}

} // namespace Detail

/// Whether the node at the path matches the JSON value `expected`. Maps match if every key of `expected` matches:
/// other keys are ignored, and the value "<absent>" stands for a key that must not exist. Lists match element by
/// element, and other values if they are equal.
inline testing::AssertionResult matches(Doc const &doc, std::string const &path, std::string const &expected)
{
   auto const *node = doc.find(path);
   if (!node) {
      return testing::AssertionFailure() << "no node '" << path << "' in " << doc.json();
   }
   auto expectedTree = RooFit::Detail::JSONTree::create(expected);
   std::string why;
   if (Detail::matchNode(*node, expectedTree->rootnode(), path, why)) {
      return testing::AssertionSuccess();
   }
   return testing::AssertionFailure() << why << "\nin '" << path << "': " << Detail::dump(*node);
}

// ---------------------------------------------------------------------------------------------------------------
// Writing documents

/// The non-empty parts, separated by commas.
inline std::string join(std::vector<std::string> const &parts)
{
   std::string out;
   for (auto const &part : parts) {
      if (!part.empty()) {
         out += (out.empty() ? "" : ", ") + part;
      }
   }
   return out;
}

/// An HS3 document of the given version with the given top-level sections.
inline std::string document(std::string const &version, std::vector<std::string> const &sections = {})
{
   return join({R"({"metadata": {"hs3_version": ")" + version + R"("})", join(sections)}) + "}";
}

/// A top-level section that is a list, like "distributions".
inline std::string section(std::string const &name, std::vector<std::string> const &items)
{
   return "\"" + name + "\": [" + join(items) + "]";
}

inline std::string distributions(std::vector<std::string> const &items)
{
   return section("distributions", items);
}

inline std::string functions(std::vector<std::string> const &items)
{
   return section("functions", items);
}

/// The "domains" with the default domain of RooFit.
inline std::string defaultDomain(std::vector<std::string> const &axes)
{
   return R"("domains": [{"name": "default_domain", "type": "product_domain", "axes": [)" + join(axes) + "]}]";
}

/// The "parameter_points" with the default values of RooFit.
inline std::string defaultValues(std::vector<std::string> const &parameters)
{
   return R"("parameter_points": [{"name": "default_values", "parameters": [)" + join(parameters) + "]}]";
}

namespace Detail {

inline std::string number(double value)
{
   std::stringstream ss;
   ss.precision(15);
   ss << value;
   const std::string out = ss.str();
   return out.find_first_of(".en") == std::string::npos ? out + ".0" : out;
}

} // namespace Detail

/// Axes with a common range for a domain.
inline std::string axes(std::vector<std::string> const &names, double min, double max)
{
   std::vector<std::string> out;
   for (auto const &name : names) {
      out.push_back(R"({"name": ")" + name + R"(", "min": )" + Detail::number(min) + R"(, "max": )" +
                    Detail::number(max) + "}");
   }
   return join(out);
}

/// The values of parameters for a parameter point.
inline std::string values(std::vector<std::pair<std::string, double>> const &parameters)
{
   std::vector<std::string> out;
   for (auto const &parameter : parameters) {
      out.push_back(R"({"name": ")" + parameter.first + R"(", "value": )" + Detail::number(parameter.second) + "}");
   }
   return join(out);
}

/// A sample of a histfactory_dist, with the JSON of its data and the list of its modifiers.
inline std::string sample(std::string const &name, std::string const &data, std::string const &modifiers)
{
   return R"({"name": ")" + name + R"(", "data": )" + data + R"(, "modifiers": [)" + modifiers + "]}";
}

/// A histfactory_dist with the given samples and further keys, over the axis "x" (2 bins in [0, 2]) unless given.
inline std::string channel(std::string const &name, std::vector<std::string> const &samples,
                           std::string const &keys = "",
                           std::string const &axes = R"({"name": "x", "min": 0.0, "max": 2.0, "nbins": 2})")
{
   return join({R"({"name": ")" + name + R"(", "type": "histfactory_dist", "axes": [)" + axes + "]", keys,
                R"("samples": [)" + join(samples) + "]}"});
}

/// A document with the histfactory_dist "model_channel0" on the axis "x" (2 bins in [0, 2]), which has one sample
/// "sig" with the contents [10, 20] and the given modifiers.
class HistFactoryDoc {
public:
   HistFactoryDoc(std::string version, std::string modifiers)
      : _version{std::move(version)}, _modifiers{std::move(modifiers)}
   {
   }
   /// The "default_interpolation" of the channel.
   HistFactoryDoc &interpolation(std::string const &value)
   {
      return channelKeys("\"default_interpolation\": " + value);
   }
   /// The "data" of the sample.
   HistFactoryDoc &sampleData(std::string value)
   {
      _data = std::move(value);
      return *this;
   }
   /// Further keys of the channel.
   HistFactoryDoc &channelKeys(std::string const &value)
   {
      _keys.push_back(value);
      return *this;
   }
   /// Further distributions.
   HistFactoryDoc &distributions(std::string const &value)
   {
      _distributions.push_back(value);
      return *this;
   }
   /// Further top-level sections, like defaultDomain().
   HistFactoryDoc &sections(std::string const &value)
   {
      _sections.push_back(value);
      return *this;
   }
   std::string str() const
   {
      std::vector<std::string> dists{channel("model_channel0", {sample("sig", _data, _modifiers)}, join(_keys))};
      dists.insert(dists.end(), _distributions.begin(), _distributions.end());
      std::vector<std::string> all{HS3Test::distributions(dists)};
      all.insert(all.end(), _sections.begin(), _sections.end());
      return document(_version, all);
   }
   operator std::string() const { return str(); }

private:
   std::string _version;
   std::string _modifiers;
   std::string _data = R"({"contents": [10.0, 20.0]})";
   std::vector<std::string> _keys;
   std::vector<std::string> _distributions;
   std::vector<std::string> _sections;
};

// ---------------------------------------------------------------------------------------------------------------
// Round trips

/// Flags that a round trip is done with, see constraintStyles().
struct RoundTripStyle {
   std::string name;
   std::vector<ConfigFlag> flags;
};

/// A model whose round trip through HS3 is tested, see expectRoundTrip() and the suite RoundTrip of
/// testRooFitHS3.cxx.
struct RoundTripModel {
   using Builder = std::function<void(RooWorkspace &)>;

   /// A model that the function builds in an empty workspace.
   RoundTripModel(std::string name_, Builder build_) : name{std::move(name_)}, build{std::move(build_)} {}
   /// A model made with the workspace factory.
   RoundTripModel(std::string name_, std::vector<std::string> const &factoryExpressions)
      : name{std::move(name_)}, build{[=](RooWorkspace &ws) {
           for (auto const &expression : factoryExpressions) {
              if (!ws.factory(expression)) {
                 throw std::runtime_error("factory expression failed: " + expression);
              }
           }
        }}
   {
   }
   /// A model that is imported from an HS3 document.
   static RoundTripModel fromDocument(std::string name, std::string const &json)
   {
      return {std::move(name), [=](RooWorkspace &ws) { importInto(ws, json); }};
   }

   /// The flags that the round trip is done with. By default, only without flags.
   RoundTripModel with(std::vector<RoundTripStyle> value) const
   {
      RoundTripModel out{*this};
      out.styles = std::move(value);
      return out;
   }
   /// The observables: the values of the functions are compared in their bins, see sameValues().
   RoundTripModel over(std::vector<std::string> value) const
   {
      RoundTripModel out{*this};
      out.observables = std::move(value);
      return out;
   }
   /// The ModelConfig and the data set whose likelihood is compared, see likelihoodScan().
   RoundTripModel likelihoodOf(std::string modelConfig_, std::string data_) const
   {
      RoundTripModel out{*this};
      out.modelConfig = std::move(modelConfig_);
      out.data = std::move(data_);
      return out;
   }
   /// A function that gives the likelihood of a workspace at some points, instead of likelihoodScan().
   RoundTripModel likelihoodBy(std::function<std::vector<double>(RooWorkspace &)> value) const
   {
      RoundTripModel out{*this};
      out.likelihood = std::move(value);
      return out;
   }
   /// The values and likelihoods agree up to this relative tolerance, instead of being identical.
   RoundTripModel within(double value) const
   {
      RoundTripModel out{*this};
      out.tolerance = value;
      return out;
   }
   /// The second workspace doesn't have the same components as the first one, e.g. because the import names them
   /// differently. The functions that it has are still compared.
   RoundTripModel withOtherComponents() const
   {
      RoundTripModel out{*this};
      out.sameComponents = false;
      return out;
   }
   /// The first document differs from the following ones, e.g. because the import adds attributes.
   RoundTripModel stableFromSecondExport() const
   {
      RoundTripModel out{*this};
      out.firstExportDiffers = true;
      return out;
   }

   std::string name;
   Builder build;
   std::vector<RoundTripStyle> styles{{"Default", {}}};
   std::vector<std::string> observables{"x"};
   std::string modelConfig;
   std::string data;
   std::function<std::vector<double>(RooWorkspace &)> likelihood;
   double tolerance = 0.;
   bool sameComponents = true;
   bool firstExportDiffers = false;
};

inline void PrintTo(RoundTripModel const &model, std::ostream *os)
{
   *os << model.name;
}

/// The four layouts of the constraint terms that the export flags select: as array distributions or on their own,
/// in "aux_distributions" or paired with data sets.
inline std::vector<RoundTripStyle> constraintStyles()
{
   return {{"ArrayAux", {}},
           {"ArrayData", {exportConstraintData(true)}},
           {"PlainAux", {exportArrayDist(false)}},
           {"PlainData", {exportArrayDist(false), exportConstraintData(true)}}};
}

namespace Detail {

inline bool sameValue(double a, double b, double tolerance)
{
   if (std::isnan(a) || std::isnan(b)) {
      return std::isnan(a) && std::isnan(b);
   }
   return tolerance == 0. ? a == b : std::abs(a - b) <= tolerance * std::max(std::abs(a), std::abs(b));
}

// The floating variables of the workspace that are no observables, with values that are moved from the current
// ones by a fraction, in a direction that depends on the name.
inline std::map<std::string, double>
displacedValues(RooWorkspace &ws, std::vector<std::string> const &observables, double fraction)
{
   std::map<std::string, double> out;
   for (auto *var : static_range_cast<RooRealVar *>(ws.allVars())) {
      const std::string name = var->GetName();
      if (var->isConstant() || std::find(observables.begin(), observables.end(), name) != observables.end()) {
         continue;
      }
      // A fraction of the value, or of the range if the value is zero.
      double scale = var->getVal() != 0. ? std::abs(var->getVal()) : 1.;
      if (var->hasMin() && var->hasMax()) {
         scale = var->getVal() != 0. ? std::min(scale, var->getMax() - var->getMin()) : var->getMax() - var->getMin();
      }
      const int sign = (name.size() + name.back()) % 2 ? 1 : -1;
      double value = var->getVal() + sign * fraction * scale;
      if (!var->inRange(value, nullptr)) {
         value = var->getVal() - sign * fraction * scale;
      }
      if (var->inRange(value, nullptr)) {
         out[name] = value;
      }
   }
   return out;
}

inline void setValues(RooWorkspace &ws, std::map<std::string, double> const &values)
{
   for (auto const &item : values) {
      if (RooRealVar *var = ws.var(item.first)) {
         var->setVal(item.second);
      }
   }
}

} // namespace Detail

/// Whether the functions and pdfs that both workspaces have give the same values: in every bin of each observable,
/// for every state of each category, with the parameters as they are and with the floating parameters moved. The
/// pdfs are normalized over the observables. With `sameComponents`, the workspaces must have the same components.
inline testing::AssertionResult
sameValues(RooWorkspace &ws1, RooWorkspace &ws2, RoundTripModel const &model, bool sameComponents)
{
   if (sameComponents) {
      std::vector<std::string> names1;
      std::vector<std::string> names2;
      for (RooAbsArg *arg : ws1.components()) {
         names1.push_back(arg->GetName());
      }
      for (RooAbsArg *arg : ws2.components()) {
         names2.push_back(arg->GetName());
      }
      std::sort(names1.begin(), names1.end());
      std::sort(names2.begin(), names2.end());
      if (names1 != names2) {
         return testing::AssertionFailure() << "the components differ: " << testing::PrintToString(names1) << " and "
                                            << testing::PrintToString(names2);
      }
   }

   // The values of the observables and the states of the categories are not part of a document: start from the ones
   // of the first workspace.
   RooArgSet nset1;
   RooArgSet nset2;
   std::vector<std::pair<RooRealVar *, RooRealVar *>> observables;
   for (auto const &name : model.observables) {
      if (ws1.var(name) && ws2.var(name)) {
         observables.emplace_back(ws1.var(name), ws2.var(name));
         nset1.add(*ws1.var(name));
         nset2.add(*ws2.var(name));
         ws2.var(name)->setVal(ws1.var(name)->getVal());
      }
   }
   for (auto *category : static_range_cast<RooCategory *>(ws1.allCats())) {
      RooCategory *other = ws2.cat(category->GetName());
      if (other && other->hasLabel(category->getCurrentLabel())) {
         other->setLabel(category->getCurrentLabel());
      }
   }
   std::vector<std::pair<RooAbsReal *, RooAbsReal *>> functions;
   RooArgSet all{ws1.allPdfs()};
   all.add(ws1.allFunctions());
   for (auto *function : static_range_cast<RooAbsReal *>(all)) {
      if (auto *other = dynamic_cast<RooAbsReal *>(ws2.arg(function->GetName()))) {
         functions.emplace_back(function, other);
      }
   }
   std::size_t nCompared = 0;
   std::string where;
   auto compare = [&]() -> testing::AssertionResult {
      for (auto const &pair : functions) {
         const double value1 = pair.first->getVal(nset1);
         const double value2 = pair.second->getVal(nset2);
         ++nCompared;
         if (!Detail::sameValue(value1, value2, model.tolerance)) {
            return testing::AssertionFailure() << "'" << pair.first->GetName() << "' is " << value1 << " and " << value2
                                               << " (difference " << value2 - value1 << ") " << where;
         }
      }
      return testing::AssertionSuccess();
   };
   auto scan = [&]() -> testing::AssertionResult {
      where = "";
      if (auto result = compare(); !result) {
         return result;
      }
      for (auto const &observable : observables) {
         const double old = observable.first->getVal();
         for (int i = 0; i < observable.first->numBins(); ++i) {
            observable.first->setBin(i);
            observable.second->setBin(i);
            where = std::string("in bin ") + std::to_string(i) + " of '" + observable.first->GetName() + "'";
            if (auto result = compare(); !result) {
               return result;
            }
         }
         observable.first->setVal(old);
         observable.second->setVal(old);
      }
      for (auto *category : static_range_cast<RooCategory *>(ws1.allCats())) {
         RooCategory *other = ws2.cat(category->GetName());
         if (!other) {
            continue;
         }
         const std::string old = category->getCurrentLabel();
         for (auto const &state : *category) {
            if (!other->hasLabel(state.first)) {
               return testing::AssertionFailure()
                      << "'" << category->GetName() << "' has no state '" << state.first << "'";
            }
            category->setLabel(state.first.c_str());
            other->setLabel(state.first.c_str());
            where = "for the state '" + state.first + "' of '" + category->GetName() + "'";
            if (auto result = compare(); !result) {
               return result;
            }
         }
         category->setLabel(old.c_str());
         other->setLabel(old.c_str());
      }
      return testing::AssertionSuccess();
   };

   if (auto result = scan(); !result) {
      return result;
   }
   const auto displaced = Detail::displacedValues(ws1, model.observables, 0.05);
   std::map<std::string, double> original;
   for (auto const &item : displaced) {
      original[item.first] = ws1.var(item.first)->getVal();
   }
   Detail::setValues(ws1, displaced);
   Detail::setValues(ws2, displaced);
   auto result = scan();
   Detail::setValues(ws1, original);
   Detail::setValues(ws2, original);
   if (!result) {
      return result << " with moved parameters";
   }
   if (!functions.empty() && nCompared == 0) {
      return testing::AssertionFailure() << "no values compared";
   }
   return testing::AssertionSuccess();
}

/// The NLL of a ModelConfig for a data set, with its external constraints and global observables: at the values
/// of the parameters in the workspace, and with the floating parameters moved in both directions.
inline std::vector<double> likelihoodScan(RooWorkspace &ws, std::string const &modelConfig, std::string const &data)
{
   auto *mc = dynamic_cast<RooFit::ModelConfig *>(ws.obj(modelConfig));
   if (!mc || !mc->GetPdf() || !ws.data(data)) {
      throw std::runtime_error("likelihoodScan(): no ModelConfig '" + modelConfig + "' with a pdf, or no data '" +
                               data + "'");
   }
   RooArgSet external;
   if (mc->GetExternalConstraints()) {
      external.add(*mc->GetExternalConstraints());
   }
   RooArgSet globals;
   if (mc->GetGlobalObservables()) {
      globals.add(*mc->GetGlobalObservables());
   }
   std::unique_ptr<RooAbsReal> nll{mc->GetPdf()->createNLL(*ws.data(data), RooFit::ExternalConstraints(external),
                                                           RooFit::GlobalObservables(globals))};
   std::vector<std::string> observables;
   for (RooAbsArg *arg : *ws.data(data)->get()) {
      observables.push_back(arg->GetName());
   }
   for (RooAbsArg *arg : globals) {
      observables.push_back(arg->GetName());
   }
   std::map<std::string, double> original;
   for (auto const &item : Detail::displacedValues(ws, observables, 0.)) {
      original[item.first] = item.second;
   }
   std::vector<double> out{nll->getVal()};
   for (double fraction : {0.03, -0.06}) {
      Detail::setValues(ws, original);
      Detail::setValues(ws, Detail::displacedValues(ws, observables, fraction));
      out.push_back(nll->getVal());
   }
   Detail::setValues(ws, original);
   return out;
}

/// Whether two lists of values, like the ones of likelihoodScan(), agree: identically, or up to a relative
/// tolerance.
inline testing::AssertionResult
sameNumbers(std::vector<double> const &a, std::vector<double> const &b, double tolerance)
{
   if (a.size() != b.size() || a.empty()) {
      return testing::AssertionFailure() << a.size() << " and " << b.size() << " values";
   }
   for (std::size_t i = 0; i < a.size(); ++i) {
      if (!std::isfinite(a[i]) || !Detail::sameValue(a[i], b[i], tolerance)) {
         return testing::AssertionFailure() << "value " << i << " is " << testing::PrintToString(a[i]) << " and "
                                            << testing::PrintToString(b[i]) << " (difference " << b[i] - a[i] << ")";
      }
   }
   return testing::AssertionSuccess();
}

/// The round trip of a workspace for each style of the model: the workspace that is read from the exported
/// document is exported to the same document again, it has the same components (without flags), and its functions
/// and likelihood have the same values, see sameValues() and likelihoodScan().
inline void expectRoundTrip(RooWorkspace &ws, RoundTripModel const &model)
{
   for (auto const &style : model.styles) {
      SCOPED_TRACE(model.name + ", " + style.name);
      const std::string json = toJSON(ws, style.flags);
      Imported ws2{json};
      const std::string json2 = toJSON(ws2, style.flags);
      if (!model.firstExportDiffers) {
         EXPECT_EQ(json2, json);
      }
      EXPECT_EQ(reexport(json2, style.flags), json2);
      EXPECT_TRUE(sameValues(ws, ws2, model, model.sameComponents && style.flags.empty()));
      if (model.likelihood) {
         // The function may move the variables.
         RooArgSet variables{ws.allVars()};
         variables.add(ws.allCats());
         std::unique_ptr<RooArgSet> values{variables.snapshot()};
         EXPECT_TRUE(sameNumbers(model.likelihood(ws), model.likelihood(ws2), model.tolerance));
         variables.assign(*values);
      } else if (!model.data.empty()) {
         EXPECT_TRUE(sameNumbers(likelihoodScan(ws, model.modelConfig, model.data),
                                 likelihoodScan(ws2, model.modelConfig, model.data), model.tolerance));
      }
   }
}

/// The round trip of a workspace without flags.
inline void expectRoundTrip(RooWorkspace &ws)
{
   expectRoundTrip(ws, RoundTripModel{"workspace", RoundTripModel::Builder{}});
}

/// The round trip of a workspace with only this function or pdf.
inline void expectRoundTrip(RooAbsArg const &arg)
{
   RooWorkspace ws;
   ws.import(arg, RooFit::Silence());
   expectRoundTrip(ws, RoundTripModel{arg.GetName(), RoundTripModel::Builder{}});
}

} // namespace HS3Test

#endif
