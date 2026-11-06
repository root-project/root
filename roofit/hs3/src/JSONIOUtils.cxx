#include "JSONIOUtils.h"

#include <RooAbsBinning.h>
#include <RooAbsLValue.h>
#include <RooArgList.h>
#include <RooStats/HistFactory/ParamHistFunc.h>
#include <RooFitHS3/RooJSONFactoryWSTool.h>

#include <cctype>
#include <string>
#include <utility>

using RooFit::Detail::JSONNode;
using RooFit::Detail::JSONTree;

bool startsWith(std::string_view str, std::string_view prefix)
{
   return str.size() >= prefix.size() && 0 == str.compare(0, prefix.size(), prefix);
}

bool endsWith(std::string_view str, std::string_view suffix)
{
   return str.size() >= suffix.size() && 0 == str.compare(str.size() - suffix.size(), suffix.size(), suffix);
}

std::string removeSuffix(std::string_view str, std::string_view suffix)
{
   std::string out;
   out += str;
   out = out.substr(0, out.length() - suffix.length());
   return out;
}

std::unique_ptr<RooFit::Detail::JSONTree> varJSONString(const JSONNode &treeRoot)
{
   std::string varName = treeRoot.find("name")->val();
   double val = 0;
   double maxVal = 0;
   double minVal = 0;
   bool isConstant = false;
   bool isRange = false;

   if (auto n = treeRoot.find("value")) {
      val = n->val_double();
      isConstant = true;
   }

   auto maxNode = treeRoot.find("max");
   auto minNode = treeRoot.find("min");
   if (maxNode && minNode) {
      maxVal = maxNode->val_double();
      minVal = minNode->val_double();
      isRange = true;
   }
   if (!isConstant) {
      val = (maxVal + minVal) / 2;
   }
   // Check if variable is at least a range or constant else throw error
   if (!isConstant && !isRange) {
      throw std::invalid_argument("Invalid Syntax: Please provide either 'value' or 'min' and 'max' or both");
   }

   std::unique_ptr<RooFit::Detail::JSONTree> jsonDict = RooFit::Detail::JSONTree::create();
   JSONNode &n = jsonDict->rootnode().set_map();
   JSONNode &_domains = n["domains"].set_seq().append_child().set_map();
   JSONNode &_parameterPoints = n["parameter_points"].set_seq().append_child().set_map();

   _domains["name"] << "default_domain";
   _domains["type"] << "product_domain";
   JSONNode &_axes = _domains["axes"].set_seq().append_child().set_map();
   _axes["name"] << varName;

   _parameterPoints["name"] << "default_values";
   JSONNode &_parameters = _parameterPoints["parameters"].set_seq().append_child().set_map();
   _parameters["name"] << varName;
   _parameters["value"] << val;

   if (isRange) {
      _axes["max"] << maxVal;
      _axes["min"] << minVal;
   }

   if (isConstant && !isRange) {
      _parameters["const"] << true;
      JSONNode &_misc = n["misc"].set_map();
      JSONNode &rootInternal = _misc["ROOT_internal"].set_map();
      JSONNode &_var = rootInternal[varName].set_map();
      _var["tags"] << "Constant";
   }

   return jsonDict;
}

void writeAxisBinning(JSONNode &node, const RooAbsBinning &binning)
{
   if (binning.isUniform()) {
      node["min"] << binning.lowBound();
      node["max"] << binning.highBound();
      node["nbins"] << binning.numBins();
      return;
   }

   auto &edges = node["edges"].set_seq();
   edges.append_child() << binning.binLow(0);
   for (int i = 0; i < binning.numBins(); ++i) {
      edges.append_child() << binning.binHigh(i);
   }
}

// Read a bound of an axis. Infinite bounds are written as null.
double readBound(JSONNode const &node, const char *key, double defaultValue)
{
   if (!node.has_child(key)) {
      return defaultValue;
   }
   auto const &bound = node[key];
   return bound.is_null() ? defaultValue : bound.val_double();
}

// Parse the leading "MAJOR.MINOR" of an HS3 version string like "0.2", "0.2.9" or "0.3".
bool parseHS3Version(std::string const &version, int &major, int &minor)
{
   std::size_t pos = 0;
   auto readNumber = [&](int &out) {
      const std::size_t begin = pos;
      while (pos < version.size() && std::isdigit(static_cast<unsigned char>(version[pos]))) {
         ++pos;
      }
      // Reject empty and absurdly long numbers, which std::stoi could not represent.
      if (pos == begin || pos - begin > 9) {
         return false;
      }
      out = std::stoi(version.substr(begin, pos - begin));
      return true;
   };
   if (!readNumber(major) || pos >= version.size() || version[pos] != '.') {
      return false;
   }
   ++pos;
   return readNumber(minor);
}

// Whether an HS3 version predates `major.minor`. Versions that can't be parsed count as old. That includes the empty
// version of objects that are imported without a document.
bool isHS3VersionBefore(std::string const &version, int major, int minor)
{
   int versionMajor = 0;
   int versionMinor = 0;
   if (!parseHS3Version(version, versionMajor, versionMinor)) {
      return true;
   }
   return std::make_pair(versionMajor, versionMinor) < std::make_pair(major, minor);
}

// Read the bin edges of an axis, which must be at least two strictly ascending numbers.
std::vector<double> readEdges(const JSONNode &node, std::string const &context)
{
   if (!node.is_seq()) {
      RooJSONFactoryWSTool::error("\"edges\" of " + context + " must be a list");
   }
   std::vector<double> edges;
   for (auto const &edge : node.children()) {
      edges.push_back(edge.val_double());
      if (edges.size() > 1 && !(edges.back() > edges[edges.size() - 2])) {
         RooJSONFactoryWSTool::error("\"edges\" of " + context + " must be strictly ascending");
      }
   }
   if (edges.size() < 2) {
      RooJSONFactoryWSTool::error("\"edges\" of " + context + " must contain at least two values");
   }
   return edges;
}

namespace {

std::vector<std::size_t> numBinsPerVariable(RooAbsCollection const &vars)
{
   std::vector<std::size_t> nBins;
   for (RooAbsArg const *var : vars) {
      nBins.push_back(static_cast<std::size_t>(dynamic_cast<RooAbsLValue const &>(*var).numBins()));
   }
   return nBins;
}

// The position in TH1 order (first variable fastest) of the bin with position `k` in RooDataHist/HS3 order (last
// variable fastest).
std::size_t th1Position(std::size_t k, std::vector<std::size_t> const &nBins)
{
   std::size_t th1 = 0;
   std::size_t stride = 1;
   std::vector<std::size_t> indices(nBins.size());
   for (std::size_t j = nBins.size(); j-- > 0;) {
      indices[j] = k % nBins[j];
      k /= nBins[j];
   }
   for (std::size_t j = 0; j < nBins.size(); ++j) {
      th1 += indices[j] * stride;
      stride *= nBins[j];
   }
   return th1;
}

} // namespace

std::vector<RooAbsReal *> paramHistFuncParamsInBinOrder(ParamHistFunc const &func)
{
   std::vector<RooAbsReal *> out;
   for (int k = 0; k < func.numBins(); ++k) {
      out.push_back(&func.getParameter(k));
   }
   return out;
}

RooArgList toParamHistFuncOrder(RooArgList const &paramsInBinOrder, RooAbsCollection const &vars)
{
   const std::vector<std::size_t> nBins = numBinsPerVariable(vars);
   std::vector<RooAbsArg *> th1Order(paramsInBinOrder.size(), nullptr);
   for (std::size_t k = 0; k < paramsInBinOrder.size(); ++k) {
      const std::size_t t = th1Position(k, nBins);
      if (t >= th1Order.size()) {
         RooJSONFactoryWSTool::error("the number of parameters does not match the number of bins");
      }
      th1Order[t] = paramsInBinOrder.at(k);
   }
   RooArgList out;
   for (RooAbsArg *arg : th1Order) {
      out.add(*arg);
   }
   return out;
}

void checkParamHistFuncOrderIsDefined(std::string const &hs3Version, RooAbsCollection const &vars,
                                      std::string const &context)
{
   if (vars.size() >= 2 && isHS3VersionBefore(hs3Version, 0, 3)) {
      RooJSONFactoryWSTool::error(context +
                                  " has a multi-dimensional binning. Documents before HS3 0.3 don't define the order "
                                  "of its per-bin parameters (RooFit wrote them in TH1 order, while the bin "
                                  "contents use the order of HS3 0.3, last axis fastest). Please re-export the "
                                  "workspace with a newer ROOT version.");
   }
}
