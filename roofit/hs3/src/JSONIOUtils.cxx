#include "JSONIOUtils.h"

#include <RooAbsBinning.h>
#include <RooAbsLValue.h>
#include <RooArgList.h>
#include <RooCategory.h>
#include <RooSuperCategory.h>
#include <RooWorkspace.h>
#include <RooStats/HistFactory/ParamHistFunc.h>
#include <RooFitHS3/RooJSONFactoryWSTool.h>

#include <algorithm>
#include <cctype>
#include <cstring>
#include <set>
#include <string>
#include <utility>

using RooFit::Detail::JSONNode;

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

// Writes a regular binning with "min", "max" and "nbins", and an irregular one with its "edges" only, since HS3 0.3
// forbids "min" and "max" alongside them.
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

namespace {

void sortByName(std::vector<RooCategory const *> &columns)
{
   std::sort(columns.begin(), columns.end(),
             [](RooCategory const *l, RooCategory const *r) { return std::strcmp(l->GetName(), r->GetName()) < 0; });
}

} // namespace

std::vector<RooCategory const *> selectionColumns(RooAbsCategory const &indexCat)
{
   std::vector<RooCategory const *> columns;
   auto addColumn = [&](RooAbsArg const &arg) {
      auto *cat = dynamic_cast<RooCategory const *>(&arg);
      if (!cat) {
         RooJSONFactoryWSTool::error("the index category '" + std::string(arg.GetName()) + "' is a " + arg.ClassName() +
                                     ", but only a RooCategory or a RooSuperCategory of RooCategories " +
                                     "can be written as the columns of a combined distribution");
      }
      columns.push_back(cat);
   };
   if (auto *super = dynamic_cast<RooSuperCategory const *>(&indexCat)) {
      for (RooAbsArg const *input : super->inputCatList()) {
         addColumn(*input);
      }
   } else {
      addColumn(indexCat);
   }
   sortByName(columns);
   return columns;
}

std::vector<RooCategory const *> selectionColumns(RooAbsCollection const &vars)
{
   std::vector<RooCategory const *> columns;
   for (RooAbsArg const *arg : vars) {
      if (auto *cat = dynamic_cast<RooCategory const *>(arg)) {
         columns.push_back(cat);
      }
   }
   sortByName(columns);
   return columns;
}

std::vector<std::vector<std::string>> selectionKeys(std::vector<RooCategory const *> const &columns)
{
   std::vector<std::vector<std::string>> keys{{}};
   for (RooCategory const *column : columns) {
      std::vector<std::vector<std::string>> next;
      for (auto const &key : keys) {
         for (unsigned int i = 0; i < column->size(); ++i) {
            next.push_back(key);
            next.back().push_back(column->getOrdinal(i).first);
         }
      }
      keys = std::move(next);
   }
   return keys;
}

std::string selectionLabel(std::vector<std::string> const &key)
{
   if (key.size() == 1) {
      return key.front();
   }
   std::string label = "{";
   for (std::size_t i = 0; i < key.size(); ++i) {
      label += (i == 0 ? "" : ";") + key[i];
   }
   return label + "}";
}

std::string selectionKeyToString(std::vector<std::string> const &key)
{
   std::string out = "[";
   for (std::size_t i = 0; i < key.size(); ++i) {
      out += (i == 0 ? "\"" : ", \"") + key[i] + "\"";
   }
   return out + "]";
}

std::string selectionLabel(RooAbsCategory const &indexCat, std::vector<RooCategory const *> const &columns,
                           std::vector<std::string> const &key)
{
   auto *super = dynamic_cast<RooSuperCategory const *>(&indexCat);
   if (!super) {
      return key.front();
   }
   std::vector<std::string> inInputOrder;
   for (RooAbsArg const *input : super->inputCatList()) {
      auto found = std::find_if(columns.begin(), columns.end(),
                                [&](RooCategory const *c) { return std::strcmp(c->GetName(), input->GetName()) == 0; });
      inInputOrder.push_back(key[found - columns.begin()]);
   }
   return selectionLabel(inInputOrder);
}

std::vector<RooCategory *>
readSelectionColumns(RooWorkspace const &ws, JSONNode const &node, const char *field, std::string const &context)
{
   auto const *list = node.find(field);
   if (!list || !list->is_seq() || list->num_children() == 0) {
      RooJSONFactoryWSTool::error("\"" + std::string(field) + "\" of " + context +
                                  " must be a non-empty list of names of categorical axes");
   }
   std::vector<RooCategory *> columns;
   for (JSONNode const &child : list->children()) {
      if (child.is_container() || child.val().empty()) {
         RooJSONFactoryWSTool::error("\"" + std::string(field) + "\" of " + context +
                                     " must be a non-empty list of names of categorical axes");
      }
      RooCategory *cat = ws.cat(child.val());
      if (!cat) {
         RooJSONFactoryWSTool::error("the column '" + child.val() + "' of " + context +
                                     " is not declared by a categorical axis of a domain");
      }
      if (std::find(columns.begin(), columns.end(), cat) != columns.end()) {
         RooJSONFactoryWSTool::error("the column '" + child.val() + "' of " + context + " is listed twice");
      }
      columns.push_back(cat);
   }
   return columns;
}

std::vector<std::string>
readSelectionKey(JSONNode const &component, std::vector<RooCategory *> const &columns, std::string const &context)
{
   auto const *keyNode = component.find("key");
   if (!keyNode || !keyNode->is_seq() || keyNode->num_children() != columns.size()) {
      RooJSONFactoryWSTool::error("every component of " + context + " must have a \"key\" with " +
                                  std::to_string(columns.size()) + " labels, one per column");
   }
   std::vector<std::string> key;
   std::size_t i = 0;
   for (JSONNode const &label : keyNode->children()) {
      if (label.is_container() || !columns[i]->hasLabel(label.val())) {
         RooJSONFactoryWSTool::error("the label '" + (label.is_container() ? std::string("") : label.val()) +
                                     "' in position " + std::to_string(i) + " of a key of " + context +
                                     " is not one of the categories of '" + columns[i]->GetName() + "'");
      }
      key.push_back(label.val());
      ++i;
   }
   return key;
}
