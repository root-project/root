#include "JSONIOUtils.h"

#include <RooAbsLValue.h>
#include <RooArgList.h>
#include <RooCategory.h>
#include <RooConstVar.h>
#include <RooDataHist.h>
#include <RooSuperCategory.h>
#include <RooWorkspace.h>
#include <RooStats/HistFactory/ParamHistFunc.h>
#include <RooFitHS3/RooJSONFactoryWSTool.h>

#include <algorithm>
#include <cstring>
#include <set>
#include <sstream>
#include <string>
#include <utility>

using RooFit::Detail::JSONNode;

// Parse the entire string as a number (integer or floating-point). The output is only valid if that succeeds.
ToNumberOutput toNumber(const std::string &str)
{
   // std::from_chars for floating-point types is not portably available on all
   // platforms ROOT supports, so we rely on the stream extraction.
   std::istringstream stream(str);
   ToNumberOutput output;
   output.valid = (stream >> output.value) && stream.eof();
   return output;
}

// True if `arg` is a RooConstVar whose name is a plain number (i.e. a literal constant).
bool isLiteralConstVar(RooAbsArg const &arg)
{
   bool isRooConstVar = dynamic_cast<RooConstVar const *>(&arg);
   return isRooConstVar && toNumber(arg.GetName()).valid;
}

// Write a reference to `arg`: the value if it is a literal constant, and the name otherwise.
void writeReference(JSONNode &node, RooAbsArg const &arg)
{
   if (isLiteralConstVar(arg)) {
      node << static_cast<RooConstVar const &>(arg).getVal();
   } else {
      node << arg.GetName();
   }
}

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

std::unique_ptr<RooArgSet>
histogramVariablesNamedAfterObservables(RooDataHist const &dh, RooArgSet const &histObs, RooArgSet const &funcObs)
{
   auto out = std::make_unique<RooArgSet>();
   for (RooAbsArg *var : *dh.get()) {
      std::string name = var->GetName();
      for (std::size_t i = 0; i < histObs.size() && i < funcObs.size(); ++i) {
         if (name == histObs[i]->GetName() && dynamic_cast<RooAbsLValue const *>(funcObs[i])) {
            name = funcObs[i]->GetName();
            break;
         }
      }
      out->addOwned(std::unique_ptr<RooAbsArg>{static_cast<RooAbsArg *>(var->clone(name.c_str()))});
   }
   return out;
}

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

void checkParamHistFuncOrderIsDefined(RooFit::HS3::Version const &version, RooAbsCollection const &vars,
                                      std::string const &context)
{
   if (vars.size() >= 2 && version < RooFit::HS3::Version{0, 3}) {
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
