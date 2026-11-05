/*
 * Project: RooFit
 * Authors:
 *   Carsten D. Burgard, DESY/ATLAS, Dec 2021
 *
 * Copyright (c) 2022, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#include <RooFitHS3/JSONIO.h>
#include <RooFitHS3/RooJSONFactoryWSTool.h>

#include <RooConstVar.h>
#include <RooNumber.h>
#include <RooRealVar.h>
#include <RooBinning.h>
#include <RooAbsCategory.h>
#include <RooArgProxy.h>
#include <RooAbsProxy.h>
#include <RooAbsLValue.h>
#include <RooCategory.h>
#include <RooDataSet.h>
#include <RooDataHist.h>
#include <RooCustomizer.h>
#include <RooSimultaneous.h>
#include <RooSuperCategory.h>
#include <RooFit/ModelConfig.h>
#include <RooFitImplHelpers.h>
#include <RooAbsCollection.h>

#include "JSONIOUtils.h"
#include "Domains.h"

#include <TROOT.h>

#include <algorithm>
#include <iterator>
#include <set>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

/** \class RooJSONFactoryWSTool
\ingroup roofit_dev_docs_hs3

When using \ref Roofitmain, statistical models can be conveniently handled and
stored as a RooWorkspace. However, for the sake of interoperability
with other statistical frameworks, and also ease of manipulation, it
may be useful to store statistical models in text form.

The RooJSONFactoryWSTool is a helper class to achieve exactly this,
exporting to and importing from JSON.

In order to import a workspace from a JSON file, you can do

~~~ {.py}
ws = ROOT.RooWorkspace("ws")
tool = ROOT.RooJSONFactoryWSTool(ws)
tool.importJSON("myjson.json")
~~~

Similarly, in order to export a workspace to a JSON file, you can do

~~~ {.py}
tool = ROOT.RooJSONFactoryWSTool(ws)
tool.exportJSON("myjson.json")
~~~

Analogously, in C++, you can do

~~~ {.cxx}
#include "RooFitHS3/RooJSONFactoryWSTool.h"
// ...
RooWorkspace ws("ws");
RooJSONFactoryWSTool tool(ws);
tool.importJSON("myjson.json");
~~~

and

~~~ {.cxx}
#include "RooFitHS3/RooJSONFactoryWSTool.h"
// ...
RooJSONFactoryWSTool tool(ws);
tool.exportJSON("myjson.json");
~~~

For more details, consult the tutorial <a href="rf515__hfJSON_8py.html">rf515_hfJSON</a>.

The RooJSONFactoryWSTool only knows about a limited set of classes for
import and export. If import or export of a class you're interested in
fails, you might need to add your own importer or exporter. Please
consult the relevant section in the \ref roofit_dev_docs to learn how to do that (\ref roofit_dev_docs_hs3).

You can always get a list of all the available importers and exporters by calling the following functions:
~~~ {.py}
ROOT.RooFit.JSONIO.printImporters()
ROOT.RooFit.JSONIO.printExporters()
ROOT.RooFit.JSONIO.printFactoryExpressions()
ROOT.RooFit.JSONIO.printExportKeys()
~~~

Alternatively, you can generate a LaTeX version of the available importers and exporters by calling
~~~ {.py}
tool = ROOT.RooJSONFactoryWSTool(ws)
tool.writedoc("hs3.tex")
~~~
*/

constexpr auto hs3VersionTag = "0.3";

using RooFit::Detail::JSONNode;
using RooFit::Detail::JSONTree;

namespace {

std::vector<std::string> valsToStringVec(JSONNode const &node)
{
   std::vector<std::string> out;
   out.reserve(node.num_children());
   for (JSONNode const &elem : node.children()) {
      out.push_back(elem.val());
   }
   return out;
}

// True if the entire string parses as a number (integer or floating-point).
bool isNumber(const std::string &str)
{
   // Parse with the same mechanism as toDouble() and require that the whole string is consumed, so that isNumber(s) is
   // true exactly when toDouble(s) can turn the entire string into a value. (std::from_chars for floating-point types
   // is not portably available on all platforms ROOT supports, so we rely on the stream extraction instead.)
   std::istringstream stream(str);
   double value = 0.0;
   return (stream >> value) && stream.eof();
}

// Configure `v` (value, error, binning, constness) from the JSON node `p`.
void configureVariable(RooFit::JSONIO::Detail::Domains &domains, const JSONNode &p, RooRealVar &v)
{
   if (!p.has_child("name")) {
      RooJSONFactoryWSTool::error("cannot instantiate variable without \"name\"!");
   }
   if (auto n = p.find("value"))
      v.setVal(n->val_double());
   domains.writeVariable(v);
   if (auto n = p.find("nbins"))
      v.setBins(n->val_int());
   if (auto n = p.find("relErr"))
      v.setError(v.getVal() * n->val_double());
   if (auto n = p.find("err"))
      v.setError(n->val_double());
   if (auto n = p.find("const")) {
      v.setConstant(n->val_bool());
   } else {
      v.setConstant(false);
   }
}

JSONNode const *getVariablesNode(JSONNode const &rootNode)
{
   auto paramPointsNode = rootNode.find("parameter_points");
   if (!paramPointsNode)
      return nullptr;
   auto out = RooJSONFactoryWSTool::findNamedChild(*paramPointsNode, "default_values");
   if (out == nullptr)
      return nullptr;
   return &((*out)["parameters"]);
}

std::string genPrefix(const JSONNode &p, bool trailing_underscore)
{
   std::string prefix;
   if (!p.is_map())
      return prefix;
   if (auto node = p.find("namespaces")) {
      for (const auto &ns : node->children()) {
         if (!prefix.empty())
            prefix += "_";
         prefix += ns.val();
      }
   }
   if (trailing_underscore && !prefix.empty())
      prefix += "_";
   return prefix;
}

// helpers for serializing / deserializing binned datasets
void genIndicesHelper(std::vector<std::vector<int>> &combinations, std::vector<int> &curr_comb,
                      const std::vector<int> &vars_numbins, size_t curridx)
{
   if (curridx == vars_numbins.size()) {
      // we have filled a combination. Copy it.
      combinations.emplace_back(curr_comb);
   } else {
      for (int i = 0; i < vars_numbins[curridx]; ++i) {
         curr_comb[curridx] = i;
         ::genIndicesHelper(combinations, curr_comb, vars_numbins, curridx + 1);
      }
   }
}

// Import string attributes ("dict") and boolean tags ("tags") from `node` onto `arg`.
void importAttributes(RooAbsArg *arg, JSONNode const &node)
{
   if (auto seq = node.find("dict")) {
      for (const auto &attr : seq->children()) {
         arg->setStringAttribute(attr.key().c_str(), attr.val().c_str());
      }
   }
   if (auto seq = node.find("tags")) {
      for (const auto &attr : seq->children()) {
         arg->setAttribute(attr.val().c_str());
      }
   }
}

void addIfPresent(RooArgSet &out, RooArgSet const *args)
{
   if (args) {
      out.add(*args, true);
   }
}

void collectParameterStepWidthCandidatesFromModelConfigs(RooWorkspace const &workspace, RooArgSet &candidates,
                                                         RooArgSet &excluded)
{
   for (TObject *obj : workspace.allGenericObjects()) {
      auto const *mc = dynamic_cast<RooFit::ModelConfig const *>(obj);
      if (!mc) {
         continue;
      }

      addIfPresent(candidates, mc->GetParametersOfInterest());
      addIfPresent(candidates, mc->GetNuisanceParameters());

      addIfPresent(excluded, mc->GetObservables());
      addIfPresent(excluded, mc->GetGlobalObservables());
      addIfPresent(excluded, mc->GetConditionalObservables());
   }
}

void collectParameterStepWidthCandidatesFromPdfs(std::vector<RooAbsPdf *> const &pdfs,
                                                 std::vector<RooAbsData *> const &data, RooArgSet &candidates,
                                                 RooArgSet &excluded)
{
   for (RooAbsPdf const *pdf : pdfs) {
      RooArgSet observables;
      for (RooAbsData const *dataset : data) {
         std::unique_ptr<RooArgSet> pdfObs{pdf->getObservables(*dataset->get())};
         observables.add(*pdfObs, true);
      }

      if (observables.empty()) {
         continue;
      }

      RooArgSet params;
      pdf->getParameters(&observables, params);
      candidates.add(params, true);
      excluded.add(observables, true);
   }
}

void exportParameterStepWidths(RooWorkspace const &workspace, std::vector<RooAbsPdf *> const &pdfs,
                               std::vector<RooAbsData *> const &data, JSONNode &rootnode)
{
   RooArgSet candidates;
   RooArgSet excluded;

   collectParameterStepWidthCandidatesFromModelConfigs(workspace, candidates, excluded);
   collectParameterStepWidthCandidatesFromPdfs(pdfs, data, candidates, excluded);

   candidates.sort();

   JSONNode *parameterStepWidthsNode = nullptr;
   for (RooAbsArg *arg : candidates) {
      if (excluded.find(*arg)) {
         continue;
      }

      auto *var = dynamic_cast<RooRealVar *>(arg);
      if (!var || !var->hasError()) {
         continue;
      }

      if (!parameterStepWidthsNode) {
         parameterStepWidthsNode = &rootnode["misc"]["minimization"]["parameter_stepwidths"].set_seq();
      }

      JSONNode &stepWidthNode = RooJSONFactoryWSTool::appendNamedChild(*parameterStepWidthsNode, var->GetName());
      stepWidthNode["step_width"] << var->getError();
   }
}

void importParameterStepWidths(RooWorkspace &workspace, JSONNode const &rootnode)
{
   auto const *parameterStepWidthsNode = rootnode.find("misc", "minimization", "parameter_stepwidths");
   if (!parameterStepWidthsNode) {
      return;
   }
   if (!parameterStepWidthsNode->is_seq()) {
      RooJSONFactoryWSTool::warning("RooFitHS3: misc.minimization.parameter_stepwidths is not a sequence, skipping.");
      return;
   }

   for (JSONNode const &stepWidthNode : parameterStepWidthsNode->children()) {
      if (!stepWidthNode.is_map() || !stepWidthNode.has_child("name") || !stepWidthNode.has_child("step_width")) {
         RooJSONFactoryWSTool::warning("RooFitHS3: skipping malformed parameter_stepwidths entry.");
         continue;
      }

      const std::string name = RooJSONFactoryWSTool::name(stepWidthNode);
      RooAbsArg *arg = workspace.arg(name);
      auto *var = dynamic_cast<RooRealVar *>(arg);
      if (!var) {
         RooJSONFactoryWSTool::warning(
            "RooFitHS3: skipping parameter_stepwidths entry for unknown or non-real variable '" + name + "'.");
         continue;
      }

      var->setError(stepWidthNode.find("step_width")->val_double());
   }
}

// RooWSFactoryTool expression handling
std::string generate(const RooFit::JSONIO::ImportExpression &ex, const JSONNode &p, RooJSONFactoryWSTool *tool)
{
   std::stringstream expression;
   std::string classname(ex.tclass->GetName());
   size_t colon = classname.find_last_of(':');
   expression << (colon < classname.size() ? classname.substr(colon + 1) : classname);
   bool first = true;
   const auto &name = RooJSONFactoryWSTool::name(p);
   for (auto k : ex.arguments) {
      expression << (first ? "::" + name + "(" : ",");
      first = false;
      if (k == "true" || k == "false") {
         expression << (k == "true" ? "1" : "0");
      } else if (!p.has_child(k)) {
         RooJSONFactoryWSTool::error("node '" + name + "' is missing key '" + k + "'");
      } else if (p[k].is_seq()) {
         bool firstInner = true;
         expression << "{";
         for (RooAbsArg *arg : tool->requestArgList<RooAbsReal>(p, k)) {
            expression << (firstInner ? "" : ",") << arg->GetName();
            firstInner = false;
         }
         expression << "}";
      } else {
         tool->requestArg<RooAbsReal>(p, p[k].key());
         expression << p[k].val();
      }
   }
   expression << ")";
   return expression.str();
}

// Generate all combinations of bin indices for the RooRealVars in `vars`.
std::vector<std::vector<int>> generateBinIndices(const RooArgSet &vars)
{
   std::vector<std::vector<int>> combinations;
   std::vector<int> vars_numbins;
   vars_numbins.reserve(vars.size());
   for (const RooAbsArg *arg : vars) {
      // Continuous variables and categories (one bin per state)
      vars_numbins.push_back(dynamic_cast<RooAbsLValue const &>(*arg).numBins());
   }
   std::vector<int> curr_comb(vars.size());
   ::genIndicesHelper(combinations, curr_comb, vars_numbins, 0);
   return combinations;
}

template <typename... Keys_t>
JSONNode const *findRooFitInternal(JSONNode const &node, Keys_t const &...keys)
{
   return node.find("misc", "ROOT_internal", keys...);
}

// True if `arg` is a RooConstVar whose name is a plain number (i.e. a literal constant).
bool isLiteralConstVar(RooAbsArg const &arg)
{
   bool isRooConstVar = dynamic_cast<RooConstVar const *>(&arg);
   return isRooConstVar && isNumber(arg.GetName());
}

// Export the string attributes and tags of `arg` into the ROOT-internal attributes node.
void exportAttributes(const RooAbsArg *arg, JSONNode &rootnode)
{
   // If this RooConst is a literal number, we don't need to export the attributes.
   if (isLiteralConstVar(*arg)) {
      return;
   }

   JSONNode *node = nullptr;

   auto initializeNode = [&]() {
      if (node)
         return;

      node = &RooJSONFactoryWSTool::getRooFitInternal(rootnode, "attributes").set_map()[arg->GetName()].set_map();
   };

   if (dynamic_cast<RooConstVar const *>(arg)) {
      return;
   }

   // export all string attributes of an object
   if (!arg->stringAttributes().empty()) {
      for (const auto &it : arg->stringAttributes()) {
         // Skip some RooFit internals
         if (it.first == "factory_tag" || it.first == "PROD_TERM_TYPE")
            continue;
         initializeNode();
         (*node)["dict"].set_map()[it.first] << it.second;
      }
   }
   if (!arg->attributes().empty()) {
      for (auto const &attr : arg->attributes()) {
         // Skip some RooFit internals
         if (attr == "SnapShot_ExtRefClone" || attr == "RooRealConstant_Factory_Object")
            continue;
         initializeNode();
         (*node)["tags"].set_seq().append_child() << attr;
      }
   }
}

// Collect the observables named in the "axes" field of `node` from the workspace into `out`.
void getObservables(RooWorkspace const &ws, const JSONNode &node, RooAbsCollection &out)
{
   for (const auto &p : node["axes"].children()) {
      std::string name(RooJSONFactoryWSTool::name(p));
      if (p.has_child("categories")) {
         RooCategory *cat = ws.cat(name);
         if (!cat) {
            RooJSONFactoryWSTool::error("The categorical observable \"" + name +
                                        "\" could not be found in the workspace! Categorical observables are "
                                        "declared by a categorical axis in a domain.");
         }
         std::set<std::string> labels;
         for (auto const &label : p["categories"].children()) {
            labels.insert(label.val());
         }
         std::set<std::string> states;
         for (auto const &state : *cat) {
            states.insert(state.first);
         }
         if (labels != states) {
            RooJSONFactoryWSTool::error("The categories of the axis \"" + name +
                                        "\" differ from the states of the category \"" + name + "\"");
         }
         out.add(*cat);
      } else if (ws.var(name)) {
         out.add(*ws.var(name));
      } else {
         RooJSONFactoryWSTool::error("The observable \"" + name + "\" could not be found in the workspace!");
      }
   }
}

// Make the ranges of the observables of unbinned data consistent with the "min" and "max" of the axes. Variables that
// didn't get their range from a domain take it from the axis. If `domains` is null, all ranges are considered explicit.
void applyUnbinnedAxisRanges(const JSONNode &p, RooWorkspace &workspace, RooFit::JSONIO::Detail::Domains const *domains)
{
   for (const auto &axis : p["axes"].children()) {
      if (!axis.has_child("min") || !axis.has_child("max")) {
         continue;
      }
      const std::string name = RooJSONFactoryWSTool::name(axis);
      RooRealVar *var = workspace.var(name);
      if (!var) {
         continue; // reported by getObservables()
      }
      const double min = readBound(axis, "min", -RooNumber::infinity());
      const double max = readBound(axis, "max", RooNumber::infinity());
      if (domains && !domains->hasVariable(name.c_str())) {
         var->setRange(min, max);
      } else if (var->getMin() != min || var->getMax() != max) {
         std::stringstream ss;
         ss << "RooFitHS3: the range [" << min << ", " << max << "] of the axis '" << name << "' of the dataset '"
            << RooJSONFactoryWSTool::name(p) << "' differs from the range [" << var->getMin() << ", " << var->getMax()
            << "] of the variable.";
         RooJSONFactoryWSTool::warning(ss.str());
      }
   }
}

// Give the categories among `vars` the states of the categories with the same name in the workspace: the same
// label-to-index mapping and the same order. RooDataHist looks up the bin of a category by the order of the states of
// the category it is given, so data and workspace category have to agree on it. The workspace categories come from
// the domains; the labels must be the same, but the data may list them in a different order (HS3 identifies
// categories by name). readBinnedData() then permutes the bin contents accordingly.
void alignCategoriesWithWorkspace(RooArgSet &vars, RooWorkspace const &workspace, std::string const &dataName)
{
   for (RooAbsArg *arg : vars) {
      auto *cat = dynamic_cast<RooCategory *>(arg);
      if (!cat) {
         continue;
      }
      RooCategory const *wsCat = workspace.cat(cat->GetName());
      if (!wsCat) {
         continue;
      }
      std::vector<std::string> labels;
      for (int i = 0; i < cat->numTypes(); ++i) {
         labels.push_back(cat->lookupName(i));
      }
      if (static_cast<int>(labels.size()) != wsCat->numTypes() ||
          std::any_of(labels.begin(), labels.end(), [&](auto const &l) { return !wsCat->hasLabel(l); })) {
         RooJSONFactoryWSTool::error("the categories of the axis '" + std::string(cat->GetName()) +
                                     "' of the dataset '" + dataName + "' differ from those of the category '" +
                                     cat->GetName() + "'");
      }
      cat->clear();
      for (unsigned int i = 0; i < wsCat->size(); ++i) {
         auto const &state = wsCat->getOrdinal(i);
         cat->defineType(state.first, state.second);
      }
   }
}

// Create a RooAbsData (binned or unbinned) from the JSON node `p`.
std::unique_ptr<RooAbsData>
loadData(const JSONNode &p, RooWorkspace &workspace, RooFit::JSONIO::Detail::Domains const *domains)
{
   std::string name(RooJSONFactoryWSTool::name(p));

   RooJSONFactoryWSTool::testValidName(name, true);

   std::string const &type = p["type"].val();
   if (type == "binned") {
      // binned
      RooArgSet axes = RooJSONFactoryWSTool::readAxes(p);
      alignCategoriesWithWorkspace(axes, workspace, name);
      return RooJSONFactoryWSTool::readBinnedData(p, name, axes);
   } else if (type == "unbinned") {
      // unbinned
      RooArgList varlist;
      getObservables(workspace, p, varlist);
      applyUnbinnedAxisRanges(p, workspace, domains);
      RooArgSet vars(varlist);
      auto data = std::make_unique<RooDataSet>(name, name, vars, RooFit::WeightVar());
      auto &coords = p["entries"];
      if (!coords.is_seq()) {
         RooJSONFactoryWSTool::error("key 'entries' is not a list!");
      }
      std::vector<double> weightVals;
      if (p.has_child("weights")) {
         auto &weights = p["weights"];
         if (coords.num_children() != weights.num_children()) {
            RooJSONFactoryWSTool::error("inconsistent number of entries and weights!");
         }
         for (auto const &weight : weights.children()) {
            weightVals.push_back(weight.val_double());
         }
      }
      std::size_t i = 0;
      for (auto const &point : coords.children()) {
         if (!point.is_seq()) {
            RooJSONFactoryWSTool::error("coordinate point '" + std::to_string(i) + "' is not a list!");
         }
         if (point.num_children() != varlist.size()) {
            RooJSONFactoryWSTool::error("inconsistent number of entries and observables!");
         }
         std::size_t j = 0;
         for (auto const &pointj : point.children()) {
            if (auto *cat = dynamic_cast<RooCategory *>(varlist.at(j))) {
               // The entries of a categorical axis are the labels of the categories.
               if (pointj.is_number() || !cat->hasLabel(pointj.val())) {
                  RooJSONFactoryWSTool::error("entry " + std::to_string(i) + " of the dataset '" + name +
                                              "' has the value '" + pointj.val() + "' on the categorical axis '" +
                                              cat->GetName() + "', which is not one of its categories");
               }
               cat->setLabel(pointj.val());
            } else {
               static_cast<RooRealVar *>(varlist.at(j))->setVal(pointj.val_double());
            }
            ++j;
         }
         if (weightVals.size() > 0) {
            data->add(vars, weightVals[i]);
         } else {
            data->add(vars, 1.);
         }
         ++i;
      }
      return data;
   }

   RooJSONFactoryWSTool::error("RooJSONFactoryWSTool() failed to create dataset " + name);
}

// What the document says about a combined data set: its columns and the keys with a sub-data set, each key as a
// map from the column to the label. Used to check the pairing with a combined distribution in a likelihood.
struct CombinedDataInfo {
   std::vector<std::string> columns;
   std::set<std::map<std::string, std::string>> keys;
};

// Check that a combined data set paired with a RooSimultaneous in a likelihood has a sub-data set for every component.
// RooFit has no notion of an unobserved component: a key without sub-data set is a channel without entries, which
// adds the expected number of events of an extended component to the likelihood. That changes the fit, so it is an
// error (https://github.com/root-project/root/issues/24004). For a non-extended component it changes nothing, and a
// sub-data set without component is ignored by RooFit; both are reported as warnings.
void checkCombinedPairing(RooSimultaneous const &simPdf, std::string const &dataName, CombinedDataInfo const &info)
{
   std::vector<RooCategory const *> columns = selectionColumns(simPdf.indexCat());
   std::vector<std::string> columnNames;
   for (RooCategory const *column : columns) {
      columnNames.push_back(column->GetName());
   }
   std::vector<std::string> dataColumns = info.columns;
   std::sort(dataColumns.begin(), dataColumns.end());
   if (dataColumns != columnNames) {
      RooJSONFactoryWSTool::error("the likelihood pairs '" + std::string(simPdf.GetName()) +
                                  "' with the combined data set '" + dataName + "', but their columns differ");
   }
   auto keyToString = [&](std::map<std::string, std::string> const &key) {
      std::vector<std::string> labels;
      for (std::string const &column : info.columns) {
         labels.push_back(key.at(column));
      }
      return selectionKeyToString(labels);
   };
   std::set<std::map<std::string, std::string>> pdfKeys;
   for (std::vector<std::string> const &labels : selectionKeys(columns)) {
      RooAbsPdf const *component = simPdf.getPdf(selectionLabel(simPdf.indexCat(), columns, labels));
      if (!component) {
         continue;
      }
      std::map<std::string, std::string> key;
      for (std::size_t i = 0; i < columns.size(); ++i) {
         key[columnNames[i]] = labels[i];
      }
      pdfKeys.insert(key);
      if (info.keys.count(key) > 0) {
         continue;
      }
      if (component->canBeExtended()) {
         RooJSONFactoryWSTool::error(
            "the combined data set '" + dataName + "' has no sub-data set for the key " + keyToString(key) +
            " of the extended component '" + component->GetName() + "' of '" + simPdf.GetName() +
            "'. RooFit cannot leave a component unobserved: it would add the expected number of events of the " +
            "component to the likelihood, see https://github.com/root-project/root/issues/24004");
      }
      RooJSONFactoryWSTool::warning("RooFitHS3: the combined data set '" + dataName +
                                    "' has no sub-data set for the key " + keyToString(key) + " of the component '" +
                                    component->GetName() + "' of '" + simPdf.GetName() +
                                    "', which RooFit treats as present and empty.");
   }
   for (auto const &key : info.keys) {
      if (pdfKeys.count(key) == 0) {
         RooJSONFactoryWSTool::warning("RooFitHS3: the sub-data set for the key " + keyToString(key) +
                                       " of the combined data set '" + dataName + "' is ignored, because '" +
                                       simPdf.GetName() + "' has no component for it.");
      }
   }
}

// Import an analysis (likelihood + domains) as one or more ModelConfig objects into the workspace.
void importAnalysis(const JSONNode &rootnode, const JSONNode &analysisNode, const JSONNode &likelihoodsNode,
                    const JSONNode &domainsNode, RooWorkspace &workspace,
                    const std::vector<std::unique_ptr<RooAbsData>> &datasets,
                    std::map<std::string, CombinedDataInfo> const &combinedData)
{
   // if this is a toplevel pdf, also create a modelConfig for it
   std::string const &analysisName = RooJSONFactoryWSTool::name(analysisNode);
   JSONNode const *mcAuxNode = findRooFitInternal(rootnode, "ModelConfigs", analysisName);

   JSONNode const *mcNameNode = mcAuxNode ? mcAuxNode->find("mcName") : nullptr;
   std::string mcname = mcNameNode ? mcNameNode->val() : analysisName;
   if (workspace.obj(mcname))
      return;

   workspace.import(RooFit::ModelConfig{mcname.c_str(), mcname.c_str()});
   auto *mc = static_cast<RooFit::ModelConfig *>(workspace.obj(mcname));
   mc->SetWS(workspace);

   auto *nllNode = RooJSONFactoryWSTool::findNamedChild(likelihoodsNode, analysisNode["likelihood"].val());
   if (!nllNode) {
      throw std::runtime_error("likelihood node not found!");
   }
   if (!nllNode->has_child("distributions")) {
      throw std::runtime_error("likelihood node has no distributions attached!");
   }
   if (!nllNode->has_child("data")) {
      throw std::runtime_error("likelihood node has no data attached!");
   }
   std::vector<std::string> nllDistNames = valsToStringVec((*nllNode)["distributions"]);
   RooArgSet extConstraints;
   for (auto &nameNode : (*nllNode)["aux_distributions"].children()) {
      if (RooAbsArg *extConstraint = workspace.arg(nameNode.val())) {
         extConstraints.add(*extConstraint);
      }
   }
   RooArgSet observables;
   for (auto &nameNode : (*nllNode)["data"].children()) {
      bool found = false;
      for (const auto &d : datasets) {
         // Components of combined data sets have been moved out of the list.
         if (d && d->GetName() == nameNode.val()) {
            found = true;
            observables.add(*d->get(), true);
         }
      }
      if (nameNode.val() != "0" && !found)
         throw std::runtime_error("dataset '" + nameNode.val() + "' cannot be found!");
   }

   // A combined distribution paired with a combined data set must have a sub-data set for every component.
   std::vector<std::string> const nllDataNames = valsToStringVec((*nllNode)["data"]);
   for (std::size_t i = 0; i < std::min(nllDistNames.size(), nllDataNames.size()); ++i) {
      auto *simPdf = dynamic_cast<RooSimultaneous const *>(workspace.pdf(nllDistNames[i]));
      auto info = combinedData.find(nllDataNames[i]);
      if (simPdf && !simPdf->getAttribute(categoryConditionalTag) && info != combinedData.end()) {
         checkCombinedPairing(*simPdf, nllDataNames[i], info->second);
      }
   }

   JSONNode const *pdfNameNode = mcAuxNode ? mcAuxNode->find("pdfName") : nullptr;
   std::string const pdfName = pdfNameNode ? pdfNameNode->val() : "simPdf";

   RooAbsPdf *pdf = workspace.pdf(pdfName);

   if (!pdf) {
      // if there is no simultaneous pdf, we can check whether there is only one pdf in the list
      if (nllDistNames.size() == 1) {
         // if so, we can use that one to populate the ModelConfig
         pdf = workspace.pdf(nllDistNames[0]);
      } else {
         // otherwise, we have no choice but to build a simPdf by hand
         std::string simPdfName = analysisName + "_simPdf";
         std::string indexCatName = analysisName + "_categoryIndex";
         RooCategory indexCat{indexCatName.c_str(), indexCatName.c_str()};
         std::map<std::string, RooAbsPdf *> pdfMap;
         for (std::size_t i = 0; i < nllDistNames.size(); ++i) {
            indexCat.defineType(nllDistNames[i], i);
            pdfMap[nllDistNames[i]] = workspace.pdf(nllDistNames[i]);
         }
         RooSimultaneous simPdf{simPdfName.c_str(), simPdfName.c_str(), pdfMap, indexCat};
         workspace.import(simPdf, RooFit::RecycleConflictNodes(true), RooFit::Silence(true));
         pdf = workspace.pdf(simPdfName);
      }
   }

   mc->SetPdf(*pdf);

   if (!extConstraints.empty())
      mc->SetExternalConstraints(extConstraints);

   // HS3 v0.2.9 used the singular keys "parameter_of_interest" (a list) and "domain" (a single name). They are still
   // accepted, but the plural keys take precedence.
   auto findAnalysisKey = [&](const char *key, const char *legacyKey) -> JSONNode const * {
      if (auto node = analysisNode.find(key)) {
         return node;
      }
      if (auto node = analysisNode.find(legacyKey)) {
         RooJSONFactoryWSTool::warning(std::string("RooFitHS3: the key \"") + legacyKey + "\" of analysis '" +
                                       analysisName + "' is deprecated, please use \"" + key + "\" instead.");
         return node;
      }
      return nullptr;
   };
   JSONNode const *poiNode = findAnalysisKey("parameters_of_interest", "parameter_of_interest");
   JSONNode const *domainsNodeOfAnalysis = findAnalysisKey("domains", "domain");
   std::vector<std::string> analysisDomains;
   if (domainsNodeOfAnalysis && domainsNodeOfAnalysis->is_seq()) {
      analysisDomains = valsToStringVec(*domainsNodeOfAnalysis);
   } else if (domainsNodeOfAnalysis) {
      analysisDomains.push_back(domainsNodeOfAnalysis->val());
   }

   RooArgSet pois;
   if (poiNode) {
      for (auto const &child : poiNode->children()) {
         pois.add(*workspace.arg(child.val()));
      }
   }
   mc->SetParametersOfInterest(pois);
   mc->SetObservables(observables);
   RooArgSet pars;
   pdf->getParameters(&observables, pars);

   // Figure out the set parameters that appear in the main measurement:
   // getAllConstraints() has the side effect to remove all parameters from
   // "mainPars" that are not part of any pdf over observables.
   RooArgSet mainPars{pars};
   pdf->getAllConstraints(observables, mainPars, /*stripDisconnected*/ true);

   RooArgSet domainPars;
   for (std::string const &domain : analysisDomains) {
      const auto &thisDomain = RooJSONFactoryWSTool::findNamedChild(domainsNode, domain);
      if (!thisDomain || !thisDomain->has_child("axes"))
         continue;
      for (auto &var : (*thisDomain)["axes"].children()) {
         auto *wsvar = workspace.var(RooJSONFactoryWSTool::name(var));
         if (wsvar)
            domainPars.add(*wsvar);
      }
   }

   RooArgSet nps;
   RooArgSet globs;
   for (const auto &p : pars) {
      if (mc->GetParametersOfInterest()->find(*p))
         continue;
      if (p->isConstant() && !mainPars.find(*p) && domainPars.find(*p)) {
         globs.add(*p);
      } else if (domainPars.find(*p)) {
         nps.add(*p);
      }
   }

   mc->SetGlobalObservables(globs);
   mc->SetNuisanceParameters(nps);

   if (mcAuxNode) {
      if (auto found = mcAuxNode->find("combined_data_name")) {
         pdf->setStringAttribute("combined_data_name", found->val().c_str());
      }
   }

   if (analysisNode.has_child("init")) {
      const std::string initName = analysisNode["init"].val();
      RooArgSet const *init = workspace.getSnapshot(initName.c_str());
      if (!init) {
         RooJSONFactoryWSTool::warning("RooFitHS3: the parameter point '" + initName +
                                       "' used as \"init\" of analysis '" + analysisName + "' does not exist.");
         return;
      }
      // ModelConfig::SetSnapshot() saves the snapshot as "<mc name>_<set name>_snapshot". If "init" already follows
      // this pattern, which is the case for files written by RooFit, pick the set name such that it is preserved.
      std::string setName = initName;
      const std::string prefix = mcname + "_";
      const std::string suffix = "_snapshot";
      if (initName.size() >= prefix.size() + suffix.size() && startsWith(initName, prefix) &&
          endsWith(initName, suffix)) {
         setName = initName.substr(prefix.size(), initName.size() - prefix.size() - suffix.size());
      }
      // Take an owning copy: SetSnapshot() might replace the workspace snapshot that `init` points to.
      RooArgSet initSet;
      init->snapshot(initSet);
      initSet.setName(setName.c_str());
      mc->SetSnapshot(initSet);
   }
}

void combinePdfs(const JSONNode &rootnode, RooWorkspace &ws)
{
   auto *combinedPdfInfoNode = findRooFitInternal(rootnode, "combined_distributions");

   // If there is no info on combining pdfs
   if (combinedPdfInfoNode == nullptr) {
      return;
   }

   for (auto &info : combinedPdfInfoNode->children()) {

      // parse the information
      std::string combinedName = info.key();
      std::string indexCatName = info["index_cat"].val();
      std::vector<std::string> labels = valsToStringVec(info["labels"]);
      std::vector<int> indices;
      std::vector<std::string> pdfNames = valsToStringVec(info["distributions"]);
      for (auto &n : info["indices"].children()) {
         indices.push_back(n.val_int());
      }

      RooCategory indexCat{indexCatName.c_str(), indexCatName.c_str()};
      std::map<std::string, RooAbsPdf *> pdfMap;

      for (std::size_t iChannel = 0; iChannel < labels.size(); ++iChannel) {
         indexCat.defineType(labels[iChannel], indices[iChannel]);
         pdfMap[labels[iChannel]] = ws.pdf(pdfNames[iChannel]);
      }

      RooSimultaneous simPdf{combinedName.c_str(), combinedName.c_str(), pdfMap, indexCat};
      ws.import(simPdf, RooFit::RecycleConflictNodes(true), RooFit::Silence(true));
   }
}

void combineDatasets(const JSONNode &rootnode, std::vector<std::unique_ptr<RooAbsData>> &datasets)
{
   auto *combinedDataInfoNode = findRooFitInternal(rootnode, "combined_datasets");

   // If there is no info on combining datasets
   if (combinedDataInfoNode == nullptr) {
      return;
   }

   for (auto &info : combinedDataInfoNode->children()) {

      // parse the information
      std::string combinedName = info.key();
      std::string indexCatName = info["index_cat"].val();
      std::vector<std::string> labels = valsToStringVec(info["labels"]);
      std::vector<int> indices;
      for (auto &n : info["indices"].children()) {
         indices.push_back(n.val_int());
      }
      if (indices.size() != labels.size()) {
         RooJSONFactoryWSTool::error("mismatch in number of indices and labels!");
      }

      // Create the combined dataset for RooFit
      std::map<std::string, std::unique_ptr<RooAbsData>> dsMap;
      RooCategory indexCat{indexCatName.c_str(), indexCatName.c_str()};
      RooArgSet allVars{indexCat};
      for (std::size_t iChannel = 0; iChannel < labels.size(); ++iChannel) {
         auto componentName = combinedName + "_" + labels[iChannel];
         // We move the found channel data out of the "datasets" vector, such that
         // the data components don't get imported anymore.
         std::unique_ptr<RooAbsData> &component = *std::find_if(
            datasets.begin(), datasets.end(), [&](auto &d) { return d && d->GetName() == componentName; });
         if (!component)
            RooJSONFactoryWSTool::error("unable to obtain component matching component name '" + componentName + "'");
         allVars.add(*component->get(), true);
         dsMap.insert({labels[iChannel], std::move(component)});
         indexCat.defineType(labels[iChannel], indices[iChannel]);
      }

      auto combined = std::make_unique<RooDataSet>(combinedName, combinedName, allVars, RooFit::Import(dsMap),
                                                   RooFit::Index(indexCat));
      datasets.emplace_back(std::move(combined));
   }
}

template <class T>
void sortByName(T &coll)
{
   std::sort(coll.begin(), coll.end(), [](auto &l, auto &r) { return strcmp(l->GetName(), r->GetName()) < 0; });
}

// Build the combined data sets of the document ("type": "combined"): a RooDataSet with one category column per entry
// of "columns" and the variables of all components, filled row by row from the components, which are then no longer
// imported on their own. The columns are categories declared by categorical axes of the domains. A key without a
// sub-data set is a state without entries: RooFit has no notion of an absent component.
// The names of the variables that appear in more than one of `datasets`.
template <class DataSets, class GetData>
std::vector<std::string> sharedVariableNames(DataSets const &datasets, GetData getData)
{
   std::map<std::string, int> counts;
   for (auto const &item : datasets) {
      for (RooAbsArg const *var : *getData(item)->get()) {
         ++counts[var->GetName()];
      }
   }
   std::vector<std::string> shared;
   for (auto const &[name, count] : counts) {
      if (count > 1) {
         shared.push_back(name);
      }
   }
   return shared;
}

std::string joinQuoted(std::vector<std::string> const &names)
{
   std::string out;
   for (std::string const &name : names) {
      out += (out.empty() ? "'" : ", '") + name + "'";
   }
   return out;
}

std::map<std::string, CombinedDataInfo> assembleCombinedData(JSONNode const &dataNode, RooWorkspace const &workspace,
                                                             std::vector<std::unique_ptr<RooAbsData>> &datasets)
{
   std::map<std::string, CombinedDataInfo> infos;
   for (JSONNode const &p : dataNode.children()) {
      if (!p.is_map() || !p.has_child("type") || p["type"].val() != "combined") {
         continue;
      }
      std::string const name = RooJSONFactoryWSTool::name(p);
      RooJSONFactoryWSTool::testValidName(name, true);
      std::string const context = "the combined data set '" + name + "'";
      if (p.has_child("axes")) {
         RooJSONFactoryWSTool::error(context + " must not have \"axes\": its columns are declared by the domains");
      }
      std::vector<RooCategory *> columns = readSelectionColumns(workspace, p, "columns", context);
      auto const *componentsNode = p.find("components");
      if (!componentsNode || !componentsNode->is_seq() || componentsNode->num_children() == 0) {
         RooJSONFactoryWSTool::error("\"components\" of " + context + " must be a non-empty list");
      }

      struct Component {
         std::vector<std::string> key;
         std::unique_ptr<RooAbsData> data;
      };
      std::vector<Component> components;
      std::set<std::string> labels;
      for (JSONNode const &c : componentsNode->children()) {
         if (!c.is_map() || !c.has_child("dataset")) {
            RooJSONFactoryWSTool::error("every component of " + context + " must have a \"key\" and a \"dataset\"");
         }
         std::vector<std::string> key = readSelectionKey(c, columns, context);
         if (!labels.insert(selectionLabel(key)).second) {
            RooJSONFactoryWSTool::error("the key " + selectionKeyToString(key) + " is used by two components of " +
                                        context);
         }
         std::string const dataName = c["dataset"].val();
         auto found =
            std::find_if(datasets.begin(), datasets.end(), [&](auto &d) { return d && d->GetName() == dataName; });
         if (found == datasets.end()) {
            RooJSONFactoryWSTool::error("the component '" + dataName + "' of " + context +
                                        " is not a data set of the document, or it is already a component of " +
                                        "another combined data set");
         }
         components.push_back({std::move(key), std::move(*found)});
      }
      // HS3 requires distinct axis names, since the components of a combined distribution share no variate. RooFit
      // can still read such a document.
      std::vector<std::string> shared = sharedVariableNames(components, [](auto const &c) { return c.data.get(); });
      if (!shared.empty()) {
         RooJSONFactoryWSTool::warning("RooFitHS3: the sub-data sets of " + context + " share the axes " +
                                       joinQuoted(shared) +
                                       ". HS3 requires distinct axis names, because the components of a combined "
                                       "distribution share no variate. RooFit reads the document anyway.");
      }

      // The variables: the columns first, then those of the components. A weight variable if any component is
      // weighted, with errors if any component is binned, like RooDataSet does when importing a map of data sets.
      RooArgSet columnClones; // owns the clones; `vars` must not own anything, it also lists the components' variables
      for (RooCategory const *column : columns) {
         columnClones.addClone(*column);
      }
      RooArgSet vars{columnClones};
      bool weighted = false;
      bool storeErrors = false;
      std::string weightName = RooFit::WeightVar().getString(0);
      for (auto const &c : components) {
         vars.add(*c.data->get(), /*silent=*/true);
         weighted |= c.data->isWeighted();
         if (auto *ds = dynamic_cast<RooDataSet const *>(c.data.get()); ds && ds->isWeighted()) {
            weightName = ds->weightVar()->GetName();
         }
         storeErrors |= dynamic_cast<RooDataHist const *>(c.data.get()) != nullptr;
      }
      std::unique_ptr<RooDataSet> combined;
      if (weighted) {
         RooRealVar weightVar{weightName.c_str(), weightName.c_str(), 1.0};
         vars.add(weightVar);
         RooArgSet errorSet{weightVar};
         combined = std::make_unique<RooDataSet>(name.c_str(), name.c_str(), vars, RooFit::WeightVar(weightVar),
                                                 storeErrors ? RooFit::StoreError(errorSet) : RooCmdArg::none());
      } else {
         combined = std::make_unique<RooDataSet>(name.c_str(), name.c_str(), vars);
      }
      for (auto const &c : components) {
         for (std::size_t i = 0; i < columns.size(); ++i) {
            static_cast<RooCategory &>(vars[columns[i]->GetName()]).setLabel(c.key[i].c_str());
         }
         for (int i = 0; i < c.data->numEntries(); ++i) {
            vars.assign(*c.data->get(i));
            if (weighted) {
               combined->add(vars, c.data->weight(), storeErrors ? c.data->weightError(RooAbsData::SumW2) : 0.0);
            } else {
               combined->add(vars);
            }
         }
      }
      datasets.emplace_back(std::move(combined));

      CombinedDataInfo &info = infos[name];
      for (RooCategory const *column : columns) {
         info.columns.push_back(column->GetName());
      }
      for (auto const &c : components) {
         std::map<std::string, std::string> key;
         for (std::size_t i = 0; i < columns.size(); ++i) {
            key[columns[i]->GetName()] = c.key[i];
         }
         info.keys.insert(key);
      }
   }
   return infos;
}

/// Check whether an object is a top-level object of the workspace, i.e. not a
/// component of some other workspace object. Only clients that are part of the
/// workspace count: RooFit also registers evaluation artifacts as clients,
/// like the normalization integral that RooAbsPdf caches after
/// getVal(normSet), or the integral returned by createIntegral(). These live
/// outside the workspace and are not evidence that the object is a sub-node of
/// a bigger model (see https://github.com/root-project/root/issues/23221).
bool isTopLevel(RooAbsArg const &arg, RooWorkspace const &ws)
{
   for (RooAbsArg const *client : arg.clients()) {
      if (ws.components().containsInstance(*client)) {
         return false;
      }
   }
   return true;
}

/// Warn if the HS3 version of an imported document is not understood or newer than what RooFit writes.
void checkHS3Version(std::string const &version)
{
   int major = 0;
   int minor = 0;
   int supportedMajor = 0;
   int supportedMinor = 0;
   parseHS3Version(hs3VersionTag, supportedMajor, supportedMinor);
   if (!parseHS3Version(version, major, minor)) {
      RooJSONFactoryWSTool::warning("RooFitHS3: cannot parse hs3_version '" + version +
                                    "', expected a string of the form \"MAJOR.MINOR\".");
      return;
   }
   if (std::make_pair(major, minor) > std::make_pair(supportedMajor, supportedMinor)) {
      RooJSONFactoryWSTool::warning("RooFitHS3: the document has hs3_version '" + version +
                                    "', which is newer than the version supported by this RooFit ('" + hs3VersionTag +
                                    "'). Some content might not be imported correctly.");
   }
}

/// HS3 requires the names of all top-level objects to be unique within a document. If `strict`, throw if a name is
/// used twice within one section or by a distribution and a function. Other clashes, and all clashes if not
/// `strict`, only warn: RooFit's own export still produces them, e.g. for a range and a pdf of one name, and it
/// must be able to read its own output.
void checkUniqueNames(JSONNode const &rootnode, bool strict)
{
   std::map<std::string, std::string> sectionOfName;
   for (const char *section :
        {"distributions", "functions", "data", "domains", "parameter_points", "likelihoods", "analyses"}) {
      auto seq = rootnode.find(section);
      if (!seq || !seq->is_seq()) {
         continue;
      }
      for (JSONNode const &child : seq->children()) {
         if (!child.is_map() || !child.has_child("name")) {
            continue; // malformed entries are reported when they are imported
         }
         std::string const name = child["name"].val();
         auto inserted = sectionOfName.emplace(name, section);
         if (inserted.second) {
            continue;
         }
         std::string const &firstSection = inserted.first->second;
         // RooFit used to give analyses and likelihoods the name of the pdf or the analysis. Accept this for backwards
         // compatibility: analyses are not referenced by name, and likelihoods only from analyses, so there is no
         // ambiguity.
         auto isLegacyNamed = [](std::string const &s) { return s == "likelihoods" || s == "analyses"; };
         // Within one section, and between distributions and functions (both can be referenced by name), a clash is
         // ambiguous. Other clashes are produced by RooFit itself (a range, snapshot or data set named like a pdf).
         auto isReferable = [](std::string const &s) { return s == "distributions" || s == "functions"; };
         const bool ambiguous = firstSection == section || (isReferable(firstSection) && isReferable(section));
         if (!strict || !ambiguous || isLegacyNamed(firstSection) || isLegacyNamed(section)) {
            RooJSONFactoryWSTool::warning("RooFitHS3: the name '" + name + "' is used in \"" + firstSection +
                                          "\" and in \"" + section + "\", which is not allowed by the HS3 standard.");
            continue;
         }
         RooJSONFactoryWSTool::error("duplicate name '" + name + "': it is defined in \"" + firstSection +
                                     "\" and in \"" + section + "\", but names must be unique within the document");
      }
   }
}

} // namespace

RooJSONFactoryWSTool::RooJSONFactoryWSTool(RooWorkspace &ws) : _workspace{ws} {}

RooJSONFactoryWSTool::~RooJSONFactoryWSTool() {}

void RooJSONFactoryWSTool::fillSeq(JSONNode &node, RooAbsCollection const &coll, size_t nMax)
{
   const size_t old_children = node.num_children();
   node.set_seq();
   size_t n = 0;
   for (RooAbsArg const *arg : coll) {
      if (n >= nMax)
         break;
      if (isLiteralConstVar(*arg)) {
         node.append_child() << static_cast<RooConstVar const *>(arg)->getVal();
      } else {
         node.append_child() << arg->GetName();
      }
      ++n;
   }
   if (node.num_children() != old_children + coll.size()) {
      error("unable to stream collection " + std::string(coll.GetName()) + " to " + node.key());
   }
}

JSONNode &RooJSONFactoryWSTool::appendNamedChild(JSONNode &node, std::string const &name)
{
   if (!useListsInsteadOfDicts) {
      return node.set_map()[name].set_map();
   }
   JSONNode &child = node.set_seq().append_child().set_map();
   child["name"] << name;
   return child;
}

JSONNode const *RooJSONFactoryWSTool::findNamedChild(JSONNode const &node, std::string const &name)
{
   if (!useListsInsteadOfDicts) {
      if (!node.is_map())
         return nullptr;
      return node.find(name);
   }
   if (!node.is_seq())
      return nullptr;
   for (JSONNode const &child : node.children()) {
      if (child["name"].val() == name)
         return &child;
   }

   return nullptr;
}

/**
 * @brief Check if a string is a valid name.
 *
 * A valid name should start with a letter or an underscore, followed by letters, digits, or underscores.
 * Only characters from the ASCII character set are allowed.
 *
 * @param str The string to be checked.
 * @return bool Returns true if the string is a valid name; otherwise, returns false.
 */
bool RooJSONFactoryWSTool::isValidName(const std::string &str)
{
   // Check if the string is empty or starts with a non-letter/non-underscore character
   if (str.empty() || !(std::isalpha(str[0]) || str[0] == '_')) {
      return false;
   }

   // Check the remaining characters in the string
   for (char c : str) {
      // Allow letters, digits, and underscore
      if (!(std::isalnum(c) || c == '_')) {
         return false;
      }
   }

   // If all characters are valid, the string is a valid name
   return true;
}

bool RooJSONFactoryWSTool::testValidName(const std::string &name, bool forceError)
{
   if (!RooJSONFactoryWSTool::isValidName(name)) {
      std::stringstream ss;
      ss << "RooJSONFactoryWSTool() name '" << name << "' is not valid!" << std::endl
         << "Sanitize names by setting RooJSONFactoryWSTool::config().allowSanitizeNames = true." << std::endl;
      if (RooJSONFactoryWSTool::config().allowExportInvalidNames && !forceError) {
         RooJSONFactoryWSTool::warning(ss.str());
         return false;
      } else {
         RooJSONFactoryWSTool::error(ss.str());
      }
   }
   return true;
}

std::string RooJSONFactoryWSTool::name(const JSONNode &n)
{
   return useListsInsteadOfDicts ? n["name"].val() : n.key();
}

JSONNode &RooJSONFactoryWSTool::makeVariablesNode(JSONNode &rootNode)
{
   return appendNamedChild(rootNode["parameter_points"], "default_values")["parameters"];
}

template <>
RooRealVar *RooJSONFactoryWSTool::requestImpl<RooRealVar>(const std::string &objname)
{
   if (RooRealVar *retval = _workspace.var(objname))
      return retval;
   if (const auto *vars = getVariablesNode(*_rootnodeInput)) {
      if (const auto &node = vars->find(objname)) {
         this->importVariable(*node);
         if (RooRealVar *retval = _workspace.var(objname))
            return retval;
      }
   }
   return nullptr;
}

template <>
RooAbsPdf *RooJSONFactoryWSTool::requestImpl<RooAbsPdf>(const std::string &objname)
{
   if (RooAbsPdf *retval = _workspace.pdf(objname))
      return retval;
   auto it = _distributionsByName.find(objname);
   if (it != _distributionsByName.end()) {
      this->importFunction(*it->second, true);
      if (RooAbsPdf *retval = _workspace.pdf(objname))
         return retval;
   }
   return nullptr;
}

template <>
RooAbsReal *RooJSONFactoryWSTool::requestImpl<RooAbsReal>(const std::string &objname)
{
   if (RooAbsReal *retval = _workspace.function(objname))
      return retval;
   if (isNumber(objname))
      return &RooFit::RooConst(toDouble(objname));
   if (RooAbsPdf *pdf = requestImpl<RooAbsPdf>(objname))
      return pdf;
   if (RooRealVar *var = requestImpl<RooRealVar>(objname))
      return var;
   if (RooAbsReal *retval = _workspace.function(objname))
      return retval;
   auto it = _functionsByName.find(objname);
   if (it != _functionsByName.end()) {
      this->importFunction(*it->second, true);
      if (RooAbsReal *retval = _workspace.function(objname))
         return retval;
   }
   return nullptr;
}

// Export a single variable (RooRealVar or RooConstVar) `v` as a named child of `node`.
void RooJSONFactoryWSTool::exportVariable(const RooAbsArg *v, JSONNode &node, bool storeConstant, bool storeBins)
{
   // A categorical parameter: its value is the label. The columns of combined distributions are not parameters.
   if (auto *cat = dynamic_cast<const RooCategory *>(v)) {
      if (_columnCategories.count(cat->GetName()) > 0) {
         return;
      }
      JSONNode &var = appendNamedChild(node, cat->GetName());
      var["value"] << cat->getCurrentLabel();
      _domains->readCategory(*cat);
      return;
   }
   auto *cv = dynamic_cast<const RooConstVar *>(v);
   auto *rrv = dynamic_cast<const RooRealVar *>(v);
   if (!cv && !rrv)
      return;

   // for RooConstVar, if name and value are the same, we don't need to do anything
   if (cv && strcmp(cv->GetName(), TString::Format("%g", cv->getVal()).Data()) == 0) {
      return;
   }

   JSONNode &var = appendNamedChild(node, v->GetName());

   if (cv) {
      var["value"] << cv->getVal();
      var["const"] << true;
   } else if (rrv) {
      var["value"] << rrv->getVal();
      if (storeConstant && (rrv->isConstant() || rrv->getMin() >= rrv->getMax())) {
         var["const"] << true;
      } else if (storeBins) {
         var["min"] << rrv->getMin();
         var["max"] << rrv->getMax();
      }
      if (rrv->getBins() != 0 && storeBins) {
         var["nbins"] << rrv->getBins();
      }
      _domains->readVariable(*rrv);
   }
}

// Export all variables in `allElems` as a sequence under `n`.
void RooJSONFactoryWSTool::exportVariables(const RooArgSet &allElems, JSONNode &n, bool storeConstant, bool storeBins)
{
   // export a list of RooRealVar objects
   n.set_seq();
   for (RooAbsArg *arg : allElems) {
      exportVariable(arg, n, storeConstant, storeBins);
   }
}

std::string RooJSONFactoryWSTool::exportTransformed(const RooAbsReal *original, const std::string &suffix,
                                                    const std::string &formula)
{
   std::string newname = std::string(original->GetName()) + suffix;
   // Another object that shares the parameter might have exported the helper already.
   if (findNamedChild((*_rootnodeOutput)["functions"], newname)) {
      return newname;
   }
   RooFit::Detail::JSONNode &trafo_node = appendNamedChild((*_rootnodeOutput)["functions"], newname);
   trafo_node["type"] << "generic";
   trafo_node["expression"] << TString::Format(formula.c_str(), original->GetName()).Data();
   this->setAttribute(newname, "roofit_skip"); // this function should not be imported back in
   return newname;
}

// Export a single object `func` (pdf, function, variable or category) to the output JSON, recording its name in
// `exportedObjectNames` to avoid exporting it twice.
void RooJSONFactoryWSTool::exportObject(RooAbsArg const &func, std::set<std::string> &exportedObjectNames)
{
   std::string name = func.GetName();

   // if this element was already exported, skip
   if (exportedObjectNames.find(name) != exportedObjectNames.end())
      return;

   exportedObjectNames.insert(name);

   if (auto simPdf = dynamic_cast<RooSimultaneous const *>(&func)) {
      exportSimultaneous(*simPdf, exportedObjectNames);
      return;
   } else if (dynamic_cast<RooAbsCategory const *>(&func)) {
      // Categories are not objects of the document: the index category of a RooSimultaneous is declared by the
      // categorical axes of the domain that its "columns" name.
      return;
   } else if (dynamic_cast<RooRealVar const *>(&func) || dynamic_cast<RooConstVar const *>(&func)) {
      exportVariable(&func, *_varsNode, true, false);
      return;
   }

   auto &collectionNode = (*_rootnodeOutput)[dynamic_cast<RooAbsPdf const *>(&func) ? "distributions" : "functions"];

   auto const &exporters = RooFit::JSONIO::exporters();
   auto const &exportKeys = RooFit::JSONIO::exportKeys();

   TClass *cl = func.IsA();

   auto &elem = appendNamedChild(collectionNode, name);

   auto it = exporters.find(cl);
   if (it != exporters.end()) { // check if we have a specific exporter available
      for (auto &exp : it->second) {
         _serversToExport.clear();
         _serversToDelete.clear();
         if (!exp->exportObject(this, &func, elem)) {
            // The exporter might have messed with the content of the node
            // before failing. That's why we clear it and only reset the name.
            elem.clear();
            elem.set_map();
            if (useListsInsteadOfDicts) {
               elem["name"] << name;
            }
            continue;
         }
         if (exp->autoExportDependants()) {
            exportObjects(func.servers(), exportedObjectNames);
         } else {
            exportObjects(_serversToExport, exportedObjectNames);
         }
         for (auto &s : _serversToDelete) {
            delete s;
         }
         return;
      }
   }

   // generic export using the factory expressions
   const auto &dict = exportKeys.find(cl);
   if (dict == exportKeys.end()) {
      std::cerr << "unable to export class '" << cl->GetName() << "' - no export keys available!\n"
                << "there are several possible reasons for this:\n"
                << " 1. " << cl->GetName() << " is a custom class that you or some package you are using added.\n"
                << " 2. " << cl->GetName()
                << " is a ROOT class that nobody ever bothered to write a serialization definition for.\n"
                << " 3. something is wrong with your setup, e.g. you might have called "
                   "RooFit::JSONIO::clearExportKeys() and/or never successfully read a file defining these "
                   "keys with RooFit::JSONIO::loadExportKeys(filename)\n"
                << "either way, please make sure that:\n"
                << " 3: you are reading a file with export keys - call RooFit::JSONIO::printExportKeys() to "
                   "see what is available\n"
                << " 2 & 1: you might need to write a serialization definition yourself. check "
                   "https://root.cern/doc/master/group__roofit__dev__docs__hs3.html to "
                   "see how to do this!\n";
      return;
   }

   elem["type"] << dict->second.type;

   size_t nprox = func.numProxies();

   for (size_t i = 0; i < nprox; ++i) {
      RooAbsProxy *p = func.getProxy(i);
      if (!p)
         continue;

      // some proxies start with a "!". This is a magic symbol that we don't want to stream
      std::string pname(p->name());
      if (pname[0] == '!')
         pname.erase(0, 1);

      auto k = dict->second.proxies.find(pname);
      if (k == dict->second.proxies.end()) {
         std::cerr << "failed to find key matching proxy '" << pname << "' for type '" << dict->second.type
                   << "', encountered in '" << func.GetName() << "', skipping" << std::endl;
         return;
      }

      // empty string is interpreted as an instruction to ignore this value
      if (k->second.empty())
         continue;

      if (auto l = dynamic_cast<RooAbsCollection *>(p)) {
         fillSeq(elem[k->second], *l);
      }
      if (auto r = dynamic_cast<RooArgProxy *>(p)) {
         if (isLiteralConstVar(*r->absArg())) {
            elem[k->second] << static_cast<RooConstVar *>(r->absArg())->getVal();
         } else {
            elem[k->second] << r->absArg()->GetName();
         }
      }
   }

   // export all the servers of a given RooAbsArg
   for (RooAbsArg *s : func.servers()) {
      if (!s) {
         std::cerr << "unable to locate server of " << func.GetName() << std::endl;
         continue;
      }
      this->exportObject(*s, exportedObjectNames);
   }
}

/**
 * @brief Export a RooSimultaneous as a "combined_dist", or as a "category_conditional_dist" if it carries the tag.
 *
 * The components are selected by a key with one label per column. The columns are the index category of the
 * RooSimultaneous or, for a RooSuperCategory index, its input categories, sorted by name. They are declared as
 * categorical axes of the default domain. States without a pdf are not written: HS3 does not require a component
 * for every key.
 *
 * For a "category_conditional_dist" the columns are categorical parameters ("categorical_parameter") selecting
 * the component the RooSimultaneous evaluates to, and their values are written to the parameter points.
 */
void RooJSONFactoryWSTool::exportSimultaneous(RooSimultaneous const &simPdf, std::set<std::string> &exportedObjectNames)
{
   RooAbsCategory const &indexCat = simPdf.indexCat();
   std::vector<RooCategory const *> columns = selectionColumns(indexCat);
   const bool conditional = simPdf.getAttribute(categoryConditionalTag);

   JSONNode &elem = appendNamedChild((*_rootnodeOutput)["distributions"], simPdf.GetName());
   elem["type"] << (conditional ? "category_conditional_dist" : "combined_dist");
   auto &columnsNode = elem[conditional ? "categorical_parameter" : "columns"].set_seq();
   for (RooCategory const *column : columns) {
      columnsNode.append_child() << column->GetName();
      _domains->readCategory(*column);
      if (!conditional) {
         _columnCategories.insert(column->GetName());
      }
   }
   auto &components = elem["components"].set_seq();
   // Components with renamed observables replace the original ones (Config::renameCombinedObservables).
   auto renamed = _renamedComponents.find(simPdf.GetName());
   std::set<RooAbsArg const *> replaced;
   std::vector<RooAbsArg const *> replacements;
   // In the order of the columns and of their states, which is the order in which the categories are written to
   // the domain. This does not depend on the order of the inputs of a RooSuperCategory, so the document is the same
   // after a round trip.
   for (std::vector<std::string> const &key : selectionKeys(columns)) {
      const std::string label = selectionLabel(indexCat, columns, key);
      RooAbsPdf const *pdf = simPdf.getPdf(label);
      if (!pdf) {
         continue;
      }
      if (renamed != _renamedComponents.end()) {
         auto found = renamed->second.find(label);
         if (found != renamed->second.end()) {
            replaced.insert(pdf);
            pdf = found->second;
            replacements.push_back(pdf);
         }
      }
      JSONNode &component = components.append_child().set_map();
      component["key"].fill_seq(key);
      component["dist"] << pdf->GetName();
   }

   std::vector<RooAbsArg const *> servers;
   for (RooAbsArg const *server : simPdf.servers()) {
      if (replaced.count(server) == 0) {
         servers.push_back(server);
      }
   }
   servers.insert(servers.end(), replacements.begin(), replacements.end());
   exportObjects(servers, exportedObjectNames);
}

/**
 * @brief Import a function from the JSONNode into the workspace.
 *
 * This function imports a function from the given JSONNode into the workspace.
 * The function's information is read from the JSONNode and added to the workspace.
 *
 * @param p The JSONNode representing the function to be imported.
 * @param importAllDependants A boolean flag indicating whether to import all dependants (servers) of the function.
 * @return void
 */
void RooJSONFactoryWSTool::importFunction(const JSONNode &p, bool importAllDependants)
{
   std::string name(RooJSONFactoryWSTool::name(p));

   // If this node if marked to be skipped by RooFit, exit
   if (hasAttribute(name, "roofit_skip")) {
      return;
   }

   auto const &importers = RooFit::JSONIO::importers();
   auto const &factoryExpressions = RooFit::JSONIO::importExpressions();

   // some preparations: what type of function are we dealing with here?
   RooJSONFactoryWSTool::testValidName(name, true);

   // if the RooAbsArg already exists, we don't need to do anything
   if (_workspace.arg(name)) {
      return;
   }
   // if the key we found is not a map, it's an error
   if (!p.is_map()) {
      RooJSONFactoryWSTool::error("RooJSONFactoryWSTool() function node " + name + " is not a map!");
   }
   std::string prefix = genPrefix(p, true);
   if (!prefix.empty())
      name = prefix + name;
   if (!p.has_child("type")) {
      RooJSONFactoryWSTool::error("RooJSONFactoryWSTool() no type given for function '" + name + "', skipping.");
   }

   std::string functype(p["type"].val());

   // import all dependents if importing a workspace, not for creating new objects
   if (!importAllDependants) {
      this->importDependants(p);
   }

   // check for specific implementations
   auto it = importers.find(functype);
   bool ok = false;
   if (it != importers.end()) {
      for (auto &imp : it->second) {
         try {
            ok = imp->importArg(this, p);
         } catch (const std::exception &e) {
            std::stringstream ss;
            const auto *ptr = imp.get();
            ss << "RooJSONFactoryWSTool() failed. The importer " << typeid(*ptr).name()
               << " emitted and error: " << e.what() << std::endl;
            RooJSONFactoryWSTool::error(ss.str());
         }
         if (ok)
            break;
      }
   }
   if (!ok) { // generic import using the factory expressions
      auto expr = factoryExpressions.find(functype);
      if (expr != factoryExpressions.end()) {
         std::string expression = ::generate(expr->second, p, this);
         if (!_workspace.factory(expression)) {
            std::stringstream ss;
            ss << "RooJSONFactoryWSTool() failed to create " << expr->second.tclass->GetName() << " '" << name
               << "', skipping. expression was\n"
               << expression << std::endl;
            RooJSONFactoryWSTool::error(ss.str());
         }
      } else {
         std::stringstream ss;
         ss << "RooJSONFactoryWSTool() no handling for type '" << functype << "' implemented, skipping."
            << "\n"
            << "there are several possible reasons for this:\n"
            << " 1. " << functype << " is a custom type that is not available in RooFit.\n"
            << " 2. " << functype
            << " is a ROOT class that nobody ever bothered to write a deserialization definition for.\n"
            << " 3. something is wrong with your setup, e.g. you might have called "
               "RooFit::JSONIO::clearFactoryExpressions() and/or never successfully read a file defining "
               "these expressions with RooFit::JSONIO::loadFactoryExpressions(filename)\n"
            << "either way, please make sure that:\n"
            << " 3: you are reading a file with factory expressions - call "
               "RooFit::JSONIO::printFactoryExpressions() "
               "to see what is available\n"
            << " 2 & 1: you might need to write a deserialization definition yourself. check "
               "https://root.cern/doc/master/group__roofit__dev__docs__hs3.html to see "
               "how to do this!"
            << std::endl;
         RooJSONFactoryWSTool::error(ss.str());
         return;
      }
   }
   RooAbsReal *func = _workspace.function(name);
   if (!func) {
      RooJSONFactoryWSTool::error("something went wrong importing function '" + name + "'.");
   }
}

/**
 * @brief Import a function from a JSON string into the workspace.
 *
 * This function imports a function from the provided JSON string into the workspace.
 * The function's information is read from the JSON string and added to the workspace.
 *
 * @param jsonString The JSON string containing the function information.
 * @param importAllDependants A boolean flag indicating whether to import all dependants (servers) of the function.
 * @return void
 */
void RooJSONFactoryWSTool::importFunction(const std::string &jsonString, bool importAllDependants)
{
   this->importFunction((JSONTree::create(jsonString))->rootnode(), importAllDependants);
}

/**
 * @brief Export the name and binning of a RooRealVar to a JSONNode.
 *
 * @param obsNode The JSONNode to which the axis information will be exported.
 * @param var The RooRealVar representing the axis to be exported.
 * @return void
 */
void RooJSONFactoryWSTool::exportAxis(JSONNode &obsNode, RooRealVar const &var)
{
   std::string name = var.GetName();
   RooJSONFactoryWSTool::testValidName(name, false);
   obsNode["name"] << name;

   writeAxisBinning(obsNode, var.getBinning());
   // Binned axes always carry their range, also if it is redundant with the "edges" of an irregular binning.
   if (!obsNode.has_child("min")) {
      obsNode["min"] << var.getBinning().lowBound();
      obsNode["max"] << var.getBinning().highBound();
   }
}

// A categorical axis: the labels in the order of the states of the category, which is the order of the bins of a
// RooDataHist.
void RooJSONFactoryWSTool::exportCategoricalAxis(JSONNode &obsNode, RooCategory const &cat)
{
   testValidName(cat.GetName(), false);
   obsNode["name"] << cat.GetName();
   auto &labels = obsNode["categories"].set_seq();
   for (unsigned int i = 0; i < cat.size(); ++i) {
      labels.append_child() << cat.getOrdinal(i).first;
   }
}

/**
 * @brief Export histogram data to a JSONNode.
 *
 * This function exports histogram data, represented by the provided variables and contents, to a JSONNode.
 * The histogram's axes information and bin contents are added as key-value pairs to the JSONNode.
 *
 * @param vars The RooArgSet representing the variables associated with the histogram.
 * @param n The number of bins in the histogram.
 * @param contents A pointer to the array containing the bin contents of the histogram.
 * @param output The JSONNode to which the histogram data will be exported.
 * @return void
 */
void RooJSONFactoryWSTool::exportHisto(RooArgSet const &vars, std::size_t n, double const *contents, JSONNode &output)
{
   auto &observablesNode = output["axes"].set_seq();
   // axes have to be ordered to get consistent bin indices
   for (RooAbsArg const *var : vars) {
      if (auto const *cat = dynamic_cast<RooCategory const *>(var)) {
         exportCategoricalAxis(observablesNode.append_child().set_map(), *cat);
      } else {
         exportAxis(observablesNode.append_child().set_map(), static_cast<RooRealVar const &>(*var));
      }
   }

   return exportArray(n, contents, output["contents"]);
}

/**
 * @brief Export an array of doubles to a JSONNode.
 *
 * This function exports an array of doubles, represented by the provided size and contents,
 * to a JSONNode. The array elements are added to the JSONNode as a sequence of values.
 *
 * @param n The size of the array.
 * @param contents A pointer to the array containing the double values.
 * @param output The JSONNode to which the array will be exported.
 * @return void
 */
void RooJSONFactoryWSTool::exportArray(std::size_t n, double const *contents, JSONNode &output)
{
   output.set_seq();
   for (std::size_t i = 0; i < n; ++i) {
      double w = contents[i];
      // To make sure there are no unnecessary floating points in the JSON
      if (int(w) == w) {
         output.append_child() << int(w);
      } else {
         output.append_child() << w;
      }
   }
}

namespace {

// Turn an arbitrary string into a valid variable name, but refuse to change the
// first character (which would silently rename the object).
std::string makeValidNameOrError(std::string const &in)
{
   if (!std::isalpha(in[0])) {
      RooJSONFactoryWSTool::error("refusing to change first character of string '" + in + "' to make a valid name!");
   }
   std::string out = RooFit::Detail::makeValidVarName(in);
   if (out != in) {
      oocoutW(nullptr, ObjectHandling) << "RooFitHS3: changed '" << in << "' to '" << out << "' to become a valid name";
   }
   return out;
}

// The labels of a selection key, made valid names and joined by "_". Used in the names of sub-data sets and of
// renamed observables.
std::string selectionKeyName(std::vector<std::string> const &key)
{
   std::string out;
   for (std::string const &label : key) {
      out += (out.empty() ? "" : "_") + makeValidNameOrError(label);
   }
   return out;
}

// The names of the category columns of `vars`.
std::vector<std::string> selectionColumnNames(RooAbsCollection const &vars)
{
   std::vector<std::string> names;
   for (RooCategory const *column : selectionColumns(vars)) {
      names.push_back(column->GetName());
   }
   return names;
}

// The RooSimultaneous of the workspace (the first by name) whose columns are `columns`, or nullptr.
RooSimultaneous const *findMatchingSimultaneous(RooWorkspace const &workspace, std::vector<std::string> const &columns)
{
   std::vector<RooSimultaneous const *> candidates;
   for (RooAbsArg *arg : workspace.allPdfs()) {
      if (auto *candidate = dynamic_cast<RooSimultaneous const *>(arg)) {
         candidates.push_back(candidate);
      }
   }
   std::sort(candidates.begin(), candidates.end(),
             [](auto *a, auto *b) { return std::string(a->GetName()) < std::string(b->GetName()); });
   for (RooSimultaneous const *candidate : candidates) {
      std::vector<std::string> candidateColumns;
      for (RooCategory const *column : selectionColumns(candidate->indexCat())) {
         candidateColumns.push_back(column->GetName());
      }
      if (candidateColumns == columns) {
         return candidate;
      }
   }
   return nullptr;
}

} // namespace

/**
 * @brief Export a data set with category observables as a "combined" data set.
 *
 * The categories are the columns, sorted by name, and are declared as categorical axes of the default domain. There
 * is one sub-data set per key, i.e. per combination of states, written like any other data set and named after the
 * data set and the labels of the key. A key is written also without entries: RooFit treats a channel without entries
 * as present and empty. If the workspace has a RooSimultaneous with the same columns, each sub-data set only carries
 * the observables of the corresponding component, and keys without entries for which it has no component are left
 * out; otherwise, every sub-data set carries all observables.
 */
// Config::renameCombinedObservables: work out, before anything is exported, the new names of the observables of
// every combined data set, and the components of the matching RooSimultaneous with these observables. One
// observable per key is named "obs_<key>", several are named "obs_<key>_<observable>". Observables that already
// have their new name are left alone, so that a second export of an imported document gives the same result.
void RooJSONFactoryWSTool::prepareCombinedObservableRenaming()
{
   _combinedObservableRenames.clear();
   _renamedComponents.clear();
   _renamedObjects.clear();
   if (!config().renameCombinedObservables) {
      return;
   }
   std::vector<RooAbsData *> datasets;
   for (RooAbsData *data : _workspace.allData()) {
      datasets.push_back(data);
   }
   sortByName(datasets);
   for (RooAbsData const *data : datasets) {
      std::vector<RooCategory const *> columns = selectionColumns(*data->get());
      if (columns.empty()) {
         continue;
      }
      RooSimultaneous const *simPdf = findMatchingSimultaneous(_workspace, selectionColumnNames(*data->get()));
      for (std::vector<std::string> const &key : selectionKeys(columns)) {
         const std::string keyName = selectionKeyName(key);
         RooArgSet candidates;
         RooAbsPdf const *component = nullptr;
         std::string label;
         if (simPdf) {
            label = selectionLabel(simPdf->indexCat(), columns, key);
            component = simPdf->getPdf(label);
            if (!component) {
               continue;
            }
            std::unique_ptr<RooArgSet> observables{component->getObservables(*data->get())};
            candidates.add(*observables);
         } else {
            candidates.add(*data->get());
         }
         std::vector<RooAbsArg *> observables;
         for (RooAbsArg *arg : candidates) {
            if (!dynamic_cast<RooAbsCategory *>(arg)) {
               observables.push_back(arg);
            }
         }
         sortByName(observables);
         std::map<std::string, std::string> renames;
         for (RooAbsArg const *obs : observables) {
            std::string const newName =
               "obs_" + keyName + (observables.size() == 1 ? "" : "_" + std::string(obs->GetName()));
            if (newName == obs->GetName()) {
               continue;
            }
            if (_workspace.arg(newName.c_str())) {
               error("can't rename the observable '" + std::string(obs->GetName()) + "' of the data set '" +
                     data->GetName() + "' to '" + newName + "': an object of that name already exists");
            }
            renames[obs->GetName()] = newName;
         }
         if (renames.empty()) {
            continue;
         }
         _combinedObservableRenames[data->GetName()][keyName] = renames;
         if (!simPdf || _renamedComponents[simPdf->GetName()].count(label) > 0) {
            continue;
         }
         // The component with the renamed observables. Nodes that depend on them are cloned with the suffix
         // "_<key>", so that nodes shared by several components get one copy per component.
         RooCustomizer customizer{*component, keyName.c_str()};
         for (auto const &[from, to] : renames) {
            RooAbsArg *original = candidates.find(from.c_str());
            _renamedObjects.emplace_back(static_cast<RooAbsArg *>(original->clone(to.c_str())));
            customizer.replaceArg(*original, *_renamedObjects.back());
         }
         auto *renamedComponent = static_cast<RooAbsPdf *>(customizer.build());
         _renamedObjects.emplace_back(renamedComponent);
         _renamedComponents[simPdf->GetName()][label] = renamedComponent;
      }
   }
}

RooJSONFactoryWSTool::CombinedData RooJSONFactoryWSTool::exportCombinedData(RooAbsData const &data)
{
   std::vector<RooCategory const *> columns = selectionColumns(*data.get());
   CombinedData info;
   info.name = data.GetName();
   for (RooCategory const *column : columns) {
      info.columns.push_back(column->GetName());
      _domains->readCategory(*column);
   }

   // Split the data by the key: by the index category of a matching RooSimultaneous, or by the columns themselves.
   RooSimultaneous const *simPdf = findMatchingSimultaneous(_workspace, info.columns);
   std::unique_ptr<RooSuperCategory> superCat;
   RooAbsCategory const *indexCat = nullptr;
   std::vector<std::unique_ptr<RooAbsData>> parts;
   if (simPdf) {
      indexCat = &simPdf->indexCat();
      parts = data.split(*simPdf, true);
   } else if (columns.size() == 1) {
      indexCat = columns.front();
      parts = data.split(*indexCat, true);
   } else {
      RooArgSet inputs;
      for (RooCategory const *column : columns) {
         inputs.add(*column);
      }
      const std::string superName = info.name + "_key";
      superCat = std::make_unique<RooSuperCategory>(superName.c_str(), superName.c_str(), inputs);
      indexCat = superCat.get();
      parts = data.split(*indexCat, true);
   }

   // Fill the "combined" entry before exporting the sub-data sets: appending to "data" can invalidate references.
   std::vector<RooAbsData *> subDataSets;
   {
      JSONNode &output = appendNamedChild((*_rootnodeOutput)["data"], info.name);
      output["type"] << "combined";
      output["columns"].fill_seq(info.columns);
      auto &components = output["components"].set_seq();
      for (std::vector<std::string> const &key : selectionKeys(columns)) {
         std::string const label = selectionLabel(*indexCat, columns, key);
         auto found = std::find_if(parts.begin(), parts.end(), [&](auto &d) { return d->GetName() == label; });
         if (found == parts.end() || ((*found)->numEntries() == 0 && simPdf && !simPdf->getPdf(label))) {
            continue;
         }
         const std::string keyName = selectionKeyName(key);
         const std::string subName = info.name + "_" + keyName;
         (*found)->SetName(subName.c_str());
         // Config::renameCombinedObservables: give the observables of this key their own names.
         auto renamesOfData = _combinedObservableRenames.find(info.name);
         if (renamesOfData != _combinedObservableRenames.end()) {
            auto renames = renamesOfData->second.find(keyName);
            if (renames != renamesOfData->second.end()) {
               for (auto const &[from, to] : renames->second) {
                  (*found)->changeObservableName(from.c_str(), to.c_str());
               }
            }
         }
         JSONNode &component = components.append_child().set_map();
         component["key"].fill_seq(key);
         component["dataset"] << subName;
         subDataSets.push_back(found->get());
      }
   }
   // In HS3, the components of a combined distribution share no variate: a quantity measured in several components is a
   // separate variate in each of them, e.g. "mass_SR" and "mass_CR". The sub-data sets, which are split by the
   // components of the RooSimultaneous, must therefore not share observables.
   if (simPdf) {
      std::vector<std::string> shared = sharedVariableNames(subDataSets, [](RooAbsData const *d) { return d; });
      if (!shared.empty()) {
         error("the data set '" + info.name + "' is split by the components of '" + simPdf->GetName() +
               "', which share the observables " + joinQuoted(shared) +
               ". HS3 requires each component of a combined distribution to have its own variates: give the "
               "observable a separate name in each component (e.g. 'mass_SR' and 'mass_CR' instead of 'mass').");
      }
   }
   for (RooAbsData *subData : subDataSets) {
      exportData(*subData);
   }
   return info;
}

// Export a single dataset `data` (binned or unbinned) to the output JSON.
void RooJSONFactoryWSTool::exportData(RooAbsData const &data)
{
   for (RooAbsArg const *obs : *data.get()) {
      if (dynamic_cast<RooAbsCategory const *>(obs) && !dynamic_cast<RooCategory const *>(obs)) {
         error("the data set '" + std::string(data.GetName()) + "' has the categorical observable '" + obs->GetName() +
               "', which is not a RooCategory, and can't be written as a combined data set");
      }
   }

   JSONNode &output = appendNamedChild((*_rootnodeOutput)["data"], data.GetName());

   // This works around a problem in RooStats/HistFactory that was only fixed
   // in ROOT 6.30: until then, the weight variable of the observed dataset,
   // called "weightVar", was added to the observables. Therefore, it also got
   // added to the Asimov dataset. But the Asimov has its own weight variable,
   // called "binWeightAsimov", making "weightVar" an actual observable in the
   // Asimov data. But this is only by accident and should be removed.
   RooArgSet variables = *data.get();
   if (auto weightVar = variables.find("weightVar")) {
      variables.remove(*weightVar);
   }

   // this is a regular binned dataset
   // Categories are categorical axes, declared by a categorical axis of the default domain.
   auto readIntoDomain = [&](RooAbsArg const &var) {
      if (auto const *cat = dynamic_cast<RooCategory const *>(&var)) {
         _domains->readCategory(*cat);
      } else {
         _domains->readVariable(static_cast<RooRealVar const &>(var));
      }
   };

   if (auto dh = dynamic_cast<RooDataHist const *>(&data)) {
      output["type"] << "binned";
      for (RooAbsArg const *var : variables) {
         readIntoDomain(*var);
      }
      return exportHisto(variables, dh->numEntries(), dh->weightArray(), output);
   }

   // Check if this actually represents a binned dataset, and then import it
   // like a RooDataHist. This happens frequently when people create combined
   // RooDataSets from binned data to fit HistFactory models. In this case, it
   // doesn't make sense to export them like an unbinned dataset, because the
   // coordinates are redundant information with the binning. We only do this
   // for 1D data for now. A variable without bins can't be binned data: unbinned data sets leave their axes without
   // binning, and would otherwise be written as binned once they have no entries.
   if (data.isWeighted() && variables.size() == 1 && dynamic_cast<RooRealVar const *>(variables[0]) &&
       static_cast<RooRealVar const &>(*variables[0]).getBins() > 0) {
      bool isBinnedData = false;
      auto &x = static_cast<RooRealVar const &>(*variables[0]);
      std::vector<double> contents;
      int i = 0;
      for (; i < data.numEntries(); ++i) {
         data.get(i);
         if (x.getBin() != i)
            break;
         contents.push_back(data.weight());
      }
      if (i == x.getBins())
         isBinnedData = true;
      if (isBinnedData) {
         output["type"] << "binned";
         for (auto *var : static_range_cast<RooRealVar *>(variables)) {
            _domains->readVariable(*var);
         }
         return exportHisto(variables, data.numEntries(), contents.data(), output);
      }
   }

   // this really is an unbinned dataset
   output["type"] << "unbinned";
   auto &observablesNode = output["axes"].set_seq();
   for (RooAbsArg const *arg : variables) {
      readIntoDomain(*arg);
      if (auto const *cat = dynamic_cast<RooCategory const *>(arg)) {
         exportCategoricalAxis(observablesNode.append_child().set_map(), *cat);
         continue;
      }
      auto const *var = static_cast<RooRealVar const *>(arg);
      // HS3 forbids binning information on the axes of unbinned data.
      auto &axis = observablesNode.append_child().set_map();
      testValidName(var->GetName(), false);
      axis["name"] << var->GetName();
      axis["min"] << var->getMin();
      axis["max"] << var->getMax();
   }
   auto &coords = output["entries"].set_seq();
   std::vector<double> weightVals;
   bool hasNonUnityWeights = false;
   for (int i = 0; i < data.numEntries(); ++i) {
      data.get(i);
      // The entries of a categorical axis are the labels of the category.
      auto &row = coords.append_child().set_seq();
      for (RooAbsArg const *arg : variables) {
         if (auto const *cat = dynamic_cast<RooCategory const *>(arg)) {
            row.append_child() << cat->getCurrentLabel();
         } else {
            row.append_child() << static_cast<RooRealVar const *>(arg)->getVal();
         }
      }
      if (data.isWeighted()) {
         weightVals.push_back(data.weight());
         if (data.weight() != 1.)
            hasNonUnityWeights = true;
      }
   }
   if (data.isWeighted() && hasNonUnityWeights) {
      output["weights"].fill_seq(weightVals);
   }
}

/**
 * @brief Read axes from the JSONNode and create a RooArgSet representing them.
 *
 * This function reads axes information from the given JSONNode and
 * creates a RooArgSet with variables representing these axes.
 *
 * @param topNode The JSONNode containing the axes information to be read.
 * @return RooArgSet A RooArgSet containing the variables created from the JSONNode.
 */
RooArgSet RooJSONFactoryWSTool::readAxes(const JSONNode &topNode)
{
   RooArgSet vars;

   for (JSONNode const &node : topNode["axes"].children()) {
      if (node.has_child("categories")) {
         // One state per category, in the order of the document, with the position as index. The bins of a
         // RooDataHist then follow the order of "categories".
         const std::string axisName = node["name"].val();
         for (const char *key : {"min", "max", "nbins", "edges"}) {
            if (node.has_child(key)) {
               error("the axis '" + axisName + "' is categorical and must not have \"" + key + "\"");
            }
         }
         auto cat = std::make_unique<RooCategory>(axisName.c_str(), axisName.c_str());
         int index = 0;
         for (JSONNode const &label : node["categories"].children()) {
            if (label.val().empty() || cat->hasLabel(label.val())) {
               error("the categories of the axis '" + axisName + "' must be unique non-empty strings");
            }
            cat->defineType(label.val(), index++);
         }
         if (index == 0) {
            error("the categorical axis '" + axisName + "' has no categories");
         }
         vars.addOwned(std::move(cat));
      } else if (node.has_child("edges")) {
         std::vector<double> edges = readEdges(node["edges"], "axis '" + node["name"].val() + "'");
         auto obs = std::make_unique<RooRealVar>(node["name"].val().c_str(), node["name"].val().c_str(), edges[0],
                                                 edges[edges.size() - 1]);
         RooBinning bins(obs->getMin(), obs->getMax());
         for (auto b : edges) {
            bins.addBoundary(b);
         }
         obs->setBinning(bins);
         vars.addOwned(std::move(obs));
      } else {
         auto obs = std::make_unique<RooRealVar>(node["name"].val().c_str(), node["name"].val().c_str(),
                                                 node["min"].val_double(), node["max"].val_double());
         obs->setBins(node["nbins"].val_int());
         vars.addOwned(std::move(obs));
      }
   }

   return vars;
}

namespace {

// For binned data in `n` with categorical axes, the document lays out the bins in the order of its "categories" (last
// axis fastest), while RooFit lays them out in the order of the states of the categories in `vars`. Returns, for each
// bin in RooFit's order, the position of that bin in the document. Empty if the orders agree.
std::vector<std::size_t> binOrderOfDocument(const JSONNode &n, RooArgSet const &vars)
{
   JSONNode const *axes = n.find("axes");
   if (!axes || static_cast<std::size_t>(axes->num_children()) != vars.size()) {
      return {};
   }
   std::vector<std::size_t> nBins;
   // For each axis, the position in the document of the bin with a given position in RooFit's order.
   std::vector<std::vector<std::size_t>> axisPositions;
   bool reordered = false;
   std::size_t iAxis = 0;
   for (JSONNode const &axis : axes->children()) {
      RooAbsArg const *var = vars[iAxis++];
      auto const *cat = dynamic_cast<RooAbsCategory const *>(var);
      const auto numBins = static_cast<std::size_t>(dynamic_cast<RooAbsLValue const &>(*var).numBins());
      nBins.push_back(numBins);
      std::vector<std::size_t> positions(numBins);
      for (std::size_t i = 0; i < numBins; ++i) {
         positions[i] = i;
      }
      if (cat && axis.has_child("categories")) {
         std::vector<std::string> labels;
         for (JSONNode const &label : axis["categories"].children()) {
            labels.push_back(label.val());
         }
         for (std::size_t i = 0; i < numBins; ++i) {
            auto found = std::find(labels.begin(), labels.end(), cat->getOrdinal(i).first);
            positions[i] = std::distance(labels.begin(), found);
            reordered |= positions[i] != i;
         }
      }
      axisPositions.push_back(std::move(positions));
   }
   if (!reordered) {
      return {};
   }
   std::size_t total = 1;
   for (std::size_t nb : nBins) {
      total *= nb;
   }
   std::vector<std::size_t> order(total);
   for (std::size_t k = 0; k < total; ++k) {
      // Decompose RooFit's bin index (last axis fastest) and recompose it with the document's positions.
      std::size_t rest = k;
      std::size_t docIndex = 0;
      std::size_t stride = 1;
      for (std::size_t j = nBins.size(); j-- > 0;) {
         const std::size_t i = rest % nBins[j];
         rest /= nBins[j];
         docIndex += axisPositions[j][i] * stride;
         stride *= nBins[j];
      }
      order[k] = docIndex;
   }
   return order;
}

std::vector<double> permute(std::vector<double> const &values, std::vector<std::size_t> const &order)
{
   if (order.empty() || values.size() != order.size()) {
      return values;
   }
   std::vector<double> out(values.size());
   for (std::size_t k = 0; k < order.size(); ++k) {
      out[k] = values[order[k]];
   }
   return out;
}

} // namespace

/**
 * @brief Read binned data from the JSONNode and create a RooDataHist object.
 *
 * This function reads binned data from the given JSONNode and creates a RooDataHist object.
 * The binned data is associated with the specified name and variables (RooArgSet) in the workspace.
 *
 * @param n The JSONNode representing the binned data to be read.
 * @param name The name to be associated with the created RooDataHist object.
 * @param vars The RooArgSet representing the variables associated with the binned data.
 * @return std::unique_ptr<RooDataHist> A unique pointer to the created RooDataHist object.
 */
std::unique_ptr<RooDataHist>
RooJSONFactoryWSTool::readBinnedData(const JSONNode &n, const std::string &name, RooArgSet const &vars)
{
   if (!n.has_child("contents"))
      RooJSONFactoryWSTool::error("no contents given");

   JSONNode const &contents = n["contents"];

   if (!contents.is_seq())
      RooJSONFactoryWSTool::error("contents are not in list form");

   JSONNode const *errors = nullptr;
   if (n.has_child("errors")) {
      errors = &n["errors"];
      if (!errors->is_seq())
         RooJSONFactoryWSTool::error("errors are not in list form");
      if (errors->num_children() != contents.num_children()) {
         RooJSONFactoryWSTool::error("inconsistent number of errors and contents in '" + name +
                                     "': errors=" + std::to_string(errors->num_children()) +
                                     ", contents=" + std::to_string(contents.num_children()));
      }
   }

   auto bins = generateBinIndices(vars);
   if (contents.num_children() != bins.size()) {
      RooJSONFactoryWSTool::error("inconsistent bin numbers: contents=" + std::to_string(contents.num_children()) +
                                  ", bins=" + std::to_string(bins.size()));
   }
   auto dh = std::make_unique<RooDataHist>(name, name, vars);
   const std::vector<std::size_t> binOrder = binOrderOfDocument(n, vars);
   std::vector<double> contentVals;
   contentVals.reserve(contents.num_children());
   for (auto const &cont : contents.children()) {
      contentVals.push_back(cont.val_double());
   }
   contentVals = permute(contentVals, binOrder);
   std::vector<double> errorVals;
   if (errors) {
      errorVals.reserve(errors->num_children());
      for (auto const &err : errors->children()) {
         errorVals.push_back(err.val_double());
      }
      errorVals = permute(errorVals, binOrder);
   }
   // RooDataHist only starts to store squared weights when it sees a positive error, and then initializes them from
   // the weights so far. Trigger that before setting any weight, so that errors of zero stay zero.
   if (!errorVals.empty()) {
      dh->set(0, 0., 1.);
   }
   for (size_t ibin = 0; ibin < bins.size(); ++ibin) {
      const double err = errors ? errorVals[ibin] : -1;
      dh->set(ibin, contentVals[ibin], err);
   }
   return dh;
}

// Import a single variable (RooRealVar or RooConstVar) from the JSON node `p` into the workspace.
void RooJSONFactoryWSTool::importVariable(const JSONNode &p)
{
   // import a RooRealVar object
   std::string name(RooJSONFactoryWSTool::name(p));
   RooJSONFactoryWSTool::testValidName(name, true);

   if (_workspace.arg(name))
      return;
   if (!p.is_map()) {
      oocoutE(nullptr, InputArguments) << "RooJSONFactoryWSTool() node '" << name << "' is not a map, skipping."
                                       << std::endl;
      return;
   }
   if (config().importNoDomainParametersAsRooConstVars && !_domains->hasVariable(name.c_str())) {
      if (!p.has_child("value")) {
         RooJSONFactoryWSTool::error("cannot instantiate RooConstVar '" + name + "' without \"value\"!");
      }
      wsEmplace<RooConstVar>(name, p["value"].val_double());
      return;
   }
   configureVariable(*_domains, p, wsEmplace<RooRealVar>(name, 1.));
}

// Import all dependants (variables, functions and distributions) of node `n` into the workspace.
void RooJSONFactoryWSTool::importDependants(const JSONNode &n)
{
   // import all the dependants of an object
   if (JSONNode const *varsNode = getVariablesNode(n)) {
      for (const auto &p : varsNode->children()) {
         importVariable(p);
      }
   }
   if (auto seq = n.find("functions")) {
      for (const auto &p : seq->children()) {
         this->importFunction(p, true);
      }
   }
   if (auto seq = n.find("distributions")) {
      for (const auto &p : seq->children()) {
         this->importFunction(p, true);
      }
   }
}

void RooJSONFactoryWSTool::exportModelConfig(JSONNode &rootnode, RooFit::ModelConfig const &mc,
                                             const std::vector<CombinedData> &combDataSets,
                                             const std::vector<RooAbsData *> &singleDataSets)
{
   // The analyses are named after the ModelConfig, so that ModelConfigs sharing a pdf and a data set (like an S+B
   // and B-only pair) get separate analyses.
   auto pdf = mc.GetPdf();
   auto simpdf = dynamic_cast<RooSimultaneous const *>(pdf);
   int founddata = 0;
   if (simpdf && !simpdf->getAttribute(categoryConditionalTag)) {
      // One analysis per combined data set with the same columns.
      std::vector<std::string> columns;
      for (RooCategory const *column : selectionColumns(simpdf->indexCat())) {
         columns.push_back(column->GetName());
      }
      for (auto const &data : combDataSets) {
         if (data.columns == columns) {
            exportSingleModelConfig(rootnode, mc, std::string(mc.GetName()) + "_" + data.name, &data.name);
            ++founddata;
         }
      }
   } else {
      RooArgSet observables(*mc.GetObservables());
      for (auto *data : singleDataSets) {
         if (observables.equals(*(data->get()))) {
            const std::string dataName = data->GetName();
            exportSingleModelConfig(rootnode, mc, std::string(mc.GetName()) + "_" + dataName, &dataName);
            ++founddata;
         }
      }
   }
   if (founddata == 0) {
      exportSingleModelConfig(rootnode, mc, mc.GetName(), nullptr);
   }
}

void RooJSONFactoryWSTool::exportSingleModelConfig(JSONNode &rootnode, RooFit::ModelConfig const &mc,
                                                   std::string const &analysisName, std::string const *dataName)
{
   auto pdf = mc.GetPdf();

   JSONNode &analysisNode = appendNamedChild(rootnode["analyses"], analysisName);

   auto &domains = analysisNode["domains"].set_seq();

   // HS3 names must be unique across all top-level components, so the likelihood can't have the analysis name.
   const std::string likelihoodName = analysisName + "_likelihood";
   analysisNode["likelihood"] << likelihoodName;

   auto &nllNode = appendNamedChild(rootnode["likelihoods"], likelihoodName);
   nllNode["distributions"].set_seq();
   nllNode["data"].set_seq();

   nllNode["distributions"].append_child() << pdf->GetName();
   if (dataName) {
      nllNode["data"].append_child() << *dataName;
   } else {
      nllNode["data"].append_child() << 0;
   }

   if (mc.GetExternalConstraints()) {
      auto &extConstrNode = nllNode["aux_distributions"];
      extConstrNode.set_seq();
      for (const auto &constr : *mc.GetExternalConstraints()) {
         extConstrNode.append_child() << constr->GetName();
      }
   }

   auto writeList = [&](const char *name, RooArgSet const *args) {
      if (!args || !args->size())
         return;

      std::vector<std::string> names;
      names.reserve(args->size());
      for (RooAbsArg const *arg : *args)
         names.push_back(arg->GetName());
      std::sort(names.begin(), names.end());
      analysisNode[name].fill_seq(names);
   };

   writeList("parameters_of_interest", mc.GetParametersOfInterest());

   auto &domainsNode = rootnode["domains"];

   auto writeProductDomain = [&](const char *suffix, RooArgSet const *args) {
      if (!args || args->empty())
         return;
      const std::string domainName = analysisName + suffix;
      domains.append_child() << domainName;
      RooFit::JSONIO::Detail::Domains::ProductDomain domain;
      for (auto *var : static_range_cast<const RooRealVar *>(*args)) {
         domain.readVariable(*var);
      }
      domain.writeJSON(appendNamedChild(domainsNode, domainName));
   };

   writeProductDomain("_nuisance_parameters", mc.GetNuisanceParameters());
   writeProductDomain("_global_observables", mc.GetGlobalObservables());
   writeProductDomain("_parameters_of_interest", mc.GetParametersOfInterest());

   // The snapshots are exported as parameter points. Reference the one of the ModelConfig as the initial values.
   if (_workspace.getSnapshot(mc.GetSnapshotName().c_str())) {
      analysisNode["init"] << mc.GetSnapshotName();
   }

   auto &modelConfigAux = getRooFitInternal(rootnode, "ModelConfigs", analysisName);
   modelConfigAux.set_map();
   modelConfigAux["pdfName"] << pdf->GetName();
   modelConfigAux["mcName"] << mc.GetName();
}

// Export all top-level pdfs, functions, datasets and ModelConfigs of the workspace into `n`.
void RooJSONFactoryWSTool::exportAllObjects(JSONNode &n)
{
   _domains = std::make_unique<RooFit::JSONIO::Detail::Domains>();
   _varsNode = &makeVariablesNode(n);
   _rootnodeOutput = &n;
   _columnCategories.clear();
   prepareCombinedObservableRenaming();

   // export all toplevel pdfs
   std::vector<RooAbsPdf *> allpdfs;
   for (auto &arg : _workspace.allPdfs()) {
      if (isTopLevel(*arg, _workspace)) {
         if (auto *pdf = dynamic_cast<RooAbsPdf *>(arg)) {
            allpdfs.push_back(pdf);
         }
      }
   }
   sortByName(allpdfs);
   std::set<std::string> exportedObjectNames;
   exportObjects(allpdfs, exportedObjectNames);

   // export all toplevel functions
   std::vector<RooAbsReal *> allfuncs;
   for (auto &arg : _workspace.allFunctions()) {
      if (isTopLevel(*arg, _workspace)) {
         if (auto *func = dynamic_cast<RooAbsReal *>(arg)) {
            allfuncs.push_back(func);
         }
      }
   }
   sortByName(allfuncs);
   exportObjects(allfuncs, exportedObjectNames);

   // export attributes of all objects
   for (RooAbsArg *arg : _workspace.components()) {
      exportAttributes(arg, n);
   }

   // collect all datasets
   std::vector<RooAbsData *> alldata;
   for (auto &d : _workspace.allData()) {
      alldata.push_back(d);
   }
   sortByName(alldata);
   // Data sets with category observables are combined data sets.
   std::vector<RooAbsData *> singleData;
   std::vector<RooJSONFactoryWSTool::CombinedData> combData;
   for (auto &d : alldata) {
      // Without a matching RooSimultaneous, the categories of a data set are ordinary observables, written as
      // categorical axes, unless its observables are renamed per key (Config::renameCombinedObservables).
      if (selectionColumns(*d->get()).empty() ||
          (!config().renameCombinedObservables &&
           !findMatchingSimultaneous(_workspace, selectionColumnNames(*d->get())))) {
         singleData.push_back(d);
         this->exportData(*d);
      } else {
         combData.push_back(this->exportCombinedData(*d));
      }
   }

   // export all ModelConfig objects and attached Pdfs
   for (TObject *obj : _workspace.allGenericObjects()) {
      if (auto mc = dynamic_cast<RooFit::ModelConfig *>(obj)) {
         exportModelConfig(n, *mc, combData, singleData);
      }
   }

   exportParameterStepWidths(_workspace, allpdfs, alldata, n);

   for (auto *snsh : static_range_cast<RooArgSet const *>(_workspace.getSnapshots())) {
      RooArgSet snapshotSorted;
      // We only want to add the variables that actually got exported and skip
      // the ones that the pdfs encoded implicitly (like in the case of
      // HistFactory).
      for (RooAbsArg *arg : *snsh) {
         bool do_export = false;
         for (const auto &pdf : allpdfs) {
            if (pdf->dependsOn(*arg)) {
               do_export = true;
            }
         }
         if (do_export) {
            RooJSONFactoryWSTool::testValidName(arg->GetName(), true);
            snapshotSorted.add(*arg);
         }
      }
      snapshotSorted.sort();
      std::string name(snsh->GetName());
      if (name != "default_values") {
         this->exportVariables(snapshotSorted, appendNamedChild(n["parameter_points"], name)["parameters"], true,
                               false);
      }
   }
   _varsNode = nullptr;
   _domains->writeJSON(n["domains"]);
   _domains.reset();
   _rootnodeOutput = nullptr;
}

/**
 * @brief Import the workspace from a JSON string.
 *
 * @param s The JSON string containing the workspace data.
 * @return bool Returns true on successful import, false otherwise.
 */
bool RooJSONFactoryWSTool::importJSONfromString(const std::string &s)
{
   std::stringstream ss(s);
   return importJSON(ss);
}

/**
 * @brief Export the workspace to a JSON string.
 *
 * @return std::string The JSON string representing the exported workspace.
 */
std::string RooJSONFactoryWSTool::exportJSONtoString()
{
   std::stringstream ss;
   exportJSON(ss);
   return ss.str();
}

/**
 * @brief Create a new JSON tree with version information.
 *
 * @return std::unique_ptr<JSONTree> A unique pointer to the created JSON tree.
 */
std::unique_ptr<JSONTree> RooJSONFactoryWSTool::createNewJSONTree()
{
   std::unique_ptr<JSONTree> tree = JSONTree::create();
   JSONNode &n = tree->rootnode();
   n.set_map();
   auto &metadata = n["metadata"].set_map();

   // add the mandatory hs3 version number
   metadata["hs3_version"] << hs3VersionTag;

   // Add information about the ROOT version that was used to generate this file
   auto &rootInfo = appendNamedChild(metadata["packages"], "ROOT");
   std::string versionName = gROOT->GetVersion();
   // We want to consistently use dots such that the version name can be easily
   // digested automatically.
   std::replace(versionName.begin(), versionName.end(), '/', '.');
   rootInfo["version"] << versionName;

   return tree;
}

/**
 * @brief Export the workspace to JSON format and write to the output stream.
 *
 * @param os The output stream to write the JSON data to.
 * @return bool Returns true on successful export, false otherwise.
 */
bool RooJSONFactoryWSTool::exportJSON(std::ostream &os)
{
   std::unique_ptr<JSONTree> tree = createNewJSONTree();
   JSONNode &n = tree->rootnode();
   this->exportAllObjects(n);
   n.writeJSON(os);
   return true;
}

/**
 * @brief Export the workspace to JSON format and write to the specified file.
 *
 * @param filename The name of the JSON file to create and write the data to.
 * @return bool Returns true on successful export, false otherwise.
 */
bool RooJSONFactoryWSTool::exportJSON(std::string const &filename)
{
   std::ofstream out(filename.c_str());
   if (!out.is_open())
      RooJSONFactoryWSTool::error("RooJSONFactoryWSTool() invalid output file '" + filename + "'.");
   return this->exportJSON(out);
}

bool RooJSONFactoryWSTool::hasAttribute(const std::string &obj, const std::string &attrib)
{
   if (!_attributesNode)
      return false;
   if (auto attrNode = _attributesNode->find(obj)) {
      if (auto seq = attrNode->find("tags")) {
         for (auto &a : seq->children()) {
            if (a.val() == attrib)
               return true;
         }
      }
   }
   return false;
}
void RooJSONFactoryWSTool::setAttribute(const std::string &obj, const std::string &attrib)
{
   auto node = &RooJSONFactoryWSTool::getRooFitInternal(*_rootnodeOutput, "attributes").set_map()[obj].set_map();
   auto &tags = (*node)["tags"];
   tags.set_seq();
   tags.append_child() << attrib;
}

std::string RooJSONFactoryWSTool::getStringAttribute(const std::string &obj, const std::string &attrib)
{
   if (!_attributesNode)
      return "";
   if (auto attrNode = _attributesNode->find(obj)) {
      if (auto dict = attrNode->find("dict")) {
         if (auto *a = dict->find(attrib)) {
            return a->val();
         }
      }
   }
   return "";
}
void RooJSONFactoryWSTool::setStringAttribute(const std::string &obj, const std::string &attrib,
                                              const std::string &value)
{
   auto node = &RooJSONFactoryWSTool::getRooFitInternal(*_rootnodeOutput, "attributes").set_map()[obj].set_map();
   auto &dict = (*node)["dict"];
   dict.set_map();
   dict[attrib] << value;
}

// Import all nodes of the JSON document rooted at `n` into the workspace.
void RooJSONFactoryWSTool::importAllNodes(const JSONNode &n)
{
   // Per HS3 standard, the hs3_version in the metadata is required. So we
   // error out if it is missing or not a string, and warn if the document is
   // newer than what this version of RooFit was written for.
   auto metadata = n.find("metadata");
   auto versionNode = metadata ? metadata->find("hs3_version") : nullptr;
   if (!versionNode) {
      std::stringstream ss;
      ss << "The HS3 version is missing in the JSON!\n"
         << "Please include the HS3 version in the metadata field, e.g.:\n"
         << "    \"metadata\" :\n"
         << "    {\n"
         << "        \"hs3_version\" : \"" << hs3VersionTag << "\"\n"
         << "    }";
      error(ss.str());
   }
   if (versionNode->is_container() || versionNode->is_null() || versionNode->is_number()) {
      error(std::string("The HS3 version in the metadata must be a string, e.g. \"") + hs3VersionTag + "\"");
   }
   _hs3VersionInput = versionNode->val();
   checkHS3Version(_hs3VersionInput);
   checkUniqueNames(n, !isHS3VersionBefore(_hs3VersionInput, 0, 3));

   _rootnodeInput = &n;

   _attributesNode = findRooFitInternal(*_rootnodeInput, "attributes");

   _domains = std::make_unique<RooFit::JSONIO::Detail::Domains>();
   if (auto domains = n.find("domains")) {
      _domains->readJSON(*domains);
   }
   _domains->populate(_workspace);

   // Build name-keyed indices over the "functions" and "distributions"
   // sequences. Without these, every cross-reference resolved during import
   // (e.g. dependencies of a PiecewiseInterpolation, or factory-expression
   // arguments) triggers a linear scan over all sibling nodes via
   // findNamedChild(), which becomes O(N^2) on workspaces with thousands of
   // entries. Populating the maps up-front turns each lookup into O(1).
   auto buildIndex = [&n](const char *key, auto &index) {
      index.clear();
      auto seq = n.find(key);
      if (!seq || !seq->is_seq())
         return;
      index.reserve(seq->num_children());
      for (const auto &p : seq->children()) {
         index.emplace(RooJSONFactoryWSTool::name(p), &p);
      }
   };
   buildIndex("functions", _functionsByName);
   buildIndex("distributions", _distributionsByName);

   this->importDependants(n);

   // This creates the index categories, which the parameter points can refer to.
   combinePdfs(*_rootnodeInput, _workspace);

   if (auto paramPointsNode = n.find("parameter_points")) {
      for (const auto &snsh : paramPointsNode->children()) {
         std::string name = RooJSONFactoryWSTool::name(snsh);
         RooJSONFactoryWSTool::testValidName(name, true);

         RooArgSet vars;
         for (const auto &var : snsh["parameters"].children()) {
            const std::string varName = RooJSONFactoryWSTool::name(var);
            JSONNode const *valueNode = var.find("value");
            const bool hasStringValue = valueNode && !valueNode->is_number();
            if (RooCategory *cat = _workspace.cat(varName)) {
               if (hasStringValue) {
                  cat->setLabel(valueNode->val().c_str());
               } else if (valueNode) {
                  cat->setIndex(valueNode->val_int());
               }
               vars.add(*cat);
               continue;
            }
            if (hasStringValue) {
               warning("RooFitHS3: ignoring the string value of parameter '" + varName + "' in parameter point '" +
                       name + "', because there is no category with that name.");
               continue;
            }
            RooRealVar *rrv = _workspace.var(varName);
            if (!rrv && !_workspace.arg(varName)) {
               // The parameter is not used by any other component: create it. It is constant if "const" is true or
               // if no domain lists it, since there is no range to vary it in. "const": false is the same as no
               // "const" at all.
               testValidName(varName, true);
               rrv = &wsEmplace<RooRealVar>(varName, 1.);
               configureVariable(*_domains, var, *rrv);
               rrv->setConstant(rrv->isConstant() || !_domains->hasVariable(varName.c_str()));
            } else if (rrv) {
               configureVariable(*_domains, var, *rrv);
            }
            if (rrv) {
               vars.add(*rrv);
            }
         }
         _workspace.saveSnapshot(name, vars);
      }
   }

   // Import attributes
   if (_attributesNode) {
      for (const auto &elem : _attributesNode->children()) {
         if (RooAbsArg *arg = _workspace.arg(elem.key()))
            importAttributes(arg, elem);
      }
   }

   _attributesNode = nullptr;

   // The components of combined data sets are moved out of the "datasets" vector when the combined data set is
   // assembled, so they don't get imported on their own. The ROOT-internal form of older files (combineDatasets())
   // is handled after the analyses, because their likelihoods refer to the components.
   std::vector<std::unique_ptr<RooAbsData>> datasets;
   std::map<std::string, CombinedDataInfo> combinedData;
   if (auto dataNode = n.find("data")) {
      for (const auto &p : dataNode->children()) {
         // Combined data sets are assembled from the others once all of them are loaded.
         if (p.is_map() && p.has_child("type") && p["type"].val() == "combined") {
            continue;
         }
         datasets.push_back(loadData(p, _workspace, _domains.get()));
      }
      combinedData = assembleCombinedData(*dataNode, _workspace, datasets);
   }

   // Now, read in analyses and likelihoods if there are any

   if (auto analysesNode = n.find("analyses")) {
      for (JSONNode const &analysisNode : analysesNode->children()) {
         importAnalysis(*_rootnodeInput, analysisNode, n["likelihoods"], n["domains"], _workspace, datasets,
                        combinedData);
      }
   }

   combineDatasets(*_rootnodeInput, datasets);

   for (auto const &d : datasets) {
      if (d) {
         _workspace.import(*d);
         for (auto const &obs : *d->get()) {
            if (auto *rrv = dynamic_cast<RooRealVar *>(obs)) {
               _workspace.var(rrv->GetName())->setBinning(rrv->getBinning());
            }
         }
      }
   }

   _rootnodeInput = nullptr;
   _hs3VersionInput.clear();
   _domains.reset();
   _functionsByName.clear();
   _distributionsByName.clear();
}

/**
 * @brief Imports a JSON file from the given input stream to the workspace.
 *
 * @param is The input stream containing the JSON data.
 * @return bool Returns true on successful import, false otherwise.
 */
bool RooJSONFactoryWSTool::importJSON(std::istream &is)
{
   // import a JSON file to the workspace
   std::unique_ptr<JSONTree> tree = JSONTree::create(is);
   JSONNode const &rootnode = tree->rootnode();
   this->importAllNodes(rootnode);
   if (this->workspace()->getSnapshot("default_values")) {
      this->workspace()->loadSnapshot("default_values");
   }
   importParameterStepWidths(*this->workspace(), rootnode);
   return true;
}

/**
 * @brief Imports a JSON file from the given filename to the workspace.
 *
 * @param filename The name of the JSON file to import.
 * @return bool Returns true on successful import, false otherwise.
 */
bool RooJSONFactoryWSTool::importJSON(std::string const &filename)
{
   // import a JSON file to the workspace
   std::ifstream infile(filename.c_str());
   if (!infile.is_open())
      RooJSONFactoryWSTool::error("RooJSONFactoryWSTool() invalid input file '" + filename + "'.");
   return this->importJSON(infile);
}

void RooJSONFactoryWSTool::importJSONElement(const std::string &name, const std::string &jsonString)
{
   // Create the JSON Tree from the string
   std::unique_ptr<RooFit::Detail::JSONTree> tree = RooFit::Detail::JSONTree::create(jsonString);
   JSONNode &n = tree->rootnode();

   // If the objects containts a parameter of interest, import it as a modelConfig
   if (n.find("poi")) {

      RooStats::ModelConfig modelConfig{"ModelConfig"};
      std::string poi = n.find("poi")->val();
      std::string pdname = n.find("pdfName")->val();
      modelConfig.SetWS(_workspace);
      modelConfig.SetPdf(pdname.c_str());
      modelConfig.SetParametersOfInterest(_workspace.argSet(poi));
      _workspace.import(modelConfig);

      return;
   }

   n["name"] << name;

   bool isVariable = true;
   bool isData = false;
   // Check for the type of object, if it doesn't contain a type, it must be a variable
   if (n.find("type")) {
      isVariable = false;
      std::string elementType = n.find("type")->val();
      if (elementType == "binned" || elementType == "unbinned") {
         isData = true;
      }
   }

   // Import the object to the workspace
   if (isVariable) {
      this->importVariableElement(n);
   } else if (isData) {
      auto absData = loadData(n, _workspace, nullptr);
      _workspace.import(*absData);
   } else {
      this->importFunction(n, false);
   }
}

void RooJSONFactoryWSTool::importVariableElement(const JSONNode &elementNode)
{
   std::unique_ptr<RooFit::Detail::JSONTree> tree = varJSONString(elementNode);
   JSONNode &n = tree->rootnode();
   _domains = std::make_unique<RooFit::JSONIO::Detail::Domains>();
   if (auto domains = n.find("domains"))
      _domains->readJSON(*domains);

   _rootnodeInput = &n;
   _attributesNode = findRooFitInternal(*_rootnodeInput, "attributes");

   JSONNode const *varsNode = getVariablesNode(n);
   const auto &p = varsNode->child(0);
   importVariable(p);

   auto paramPointsNode = n.find("parameter_points");
   const auto &var = paramPointsNode->child(0)["parameters"].child(0);
   if (RooRealVar *rrv = _workspace.var(RooJSONFactoryWSTool::name(var))) {
      configureVariable(*_domains, var, *rrv);
   }

   // Import attributes
   if (_attributesNode) {
      for (const auto &elem : _attributesNode->children()) {
         if (RooAbsArg *arg = _workspace.arg(elem.key()))
            importAttributes(arg, elem);
      }
   }

   _attributesNode = nullptr;
   _rootnodeInput = nullptr;
   _domains.reset();
}

/**
 * @brief Writes a warning message to the RooFit message service.
 *
 * @param str The warning message to be logged.
 * @return std::ostream& A reference to the output stream.
 */
std::ostream &RooJSONFactoryWSTool::warning(std::string const &str)
{
   return RooMsgService::instance().log(nullptr, RooFit::MsgLevel::WARNING, RooFit::ObjectHandling) << str << std::endl;
}

/**
 * @brief Writes an error message to the RooFit message service and throws a runtime_error.
 *
 * @param s The error message to be logged and thrown.
 * @return void
 */
void RooJSONFactoryWSTool::error(const char *s)
{
   RooMsgService::instance().log(nullptr, RooFit::MsgLevel::ERROR, RooFit::IO) << s << std::endl;
   throw std::runtime_error(s);
}

/**
 * @brief Cleans up names to the HS3 standard
 *
 * @param str The string to be sanitized.
 * @return std::string
 */
std::string RooJSONFactoryWSTool::sanitizeName(const std::string str)
{
   std::string result;
   if (RooJSONFactoryWSTool::config().allowSanitizeNames) {
      for (char c : str) {
         switch (c) {
         case '[':
         case '|':
         case ',':
         case '(': result += '_'; break;
         case ']':
         case ')':
            // skip these characters entirely
            break;
         case '.': result += "_dot_"; break;
         case '@': result += "at"; break;
         case '-': result += "minus"; break;
         case '/': result += "_div_"; break;

         default: result += c; break;
         }
      }
      return result;
   }
   return str;
}

RooWorkspace RooJSONFactoryWSTool::cleanWS(const RooWorkspace &ws, bool onlyModelConfig)
{
   // Variables

   RooWorkspace tmpWS = RooWorkspace();
   if (onlyModelConfig) {
      for (auto *obj : ws.allGenericObjects()) {
         if (auto *mc = dynamic_cast<RooFit::ModelConfig *>(obj)) {
            tmpWS.import(*mc->GetPdf(), RooFit::RecycleConflictNodes(true));
         }
      }

   } else {

      for (auto *pdf : ws.allPdfs()) {
         if (isTopLevel(*pdf, ws)) {
            tmpWS.import(*pdf, RooFit::RecycleConflictNodes(true));
         }
      }

      for (auto *func : ws.allFunctions()) {
         if (isTopLevel(*func, ws)) {
            tmpWS.import(*func, RooFit::RecycleConflictNodes(true));
         }
      }
   }

   for (auto *data : ws.allData()) {
      tmpWS.import(*data);
   }

   for (auto *obj : ws.allGenericObjects()) {
      tmpWS.import(*obj);
   }

   for (auto *obj : ws.allResolutionModels()) {
      tmpWS.import(*obj);
   }

   for (auto *snsh : ws.getSnapshots()) {
      auto *snshSet = dynamic_cast<RooArgSet *>(snsh);
      if (snshSet) {
         tmpWS.saveSnapshot(snshSet->GetName(), *snshSet, true);
      }
   }

   return tmpWS;
}

RooJSONFactoryWSTool::Config &RooJSONFactoryWSTool::config()
{
   static Config conf;
   return conf;
}

// Sanitize all names in the workspace to be HS3 compliant
RooWorkspace RooJSONFactoryWSTool::sanitizeWS(const RooWorkspace &ws)
{
   // Variables

   RooWorkspace tmpWS = cleanWS(ws, false);

   auto sanitizeIfNeeded = [](auto const &list) {
      for (auto *obj : list) {
         if (!isValidName(obj->GetName())) {
            obj->SetName(sanitizeName(obj->GetName()).c_str());
         }
      }
   };
   sanitizeIfNeeded(tmpWS.allVars());
   sanitizeIfNeeded(tmpWS.allFunctions());
   sanitizeIfNeeded(tmpWS.allPdfs());
   sanitizeIfNeeded(tmpWS.allResolutionModels());
   // Datasets
   for (auto *data : tmpWS.allData()) {
      // Sanitize dataset name
      if (!isValidName(data->GetName())) {
         data->SetName(sanitizeName(data->GetName()).c_str());
      }
      for (auto *obj : *data->get()) {
         obj->SetName(sanitizeName(obj->GetName()).c_str());
      }
   }
   for (auto *data : tmpWS.allEmbeddedData()) {
      // Sanitize dataset name
      data->SetName(sanitizeName(data->GetName()).c_str());
      for (auto *obj : *data->get()) {
         obj->SetName(sanitizeName(obj->GetName()).c_str());
      }
   }
   for (auto *snshObj : tmpWS.getSnapshots()) {
      // Snapshots are stored as TObject*, but really they are RooArgSet*
      auto *snsh = dynamic_cast<RooArgSet *>(snshObj);
      if (!snsh) {
         std::cerr << "Warning: found snapshot that is not a RooArgSet, skipping\n";
         continue;
      }

      // Sanitize snapshot name
      if (!isValidName(snsh->GetName())) {
         snsh->setName(sanitizeName(snsh->GetName()).c_str());
      }

      // Sanitize the variables inside the snapshot
      for (auto *arg : *snsh) {
         if (!isValidName(arg->GetName())) {
            arg->SetName(sanitizeName(arg->GetName()).c_str());
         }
      }
   }

   // Generic objects (ModelConfigs, attributes, etc.)
   for (auto *obj : tmpWS.allGenericObjects()) {
      if (!isValidName(obj->GetName())) {
         if (auto *named = dynamic_cast<TNamed *>(obj)) {
            named->SetName(sanitizeName(named->GetName()).c_str());
         } else {
            std::cerr << "Warning: object " << obj->GetName() << " is not TNamed, cannot rename.\n";
         }
      }

      if (auto *mc = dynamic_cast<RooFit::ModelConfig *>(obj)) {
         // Sanitize ModelConfig name
         if (!isValidName(mc->GetName())) {
            mc->SetName(sanitizeName(mc->GetName()).c_str());
         }

         // Sanitize the sets inside ModelConfig
         for (auto *obs : mc->GetObservables()->get()) {
            if (obs) {
               obs->SetName(sanitizeName(obs->GetName()).c_str());
            }
         }
         for (auto *poi : mc->GetParametersOfInterest()->get()) {
            if (poi) {
               poi->SetName(sanitizeName(poi->GetName()).c_str());
            }
         }
         for (auto *nuis : mc->GetNuisanceParameters()->get()) {
            if (nuis) {
               nuis->SetName(sanitizeName(nuis->GetName()).c_str());
            }
         }
         for (auto *glob : mc->GetGlobalObservables()->get()) {
            if (glob) {
               glob->SetName(sanitizeName(glob->GetName()).c_str());
            }
         }
      }
   }
   std::string wsName = std::string{ws.GetName()} + "_sanitized";
   RooWorkspace newWS = cleanWS(tmpWS, false);
   newWS.SetName(wsName.c_str());

   return newWS;
}
