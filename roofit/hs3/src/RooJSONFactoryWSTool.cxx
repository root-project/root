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
#include <RooRealVar.h>
#include <RooBinning.h>
#include <RooAbsCategory.h>
#include <RooRealProxy.h>
#include <RooListProxy.h>
#include <RooAbsProxy.h>
#include <RooCategory.h>
#include <RooDataSet.h>
#include <RooDataHist.h>
#include <RooSimultaneous.h>
#include <RooFormulaVar.h>
#include <RooFit/ModelConfig.h>

#include "JSONIOUtils.h"
#include "Domains.h"

#include "RooFitImplHelpers.h"

#include <TROOT.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <stack>
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

constexpr auto hs3VersionTag = "0.2";

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

/**
 * @brief Check if the number of components in CombinedData matches the number of categories in the RooSimultaneous PDF.
 *
 * This function checks whether the number of components in the provided CombinedData 'data' matches the number of
 * categories in the provided RooSimultaneous PDF 'pdf'.
 *
 * @param data The reference to the CombinedData to be checked.
 * @param pdf The pointer to the RooSimultaneous PDF for comparison.
 * @return bool Returns true if the number of components in 'data' matches the number of categories in 'pdf'; otherwise,
 * returns false.
 */
bool matches(const RooJSONFactoryWSTool::CombinedData &data, const RooSimultaneous *pdf)
{
   return data.components.size() == pdf->indexCat().size();
}

/**
 * @struct Var
 * @brief Structure to store variable information.
 *
 * This structure represents variable information such as the number of bins, minimum and maximum values,
 * and a vector of binning edges for a variable.
 */
struct Var {
   int nbins;                 // Number of bins
   double min;                // Minimum value
   double max;                // Maximum value
   std::vector<double> edges; // Vector of edges

   /**
    * @brief Constructor for Var.
    * @param n Number of bins.
    */
   Var(int n) : nbins(n), min(0), max(n) {}

   /**
    * @brief Constructor for Var from JSONNode.
    * @param val JSONNode containing variable information.
    */
   Var(const JSONNode &val);
};

/**
 * @brief Check if a string represents a valid number.
 *
 * This function checks whether the provided string 'str' represents a valid number.
 * The function returns true if the entire string can be parsed as a number (integer or floating-point); otherwise, it
 * returns false.
 *
 * @param str The string to be checked.
 * @return bool Returns true if the string 'str' represents a valid number; otherwise, returns false.
 */
bool isNumber(const std::string &str)
{
   bool seen_digit = false;
   bool seen_dot = false;
   bool seen_e = false;
   bool after_e = false;
   bool sign_allowed = true;

   for (size_t i = 0; i < str.size(); ++i) {
      char c = str[i];

      if (std::isdigit(c)) {
         seen_digit = true;
         sign_allowed = false;
      } else if ((c == '+' || c == '-') && sign_allowed) {
         // Sign allowed at the beginning or right after 'e'/'E'
         sign_allowed = false;
      } else if (c == '.' && !seen_dot && !after_e) {
         seen_dot = true;
         sign_allowed = false;
      } else if ((c == 'e' || c == 'E') && seen_digit && !seen_e) {
         seen_e = true;
         after_e = true;
         sign_allowed = true; // allow sign immediately after 'e'
         seen_digit = false;  // reset: we now expect digits after e
      } else {
         return false;
      }
   }

   return seen_digit;
}

/**
 * @brief Configure a RooRealVar based on information from a JSONNode.
 *
 * This function configures the provided RooRealVar 'v' based on the information provided in the JSONNode 'p'.
 * The JSONNode 'p' contains information about various properties of the RooRealVar, such as its value, error, number of
 * bins, etc. The function reads these properties from the JSONNode and sets the corresponding properties of the
 * RooRealVar accordingly.
 *
 * @param domains The reference to the RooFit::JSONIO::Detail::Domains containing domain information for variables (not
 * used in this function).
 * @param p The JSONNode containing information about the properties of the RooRealVar 'v'.
 * @param v The reference to the RooRealVar to be configured.
 * @return void
 */
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

Var::Var(const JSONNode &val)
{
   if (val.find("edges")) {
      for (auto const &child : val.children()) {
         this->edges.push_back(child.val_double());
      }
      this->nbins = this->edges.size();
      this->min = this->edges[0];
      this->max = this->edges[this->nbins - 1];
   } else {
      if (!val.find("nbins")) {
         this->nbins = 1;
      } else {
         this->nbins = val["nbins"].val_int();
      }
      if (!val.find("min")) {
         this->min = 0;
      } else {
         this->min = val["min"].val_double();
      }
      if (!val.find("max")) {
         this->max = 1;
      } else {
         this->max = val["max"].val_double();
      }
   }
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

/**
 * @brief Import attributes from a JSONNode into a RooAbsArg.
 *
 * This function imports attributes, represented by the provided JSONNode 'node', into the provided RooAbsArg 'arg'.
 * The attributes are read from the JSONNode and applied to the RooAbsArg.
 *
 * @param arg The pointer to the RooAbsArg to which the attributes will be imported.
 * @param node The JSONNode containing information about the attributes to be imported.
 * @return void
 */
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
         std::stringstream errMsg;
         errMsg << "node '" << name << "' is missing key '" << k << "'";
         RooJSONFactoryWSTool::error(errMsg.str());
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

/**
 * @brief Generate bin indices for a set of RooRealVars.
 *
 * This function generates all possible combinations of bin indices for the provided RooArgSet 'vars' containing
 * RooRealVars. Each bin index represents a possible bin selection for the corresponding RooRealVar. The bin indices are
 * stored in a vector of vectors, where each inner vector represents a combination of bin indices for all RooRealVars.
 *
 * @param vars The RooArgSet containing the RooRealVars for which bin indices will be generated.
 * @return std::vector<std::vector<int>> A vector of vectors containing all possible combinations of bin indices.
 */
std::vector<std::vector<int>> generateBinIndices(const RooArgSet &vars)
{
   std::vector<std::vector<int>> combinations;
   std::vector<int> vars_numbins;
   vars_numbins.reserve(vars.size());
   for (const auto *absv : static_range_cast<RooRealVar *>(vars)) {
      vars_numbins.push_back(absv->getBins());
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

/**
 * @brief Check if a RooAbsArg is a literal constant variable.
 *
 * This function checks whether the provided RooAbsArg 'arg' is a literal constant variable.
 * A literal constant variable is a RooConstVar with a numeric value as a name.
 *
 * @param arg The reference to the RooAbsArg to be checked.
 * @return bool Returns true if 'arg' is a literal constant variable; otherwise, returns false.
 */
bool isLiteralConstVar(RooAbsArg const &arg)
{
   bool isRooConstVar = dynamic_cast<RooConstVar const *>(&arg);
   return isRooConstVar && isNumber(arg.GetName());
}

/**
 * @brief Export attributes of a RooAbsArg to a JSONNode.
 *
 * This function exports the attributes of the provided RooAbsArg 'arg' to the JSONNode 'rootnode'.
 *
 * @param arg The pointer to the RooAbsArg from which attributes will be exported.
 * @param rootnode The JSONNode to which the attributes will be exported.
 * @return void
 */
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

   // RooConstVars are not a thing in HS3, and also for RooFit they are not
   // that important: they are just constants. So we don't need to remember
   // any information about them.
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

/**
 * @brief Create several observables in the workspace.
 *
 * This function obtains a list of observables from the provided
 * RooWorkspace 'ws' based on their names given in the 'axes" field of
 * the JSONNode 'node'.  The observables are added to the RooArgSet
 * 'out'.
 *
 * @param ws The RooWorkspace in which the observables will be created.
 * @param node The JSONNode containing information about the observables to be created.
 * @param out The RooArgSet to which the created observables will be added.
 * @return void
 */
void getObservables(RooWorkspace const &ws, const JSONNode &node, RooArgSet &out)
{
   std::map<std::string, Var> vars;
   for (const auto &p : node["axes"].children()) {
      vars.emplace(RooJSONFactoryWSTool::name(p), Var(p));
   }

   for (auto v : vars) {
      std::string name(v.first);
      if (ws.var(name)) {
         out.add(*ws.var(name));
      } else {
         std::stringstream errMsg;
         errMsg << "The observable \"" << name << "\" could not be found in the workspace!";
         RooJSONFactoryWSTool::error(errMsg.str());
      }
   }
}

/**
 * @brief Import data from the JSONNode into the workspace.
 *
 * This function imports data, represented by the provided JSONNode 'p', into the workspace represented by the provided
 * RooWorkspace. The data information is read from the JSONNode and added to the workspace.
 *
 * @param p The JSONNode representing the data to be imported.
 * @param workspace The RooWorkspace to which the data will be imported.
 * @return std::unique_ptr<RooAbsData> A unique pointer to the RooAbsData object representing the imported data.
 *                                     The caller is responsible for managing the memory of the returned object.
 */
std::unique_ptr<RooAbsData> loadData(const JSONNode &p, RooWorkspace &workspace)
{
   std::string name(RooJSONFactoryWSTool::name(p));

   RooJSONFactoryWSTool::testValidName(name, true);

   std::string const &type = p["type"].val();
   if (type == "binned") {
      // binned
      return RooJSONFactoryWSTool::readBinnedData(p, name, RooJSONFactoryWSTool::readAxes(p));
   } else if (type == "unbinned") {
      // unbinned
      RooArgSet vars;
      getObservables(workspace, p, vars);
      RooArgList varlist(vars);
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
            std::stringstream errMsg;
            errMsg << "coordinate point '" << i << "' is not a list!";
            RooJSONFactoryWSTool::error(errMsg.str());
         }
         if (point.num_children() != varlist.size()) {
            RooJSONFactoryWSTool::error("inconsistent number of entries and observables!");
         }
         std::size_t j = 0;
         for (auto const &pointj : point.children()) {
            auto *v = static_cast<RooRealVar *>(varlist.at(j));
            v->setVal(pointj.val_double());
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

   std::stringstream ss;
   ss << "RooJSONFactoryWSTool() failed to create dataset " << name << std::endl;
   RooJSONFactoryWSTool::error(ss.str());
   return nullptr;
}

/**
 * @brief Import an analysis from the JSONNode into the workspace.
 *
 * This function imports an analysis, represented by the provided JSONNodes 'analysisNode' and 'likelihoodsNode',
 * into the workspace represented by the provided RooWorkspace. The analysis information is read from the JSONNodes
 * and added to the workspace as one or more RooStats::ModelConfig objects.
 *
 * @param rootnode The root JSONNode representing the entire JSON file.
 * @param analysisNode The JSONNode representing the analysis to be imported.
 * @param likelihoodsNode The JSONNode containing information about likelihoods associated with the analysis.
 * @param domainsNode The JSONNode containing information about domains associated with the analysis.
 * @param workspace The RooWorkspace to which the analysis will be imported.
 * @param datasets A vector of unique pointers to RooAbsData objects representing the data associated with the analysis.
 * @return void
 */
void importAnalysis(const JSONNode &rootnode, const JSONNode &analysisNode, const JSONNode &likelihoodsNode,
                    const JSONNode &domainsNode, RooWorkspace &workspace,
                    const std::vector<std::unique_ptr<RooAbsData>> &datasets)
{
   // if this is a toplevel pdf, also create a modelConfig for it
   std::string const &analysisName = RooJSONFactoryWSTool::name(analysisNode);
   JSONNode const *mcAuxNode = findRooFitInternal(rootnode, "ModelConfigs", analysisName);

   JSONNode const *mcNameNode = mcAuxNode ? mcAuxNode->find("mcName") : nullptr;
   std::string mcname = mcNameNode ? mcNameNode->val() : analysisName;
   if (workspace.obj(mcname))
      return;

   workspace.import(RooStats::ModelConfig{mcname.c_str(), mcname.c_str()});
   auto *mc = static_cast<RooStats::ModelConfig *>(workspace.obj(mcname));
   mc->SetWS(workspace);

   std::vector<std::string> nllDataNames;

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
      nllDataNames.push_back(nameNode.val());
      for (const auto &d : datasets) {
         if (d->GetName() == nameNode.val()) {
            observables.add(*d->get());
         }
      }
   }

   JSONNode const *pdfNameNode = mcAuxNode ? mcAuxNode->find("pdfName") : nullptr;
   std::string const pdfName = pdfNameNode ? pdfNameNode->val() : "simPdf";

   RooAbsPdf *pdf = static_cast<RooSimultaneous *>(workspace.pdf(pdfName));

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
         pdf = static_cast<RooSimultaneous *>(workspace.pdf(simPdfName));
      }
   }

   mc->SetPdf(*pdf);

   if (!extConstraints.empty())
      mc->SetExternalConstraints(extConstraints);

   auto readArgSet = [&](std::string const &name) {
      RooArgSet out;
      for (auto const &child : analysisNode[name].children()) {
         out.add(*workspace.arg(child.val()));
      }
      return out;
   };

   mc->SetParametersOfInterest(readArgSet("parameters_of_interest"));
   mc->SetObservables(observables);
   RooArgSet pars;
   pdf->getParameters(&observables, pars);

   // Figure out the set parameters that appear in the main measurement:
   // getAllConstraints() has the side effect to remove all parameters from
   // "mainPars" that are not part of any pdf over observables.
   RooArgSet mainPars{pars};
   pdf->getAllConstraints(observables, mainPars, /*stripDisconnected*/ true);

   RooArgSet domainPars;
   for (auto &domain : analysisNode["domains"].children()) {
      const auto &thisDomain = RooJSONFactoryWSTool::findNamedChild(domainsNode, domain.val());
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
         allVars.add(*component->get());
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

bool RooJSONFactoryWSTool::allowExportInvalidNames(true);
bool RooJSONFactoryWSTool::testValidName(const std::string &name, bool forceError)
{
   if (!RooJSONFactoryWSTool::isValidName(name)) {
      std::stringstream ss;
      ss << "RooJSONFactoryWSTool() name '" << name << "' is not valid!" << std::endl;
      if (RooJSONFactoryWSTool::allowExportInvalidNames && !forceError) {
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
   if (const auto &distributionsNode = _rootnodeInput->find("distributions")) {
      if (const auto &child = findNamedChild(*distributionsNode, objname)) {
         this->importFunction(*child, true);
         if (RooAbsPdf *retval = _workspace.pdf(objname))
            return retval;
      }
   }
   return nullptr;
}

template <>
RooAbsReal *RooJSONFactoryWSTool::requestImpl<RooAbsReal>(const std::string &objname)
{
   if (RooAbsReal *retval = _workspace.function(objname))
      return retval;
   if (isNumber(objname))
      return &RooFit::RooConst(std::stod(objname));
   if (RooAbsPdf *pdf = requestImpl<RooAbsPdf>(objname))
      return pdf;
   if (RooRealVar *var = requestImpl<RooRealVar>(objname))
      return var;
   if (const auto &functionNode = _rootnodeInput->find("functions")) {
      if (const auto &child = findNamedChild(*functionNode, objname)) {
         this->importFunction(*child, true);
         if (RooAbsReal *retval = _workspace.function(objname))
            return retval;
      }
   }
   return nullptr;
}

/**
 * @brief Export a variable from the workspace to a JSONNode.
 *
 * This function exports a variable, represented by the provided RooAbsArg pointer 'v', from the workspace to a
 * JSONNode. The variable's information is added to the JSONNode as key-value pairs.
 *
 * @param v The pointer to the RooAbsArg representing the variable to be exported.
 * @param node The JSONNode to which the variable will be exported.
 * @return void
 */
void RooJSONFactoryWSTool::exportVariable(const RooAbsArg *v, JSONNode &node)
{
   auto *cv = dynamic_cast<const RooConstVar *>(v);
   auto *rrv = dynamic_cast<const RooRealVar *>(v);
   if (!cv && !rrv)
      return;

   // for RooConstVar, if name and value are the same, we don't need to do anything
   if (cv && strcmp(cv->GetName(), TString::Format("%g", cv->getVal()).Data()) == 0) {
      return;
   }

   // this variable was already exported
   if (findNamedChild(node, v->GetName())) {
      return;
   }

   JSONNode &var = appendNamedChild(node, v->GetName());

   if (cv) {
      var["value"] << cv->getVal();
      var["const"] << true;
   } else if (rrv) {
      var["value"] << rrv->getVal();
      if (rrv->isConstant()) {
         var["const"] << rrv->isConstant();
      }
      if (rrv->getBins() != 100) {
         var["nbins"] << rrv->getBins();
      }
      _domains->readVariable(*rrv);
   }
}

/**
 * @brief Export variables from the workspace to a JSONNode.
 *
 * This function exports variables, represented by the provided RooArgSet, from the workspace to a JSONNode.
 * The variables' information is added to the JSONNode as key-value pairs.
 *
 * @param allElems The RooArgSet representing the variables to be exported.
 * @param n The JSONNode to which the variables will be exported.
 * @return void
 */
void RooJSONFactoryWSTool::exportVariables(const RooArgSet &allElems, JSONNode &n)
{
   // export a list of RooRealVar objects
   for (RooAbsArg *arg : allElems) {
      exportVariable(arg, n);
   }
}

std::string RooJSONFactoryWSTool::exportTransformed(const RooAbsReal *original, const std::string &suffix,
                                                    const std::string &formula)
{
   std::string newname = std::string(original->GetName()) + suffix;
   RooFit::Detail::JSONNode &trafo_node = appendNamedChild((*_rootnodeOutput)["functions"], newname);
   trafo_node["type"] << "generic_function";
   trafo_node["expression"] << TString::Format(formula.c_str(), original->GetName()).Data();
   this->setAttribute(newname, "roofit_skip"); // this function should not be imported back in
   return newname;
}

/**
 * @brief Export an object from the workspace to a JSONNode.
 *
 * This function exports an object, represented by the provided RooAbsArg, from the workspace to a JSONNode.
 * The object's information is added to the JSONNode as key-value pairs.
 *
 * @param func The RooAbsArg representing the object to be exported.
 * @param exportedObjectNames A set of strings containing names of previously exported objects to avoid duplicates.
 *                            This set is updated with the name of the newly exported object.
 * @return void
 */
void RooJSONFactoryWSTool::exportObject(RooAbsArg const &func, std::set<std::string> &exportedObjectNames)
{
   const std::string name = func.GetName();

   // if this element was already exported, skip
   if (exportedObjectNames.find(name) != exportedObjectNames.end())
      return;

   exportedObjectNames.insert(name);

   if (auto simPdf = dynamic_cast<RooSimultaneous const *>(&func)) {
      // RooSimultaneous is not used in the HS3 standard, we only export the
      // dependents and some ROOT internal information.
      exportObjects(func.servers(), exportedObjectNames);

      std::vector<std::string> channelNames;
      for (auto const &item : simPdf->indexCat()) {
         channelNames.push_back(item.first);
      }

      auto &infoNode = getRooFitInternal(*_rootnodeOutput, "combined_distributions").set_map();
      auto &child = infoNode[simPdf->GetName()].set_map();
      child["index_cat"] << simPdf->indexCat().GetName();
      exportCategory(simPdf->indexCat(), child);
      child["distributions"].set_seq();
      for (auto const &item : simPdf->indexCat()) {
         child["distributions"].append_child() << simPdf->getPdf(item.first.c_str())->GetName();
      }

      return;
   } else if (dynamic_cast<RooAbsCategory const *>(&func)) {
      // categories are created by the respective RooSimultaneous, so we're skipping the export here
      return;
   } else if (dynamic_cast<RooRealVar const *>(&func) || dynamic_cast<RooConstVar const *>(&func)) {
      exportVariable(&func, *_varsNode);
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
      std::stringstream ss;
      ss << "RooJSONFactoryWSTool() function node " + name + " is not a map!";
      RooJSONFactoryWSTool::error(ss.str());
      return;
   }
   std::string prefix = genPrefix(p, true);
   if (!prefix.empty())
      name = prefix + name;
   if (!p.has_child("type")) {
      std::stringstream ss;
      ss << "RooJSONFactoryWSTool() no type given for function '" << name << "', skipping." << std::endl;
      RooJSONFactoryWSTool::error(ss.str());
      return;
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
         ok = imp->importArg(this, p);
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
      std::stringstream err;
      err << "something went wrong importing function '" << name << "'.";
      RooJSONFactoryWSTool::error(err.str());
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
   for (auto *var : static_range_cast<RooRealVar *>(vars)) {
      JSONNode &obsNode = observablesNode.append_child().set_map();
      std::string name = var->GetName();
      RooJSONFactoryWSTool::testValidName(name, false);
      obsNode["name"] << name;
      if (var->getBinning().isUniform()) {
         obsNode["min"] << var->getMin();
         obsNode["max"] << var->getMax();
         obsNode["nbins"] << var->getBins();
      } else {
         auto &edges = obsNode["edges"];
         edges.set_seq();
         double val = var->getBinning().binLow(0);
         edges.append_child() << val;
         for (int i = 0; i < var->getBinning().numBins(); ++i) {
            val = var->getBinning().binHigh(i);
            edges.append_child() << val;
         }
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

/**
 * @brief Export a RooAbsCategory object to a JSONNode.
 *
 * This function exports a RooAbsCategory object, represented by the provided categories and indices,
 * to a JSONNode. The category labels and corresponding indices are added to the JSONNode as key-value pairs.
 *
 * @param cat The RooAbsCategory object to be exported.
 * @param node The JSONNode to which the category data will be exported.
 * @return void
 */
void RooJSONFactoryWSTool::exportCategory(RooAbsCategory const &cat, JSONNode &node)
{
   auto &labels = node["labels"].set_seq();
   auto &indices = node["indices"].set_seq();

   for (auto const &item : cat) {
      std::string label;
      if (std::isalpha(item.first[0])) {
         label = RooFit::Detail::makeValidVarName(item.first);
         if (label != item.first) {
            oocoutW(nullptr, IO) << "RooFitHS3: changed '" << item.first << "' to '" << label
                                 << "' to become a valid name";
         }
      } else {
         RooJSONFactoryWSTool::error("refusing to change first character of string '" + item.first +
                                     "' to make a valid name!");
         label = item.first;
      }
      labels.append_child() << label;
      indices.append_child() << item.second;
   }
}

/**
 * @brief Export combined data from the workspace to a custom struct.
 *
 * This function exports combined data from the workspace, represented by the provided RooAbsData object,
 * to a CombinedData struct. The struct contains information such as variables, categories,
 * and bin contents of the combined data.
 *
 * @param data The RooAbsData object representing the combined data to be exported.
 * @return CombinedData A custom struct containing the exported combined data.
 */
RooJSONFactoryWSTool::CombinedData RooJSONFactoryWSTool::exportCombinedData(RooAbsData const &data)
{
   // find category observables
   RooAbsCategory *cat = nullptr;
   for (RooAbsArg *obs : *data.get()) {
      if (dynamic_cast<RooAbsCategory *>(obs)) {
         if (cat) {
            RooJSONFactoryWSTool::error("dataset '" + std::string(data.GetName()) +
                                        " has several category observables!");
         }
         cat = static_cast<RooAbsCategory *>(obs);
      }
   }

   // prepare return value
   RooJSONFactoryWSTool::CombinedData datamap;

   if (!cat)
      return datamap;
   // this is a combined dataset

   datamap.name = data.GetName();

   // Write information necessary to reconstruct the combined dataset upon import
   auto &child = getRooFitInternal(*_rootnodeOutput, "combined_datasets").set_map()[data.GetName()].set_map();
   child["index_cat"] << cat->GetName();
   exportCategory(*cat, child);

   // Find a RooSimultaneous model that would fit to this dataset
   RooSimultaneous const *simPdf = nullptr;
   auto *combinedPdfInfoNode = findRooFitInternal(*_rootnodeOutput, "combined_distributions");
   if (combinedPdfInfoNode) {
      for (auto &info : combinedPdfInfoNode->children()) {
         if (info["index_cat"].val() == cat->GetName()) {
            simPdf = static_cast<RooSimultaneous const *>(_workspace.pdf(info.key()));
         }
      }
   }

   // If there is an associated simultaneous pdf for the index category, we
   // use the RooAbsData::split() overload that takes the RooSimultaneous.
   // Like this, the observables that are not relevant for a given channel
   // are automatically split from the component datasets.
   std::vector<std::unique_ptr<RooAbsData>> dataList{simPdf ? data.split(*simPdf, true) : data.split(*cat, true)};

   for (std::unique_ptr<RooAbsData> const &absData : dataList) {
      std::string catName(absData->GetName());
      std::string dataName;
      if (std::isalpha(catName[0])) {
         dataName = RooFit::Detail::makeValidVarName(catName);
         if (dataName != catName) {
            oocoutW(nullptr, IO) << "RooFitHS3: changed '" << catName << "' to '" << dataName
                                 << "' to become a valid name";
         }
      } else {
         RooJSONFactoryWSTool::error("refusing to change first character of string '" + catName +
                                     "' to make a valid name!");
         dataName = catName;
      }
      absData->SetName((std::string(data.GetName()) + "_" + dataName).c_str());
      datamap.components[catName] = absData->GetName();
      this->exportData(*absData);
   }
   return datamap;
}

/**
 * @brief Export data from the workspace to a JSONNode.
 *
 * This function exports data represented by the provided RooAbsData object,
 * to a JSONNode. The data's information is added as key-value pairs to the JSONNode.
 *
 * @param data The RooAbsData object representing the data to be exported.
 * @return void
 */
void RooJSONFactoryWSTool::exportData(RooAbsData const &data)
{
   // find category observables
   RooAbsCategory *cat = nullptr;
   for (RooAbsArg *obs : *data.get()) {
      if (dynamic_cast<RooAbsCategory *>(obs)) {
         if (cat) {
            RooJSONFactoryWSTool::error("dataset '" + std::string(data.GetName()) +
                                        " has several category observables!");
         }
         cat = static_cast<RooAbsCategory *>(obs);
      }
   }

   if (cat)
      return;

   JSONNode &output = appendNamedChild((*_rootnodeOutput)["data"], data.GetName());

   // this is a binned dataset
   if (auto dh = dynamic_cast<RooDataHist const *>(&data)) {
      output["type"] << "binned";
      return exportHisto(*dh->get(), dh->numEntries(), dh->weightArray(), output);
   }

   // this is a regular unbinned dataset

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

   // Check if this actually represents a binned dataset, and then import it
   // like a RooDataHist. This happens frequently when people create combined
   // RooDataSets from binned data to fit HistFactory models. In this case, it
   // doesn't make sense to export them like an unbinned dataset, because the
   // coordinates are redundant information with the binning. We only do this
   // for 1D data for now.
   if (data.isWeighted() && variables.size() == 1) {
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
         return exportHisto(variables, data.numEntries(), contents.data(), output);
      }
   }

   output["type"] << "unbinned";

   for (RooAbsArg *arg : variables) {
      exportVariable(arg, output["axes"]);
   }
   auto &coords = output["entries"].set_seq();
   std::vector<double> weightVals;
   bool hasNonUnityWeights = false;
   for (int i = 0; i < data.numEntries(); ++i) {
      data.get(i);
      coords.append_child().fill_seq(variables, [](auto x) { return static_cast<RooRealVar *>(x)->getVal(); });
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
      if (node.has_child("edges")) {
         std::vector<double> edges;
         for (auto const &bound : node["edges"].children()) {
            edges.push_back(bound.val_double());
         }
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
   }

   auto bins = generateBinIndices(vars);
   if (contents.num_children() != bins.size()) {
      std::stringstream errMsg;
      errMsg << "inconsistent bin numbers: contents=" << contents.num_children() << ", bins=" << bins.size();
      RooJSONFactoryWSTool::error(errMsg.str());
   }
   auto dh = std::make_unique<RooDataHist>(name, name, vars);
   std::vector<double> contentVals;
   contentVals.reserve(contents.num_children());
   for (auto const &cont : contents.children()) {
      contentVals.push_back(cont.val_double());
   }
   std::vector<double> errorVals;
   if (errors) {
      errorVals.reserve(errors->num_children());
      for (auto const &err : errors->children()) {
         errorVals.push_back(err.val_double());
      }
   }
   for (size_t ibin = 0; ibin < bins.size(); ++ibin) {
      const double err = errors ? errorVals[ibin] : -1;
      dh->set(ibin, contentVals[ibin], err);
   }
   return dh;
}

/**
 * @brief Import a variable from the JSONNode into the workspace.
 *
 * This function imports a variable from the given JSONNode into the workspace.
 * The variable's information is read from the JSONNode and added to the workspace.
 *
 * @param p The JSONNode representing the variable to be imported.
 * @return void
 */
void RooJSONFactoryWSTool::importVariable(const JSONNode &p)
{
   // import a RooRealVar object
   std::string name(RooJSONFactoryWSTool::name(p));
   RooJSONFactoryWSTool::testValidName(name, true);

   if (_workspace.var(name))
      return;
   if (!p.is_map()) {
      std::stringstream ss;
      ss << "RooJSONFactoryWSTool() node '" << name << "' is not a map, skipping.";
      oocoutE(nullptr, InputArguments) << ss.str() << std::endl;
      return;
   }
   if (_attributesNode) {
      if (auto *attrNode = _attributesNode->find(name)) {
         // We should not create RooRealVar objects for RooConstVars!
         if (attrNode->has_child("is_const_var") && (*attrNode)["is_const_var"].val_int() == 1) {
            wsEmplace<RooConstVar>(name, p["value"].val_double());
            return;
         }
      }
   }
   configureVariable(*_domains, p, wsEmplace<RooRealVar>(name, 1.));
}

/**
 * @brief Import all dependants (servers) of a node into the workspace.
 *
 * This function imports all the dependants (servers) of the given JSONNode into the workspace.
 * The dependants' information is read from the JSONNode and added to the workspace.
 *
 * @param n The JSONNode representing the node whose dependants are to be imported.
 * @return void
 */
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

void RooJSONFactoryWSTool::exportModelConfig(JSONNode &rootnode, RooStats::ModelConfig const &mc,
                                             const std::vector<CombinedData> &combDataSets)
{
   auto pdf = dynamic_cast<RooSimultaneous const *>(mc.GetPdf());
   if (pdf == nullptr) {
      warning("RooFitHS3 only supports ModelConfigs with RooSimultaneous! Skipping ModelConfig.");
      return;
   }

   for (std::size_t i = 0; i < std::max(combDataSets.size(), std::size_t(1)); ++i) {
      const bool hasdata = i < combDataSets.size();
      if (hasdata && !matches(combDataSets.at(i), pdf))
         continue;

      std::string analysisName(pdf->GetName());
      if (hasdata)
         analysisName += "_" + combDataSets[i].name;

      exportSingleModelConfig(rootnode, mc, analysisName, hasdata ? &combDataSets[i].components : nullptr);
   }
}

void RooJSONFactoryWSTool::exportSingleModelConfig(JSONNode &rootnode, RooStats::ModelConfig const &mc,
                                                   std::string const &analysisName,
                                                   std::map<std::string, std::string> const *dataComponents)
{
   auto pdf = static_cast<RooSimultaneous const *>(mc.GetPdf());

   JSONNode &analysisNode = appendNamedChild(rootnode["analyses"], analysisName);

   auto &domains = analysisNode["domains"].set_seq();

   analysisNode["likelihood"] << analysisName;

   auto &nllNode = appendNamedChild(rootnode["likelihoods"], analysisName);
   nllNode["distributions"].set_seq();
   nllNode["data"].set_seq();

   if (dataComponents) {
      for (auto const &item : pdf->indexCat()) {
         const auto &dataComp = dataComponents->find(item.first);
         nllNode["distributions"].append_child() << pdf->getPdf(item.first)->GetName();
         nllNode["data"].append_child() << dataComp->second;
      }
   }

   if (mc.GetExternalConstraints()) {
      auto &extConstrNode = nllNode["aux_distributions"];
      extConstrNode.set_seq();
      for (const auto &constr : *mc.GetExternalConstraints()) {
         extConstrNode.append_child() << constr->GetName();
      }
   }

   auto writeList = [&](const char *name, RooArgSet const *args) {
      if (!args)
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

   if (mc.GetNuisanceParameters()) {
      std::string npDomainName = analysisName + "_nuisance_parameters";
      domains.append_child() << npDomainName;
      RooFit::JSONIO::Detail::Domains::ProductDomain npDomain;
      for (auto *np : static_range_cast<const RooRealVar *>(*mc.GetNuisanceParameters())) {
         npDomain.readVariable(*np);
      }
      npDomain.writeJSON(appendNamedChild(domainsNode, npDomainName));
   }

   if (mc.GetGlobalObservables()) {
      std::string globDomainName = analysisName + "_global_observables";
      domains.append_child() << globDomainName;
      RooFit::JSONIO::Detail::Domains::ProductDomain globDomain;
      for (auto *glob : static_range_cast<const RooRealVar *>(*mc.GetGlobalObservables())) {
         globDomain.readVariable(*glob);
      }
      globDomain.writeJSON(appendNamedChild(domainsNode, globDomainName));
   }

   if (mc.GetParametersOfInterest()) {
      std::string poiDomainName = analysisName + "_parameters_of_interest";
      domains.append_child() << poiDomainName;
      RooFit::JSONIO::Detail::Domains::ProductDomain poiDomain;
      for (auto *poi : static_range_cast<const RooRealVar *>(*mc.GetParametersOfInterest())) {
         poiDomain.readVariable(*poi);
      }
      poiDomain.writeJSON(appendNamedChild(domainsNode, poiDomainName));
   }

   auto &modelConfigAux = getRooFitInternal(rootnode, "ModelConfigs", analysisName);
   modelConfigAux.set_map();
   modelConfigAux["pdfName"] << pdf->GetName();
   modelConfigAux["mcName"] << mc.GetName();
}

/**
 * @brief Export all objects in the workspace to a JSONNode.
 *
 * This function exports all the objects in the workspace to the provided JSONNode.
 * The objects' information is added as key-value pairs to the JSONNode.
 *
 * @param n The JSONNode to which the objects will be exported.
 * @return void
 */
void RooJSONFactoryWSTool::exportAllObjects(JSONNode &n)
{
   _domains = std::make_unique<RooFit::JSONIO::Detail::Domains>();
   _varsNode = &makeVariablesNode(n);
   _rootnodeOutput = &n;

   // export all toplevel pdfs
   std::vector<RooAbsPdf *> allpdfs;
   for (auto &arg : _workspace.allPdfs()) {
      if (!arg->hasClients()) {
         if (auto *pdf = dynamic_cast<RooAbsPdf *>(arg)) {
            allpdfs.push_back(pdf);
         }
      }
   }
   sortByName(allpdfs);
   std::set<std::string> exportedObjectNames;
   exportObjects(allpdfs, exportedObjectNames);

   // export attributes of all objects
   for (RooAbsArg *arg : _workspace.components()) {
      exportAttributes(arg, n);
   }

   // export all datasets
   std::vector<RooAbsData *> alldata;
   for (auto &d : _workspace.allData()) {
      alldata.push_back(d);
   }
   sortByName(alldata);
   // first, take care of combined datasets
   std::vector<RooJSONFactoryWSTool::CombinedData> combData;
   for (auto &d : alldata) {
      auto data = this->exportCombinedData(*d);
      if (!data.components.empty())
         combData.push_back(data);
   }
   // next, take care of regular datasets
   for (auto &d : alldata) {
      this->exportData(*d);
   }

   // export all ModelConfig objects and attached Pdfs
   for (TObject *obj : _workspace.allGenericObjects()) {
      if (auto mc = dynamic_cast<RooStats::ModelConfig *>(obj)) {
         exportModelConfig(n, *mc, combData);
      }
   }

   for (auto *snsh : static_range_cast<RooArgSet const *>(_workspace.getSnapshots())) {
      RooArgSet snapshotSorted;
      // We only want to add the variables that actually got exported and skip
      // the ones that the pdfs encoded implicitly (like in the case of
      // HistFactory).
      for (RooAbsArg *arg : *snsh) {
         if (exportedObjectNames.find(arg->GetName()) != exportedObjectNames.end()) {
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
      }
      snapshotSorted.sort();
      std::string name(snsh->GetName());
      if (name != "default_values") {
         this->exportVariables(snapshotSorted, appendNamedChild(n["parameter_points"], name)["parameters"]);
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
 * @brief Import the workspace from a YML string.
 *
 * @param s The YML string containing the workspace data.
 * @return bool Returns true on successful import, false otherwise.
 */
bool RooJSONFactoryWSTool::importYMLfromString(const std::string &s)
{
   std::stringstream ss(s);
   return importYML(ss);
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
 * @brief Export the workspace to a YML string.
 *
 * @return std::string The YML string representing the exported workspace.
 */
std::string RooJSONFactoryWSTool::exportYMLtoString()
{
   std::stringstream ss;
   exportYML(ss);
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
   if (!out.is_open()) {
      std::stringstream ss;
      ss << "RooJSONFactoryWSTool() invalid output file '" << filename << "'." << std::endl;
      RooJSONFactoryWSTool::error(ss.str());
      return false;
   }
   return this->exportJSON(out);
}

/**
 * @brief Export the workspace to YML format and write to the output stream.
 *
 * @param os The output stream to write the YML data to.
 * @return bool Returns true on successful export, false otherwise.
 */
bool RooJSONFactoryWSTool::exportYML(std::ostream &os)
{
   std::unique_ptr<JSONTree> tree = createNewJSONTree();
   JSONNode &n = tree->rootnode();
   this->exportAllObjects(n);
   n.writeYML(os);
   return true;
}

/**
 * @brief Export the workspace to YML format and write to the specified file.
 *
 * @param filename The name of the YML file to create and write the data to.
 * @return bool Returns true on successful export, false otherwise.
 */
bool RooJSONFactoryWSTool::exportYML(std::string const &filename)
{
   std::ofstream out(filename.c_str());
   if (!out.is_open()) {
      std::stringstream ss;
      ss << "RooJSONFactoryWSTool() invalid output file '" << filename << "'." << std::endl;
      RooJSONFactoryWSTool::error(ss.str());
      return false;
   }
   return this->exportYML(out);
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

/**
 * @brief Imports all nodes of the JSON data and adds them to the workspace.
 *
 * @param n The JSONNode representing the root node of the JSON data.
 * @return void
 */
void RooJSONFactoryWSTool::importAllNodes(const JSONNode &n)
{
   // Per HS3 standard, the hs3_version in the metadata is required. So we
   // error out if it is missing. TODO: now we are only checking if the
   // hs3_version tag exists, but in the future when the HS3 specification
   // versions are actually frozen, we should also check if the hs3_version is
   // one that RooFit can actually read.
   auto metadata = n.find("metadata");
   if (!metadata || !metadata->find("hs3_version")) {
      std::stringstream ss;
      ss << "The HS3 version is missing in the JSON!\n"
         << "Please include the HS3 version in the metadata field, e.g.:\n"
         << "    \"metadata\" :\n"
         << "    {\n"
         << "        \"hs3_version\" : \"" << hs3VersionTag << "\"\n"
         << "    }";
      error(ss.str());
   }

   _domains = std::make_unique<RooFit::JSONIO::Detail::Domains>();
   if (auto domains = n.find("domains")) {
      _domains->readJSON(*domains);
   }
   _domains->populate(_workspace);

   _rootnodeInput = &n;

   _attributesNode = findRooFitInternal(*_rootnodeInput, "attributes");

   this->importDependants(n);

   if (auto paramPointsNode = n.find("parameter_points")) {
      for (const auto &snsh : paramPointsNode->children()) {
         std::string name = RooJSONFactoryWSTool::name(snsh);
         RooJSONFactoryWSTool::testValidName(name, true);

         RooArgSet vars;
         for (const auto &var : snsh["parameters"].children()) {
            if (RooRealVar *rrv = _workspace.var(RooJSONFactoryWSTool::name(var))) {
               configureVariable(*_domains, var, *rrv);
               vars.add(*rrv);
            }
         }
         _workspace.saveSnapshot(name, vars);
      }
   }

   combinePdfs(*_rootnodeInput, _workspace);

   // Import attributes
   if (_attributesNode) {
      for (const auto &elem : _attributesNode->children()) {
         if (RooAbsArg *arg = _workspace.arg(elem.key()))
            importAttributes(arg, elem);
      }
   }

   _attributesNode = nullptr;

   // We delay the import of the data to after combineDatasets(), because it
   // might be that some datasets are merged to combined datasets there. In
   // that case, we will remove the components from the "datasets" vector so they
   // don't get imported.
   std::vector<std::unique_ptr<RooAbsData>> datasets;
   if (auto dataNode = n.find("data")) {
      for (const auto &p : dataNode->children()) {
         datasets.push_back(loadData(p, _workspace));
      }
   }

   // Now, read in analyses and likelihoods if there are any

   if (auto analysesNode = n.find("analyses")) {
      for (JSONNode const &analysisNode : analysesNode->children()) {
         importAnalysis(*_rootnodeInput, analysisNode, n["likelihoods"], n["domains"], _workspace, datasets);
      }
   }

   combineDatasets(*_rootnodeInput, datasets);

   for (auto const &d : datasets) {
      if (d)
         _workspace.import(*d);
   }

   _rootnodeInput = nullptr;
   _domains.reset();
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
   this->importAllNodes(tree->rootnode());
   if (this->workspace()->getSnapshot("default_values")) {
      this->workspace()->loadSnapshot("default_values");
   }
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
   if (!infile.is_open()) {
      std::stringstream ss;
      ss << "RooJSONFactoryWSTool() invalid input file '" << filename << "'." << std::endl;
      RooJSONFactoryWSTool::error(ss.str());
      return false;
   }
   return this->importJSON(infile);
}

/**
 * @brief Imports a YML file from the given input stream to the workspace.
 *
 * @param is The input stream containing the YML data.
 * @return bool Returns true on successful import, false otherwise.
 */
bool RooJSONFactoryWSTool::importYML(std::istream &is)
{
   // import a YML file to the workspace
   std::unique_ptr<JSONTree> tree = JSONTree::create(is);
   this->importAllNodes(tree->rootnode());
   return true;
}

/**
 * @brief Imports a YML file from the given filename to the workspace.
 *
 * @param filename The name of the YML file to import.
 * @return bool Returns true on successful import, false otherwise.
 */
bool RooJSONFactoryWSTool::importYML(std::string const &filename)
{
   // import a YML file to the workspace
   std::ifstream infile(filename.c_str());
   if (!infile.is_open()) {
      std::stringstream ss;
      ss << "RooJSONFactoryWSTool() invalid input file '" << filename << "'." << std::endl;
      RooJSONFactoryWSTool::error(ss.str());
      return false;
   }
   return this->importYML(infile);
}

void RooJSONFactoryWSTool::importJSONElement(const std::string &name, const std::string &jsonString)
{
   std::unique_ptr<RooFit::Detail::JSONTree> tree = RooFit::Detail::JSONTree::create(jsonString);
   JSONNode &n = tree->rootnode();
   n["name"] << name;

   bool isVariable = true;
   if (n.find("type")) {
      isVariable = false;
   }

   if (isVariable) {
      this->importVariableElement(n);
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
   const auto &snsh = paramPointsNode->child(0);
   std::string name = RooJSONFactoryWSTool::name(snsh);
   RooArgSet vars;
   const auto &var = snsh["parameters"].child(0);
   if (RooRealVar *rrv = _workspace.var(RooJSONFactoryWSTool::name(var))) {
      configureVariable(*_domains, var, *rrv);
      vars.add(*rrv);
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
   return RooMsgService::instance().log(nullptr, RooFit::MsgLevel::ERROR, RooFit::IO) << str << std::endl;
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
