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
#include <RooFit/ModelConfig.h>

#include "JSONIOUtils.h"
#include "Domains.h"

#include <TROOT.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <stack>
#include <stdexcept>

/** \class RooJSONFactoryWSTool
\ingroup roofit

When using `RooFit`, statistical models can be conveniently handled and
stored as a `RooWorkspace`. However, for the sake of interoperability
with other statistical frameworks, and also ease of manipulation, it
may be useful to store statistical models in text form.

The RooJSONFactoryWSTool is a helper class to achieve exactly this,
exporting to and importing from JSON and YML.

In order to import a workspace from a JSON file, you can do

~~ {.py}
ws = ROOT.RooWorkspace("ws")
tool = ROOT.RooJSONFactoryWSTool(ws)
tool.importJSON("myjson.json")
~~

Similarly, in order to export a workspace to a JSON file, you can do

~~ {.py}
tool = ROOT.RooJSONFactoryWSTool(ws)
tool.exportJSON("myjson.json")
~~

For more details, consult the tutorial <a
href="https://root.cern/doc/v626/rf515__hfJSON_8py.html">rf515_hfJSON</a>.

In order to import and export YML files, `ROOT` needs to be compiled
with the external dependency <a
href="https://github.com/biojppm/rapidyaml">RapidYAML</a>, which needs
to be installed on your system and enabled via the CMake option
`roofit_hs3_ryml`.

The RooJSONFactoryWSTool only knows about a limited set of classes for
import and export. If import or export of a class you're interested in
fails, you might need to add your own importer or exporter. Please
consult the <a
href="https://github.com/root-project/root/blob/master/roofit/hs3/README.md">README</a>
to learn how to do that.

You can always get a list of all the avialable importers and exporters by calling the following functions:
~~ {.py}
ROOT.RooFit.JSONIO.printImporters()
ROOT.RooFit.JSONIO.printExporters()
ROOT.RooFit.JSONIO.printFactoryExpressions()
ROOT.RooFit.JSONIO.printExportKeys()
~~

Alternatively, you can generate a LaTeX version of the available importers and exporters by calling
~~ {.py}
tool = ROOT.RooJSONFactoryWSTool(ws)
tool.writedoc("hs3.tex")
~~

*/

using RooFit::Detail::JSONNode;
using RooFit::Detail::JSONTree;

namespace {

bool matches(const RooJSONFactoryWSTool::CombinedData &data, const RooSimultaneous *pdf)
{
   return data.components.size() == pdf->indexCat().size();
}

struct Var {
   int nbins;
   double min;
   double max;
   std::vector<double> bounds;

   Var(int n) : nbins(n), min(0), max(n) {}
   Var(const JSONNode &val);
};

bool isNumber(const std::string &str)
{
   bool first = true;
   for (char const &c : str) {
      if (std::isdigit(c) == 0 && c != '.' && !(first && (c == '-' || c == '+')))
         return false;
      first = false;
   }
   return true;
}

void configureVariable(RooFit::JSONIO::Detail::Domains &domains, const JSONNode &p, RooRealVar &v)
{
   if (auto n = p.find("value"))
      v.setVal(n->val_double());
   domains.writeVariable(v);
   if (auto n = p.find("nbins"))
      v.setBins(n->val_int());
   if (auto n = p.find("relErr"))
      v.setError(v.getVal() * n->val_double());
   if (auto n = p.find("err"))
      v.setError(n->val_double());
   if (auto n = p.find("const"))
      v.setConstant(n->val_bool());
   else
      v.setConstant(false);
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

void logInputArgumentsError(std::stringstream &&ss)
{
   oocoutE(nullptr, InputArguments) << ss.str() << std::endl;
}

Var::Var(const JSONNode &val)
{
   if (val.find("bounds")) {
      for (auto const &child : val.children()) {
         this->bounds.push_back(child.val_double());
      }
      this->nbins = this->bounds.size();
      this->min = this->bounds[0];
      this->max = this->bounds[this->nbins - 1];
   } else {
      if (!val.find("nbins"))
         this->nbins = 1;
      else
         this->nbins = val["nbins"].val_int();
      if (!val.find("min"))
         this->min = 0;
      else
         this->min = val["min"].val_double();
      if (!val.find("max"))
         this->max = 1;
      else
         this->max = val["max"].val_double();
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
   size_t colon = classname.find_last_of(":");
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
         for (RooAbsArg *arg : tool->requestArgList<RooAbsReal>(p, k)) {
            expression << (firstInner ? "{" : ",") << arg->GetName();
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

std::vector<std::vector<int>> generateBinIndices(const RooArgList &vars)
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

bool isLiteralConstVar(RooAbsArg const &arg)
{
   bool isRooConstVar = dynamic_cast<RooConstVar const *>(&arg);
   return isRooConstVar && isNumber(arg.GetName());
}

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

   // We have to remember if the variable was a constant RooRealVar or a
   // RooConstVar in RooFit to reconstruct the workspace correctly. The HS3
   // standard does not make this distinction.
   bool isRooConstVar = dynamic_cast<RooConstVar const *>(arg);
   if (isRooConstVar) {
      initializeNode();
      (*node)["is_const_var"] << 1;
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

///////////////////////////////////////////////////////////////////////////////////////////////////////
// create several observables
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

///////////////////////////////////////////////////////////////////////////////////////////////////////
// importing data
std::unique_ptr<RooAbsData> loadData(const JSONNode &p, RooWorkspace &workspace)
{
   std::string name(RooJSONFactoryWSTool::name(p));
   std::string const &type = p["type"].val();
   if (type == "binned") {
      // binned
      return RooJSONFactoryWSTool::readBinnedData(p, name);
   } else if (type == "unbinned") {
      // unbinned
      RooArgSet vars;
      getObservables(workspace, p, vars);
      RooArgList varlist(vars);
      auto data = std::make_unique<RooDataSet>(name, name, vars, RooFit::WeightVar());
      auto &coords = p["entries"];
      auto &weights = p["weights"];
      if (coords.num_children() != weights.num_children()) {
         RooJSONFactoryWSTool::error("inconsistent number of entries and weights!");
      }
      if (!coords.is_seq()) {
         RooJSONFactoryWSTool::error("key 'entries' is not a list!");
      }
      std::vector<double> weightVals;
      for (auto const &weight : weights.children()) {
         weightVals.push_back(weight.val_double());
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
         data->add(vars, weightVals[i]);
         ++i;
      }
      return data;
   }

   std::stringstream ss;
   ss << "RooJSONFactoryWSTool() failed to create dataset " << name << std::endl;
   logInputArgumentsError(std::move(ss));
   return nullptr;
}

void importAnalysis(const JSONNode &rootnode, const JSONNode &analysisNode, const JSONNode &likelihoodsNode,
                    RooWorkspace &workspace, const std::vector<std::unique_ptr<RooAbsData>> &datas)
{
   // if this is a toplevel pdf, also create a modelConfig for it
   std::string const &analysisName = RooJSONFactoryWSTool::name(analysisNode);
   JSONNode const *mcAuxNode = findRooFitInternal(rootnode, "ModelConfigs", analysisName);

   JSONNode const *mcNameNode = mcAuxNode ? mcAuxNode->find("mcName") : nullptr;
   std::string mcname = mcNameNode ? mcNameNode->val() : analysisName.c_str();
   if (workspace.obj(mcname))
      return;

   workspace.import(RooStats::ModelConfig{mcname.c_str(), mcname.c_str()});
   auto *mc = static_cast<RooStats::ModelConfig *>(workspace.obj(mcname));
   mc->SetWS(workspace);

   std::vector<std::string> nllDistNames;
   std::vector<std::string> nllDataNames;

   auto *nllNode = RooJSONFactoryWSTool::findNamedChild(likelihoodsNode, analysisNode["likelihood"].val());
   if (!nllNode) {
      throw std::runtime_error("likelihood node not found!");
   }
   for (auto &nameNode : (*nllNode)["distributions"].children()) {
      nllDistNames.push_back(nameNode.val());
   }
   RooArgSet extConstraints;
   for (auto &nameNode : (*nllNode)["aux_distributions"].children()) {
      RooAbsArg *extConstraint = workspace.arg(nameNode.val());
      if (extConstraint) {
         extConstraints.add(*extConstraint);
      }
   }
   RooArgSet observables;
   for (auto &nameNode : (*nllNode)["data"].children()) {
      nllDataNames.push_back(nameNode.val());
      for (const auto &d : datas) {
         if (d->GetName() == nameNode.val()) {
            observables.add(*d->get());
         }
      }
   }

   JSONNode const *pdfNameNode = mcAuxNode ? mcAuxNode->find("pdfName") : nullptr;
   std::string const pdfName = pdfNameNode ? pdfNameNode->val() : "simPdf";

   auto *pdf = static_cast<RooSimultaneous *>(workspace.pdf(pdfName));
   if (!pdf)
      std::runtime_error("pdf not found!");

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

   RooArgSet nps;
   RooArgSet globs;
   for (const auto &p : pars) {
      if (mc->GetParametersOfInterest()->find(*p))
         continue;
      if (p->isConstant() && !mainPars.find(*p))
         globs.add(*p);
      else
         nps.add(*p);
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
      std::vector<std::string> labels;
      std::vector<int> indices;
      std::vector<std::string> pdfNames;
      for (auto &n : info["indices"].children()) {
         indices.push_back(n.val_int());
      }
      for (auto &n : info["labels"].children()) {
         labels.push_back(n.val());
      }
      for (auto &n : info["distributions"].children()) {
         pdfNames.push_back(n.val());
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

void combineDatasets(const JSONNode &rootnode, std::vector<std::unique_ptr<RooAbsData>> &datas)
{
   auto *combinedDataInfoNode = findRooFitInternal(rootnode, "combined_datas");

   // If there is no info on combining datasets
   if (combinedDataInfoNode == nullptr) {
      return;
   }

   for (auto &info : combinedDataInfoNode->children()) {

      // parse the information
      std::string combinedName = info.key();
      std::string indexCatName = info["index_cat"].val();
      std::vector<std::string> labels;
      std::vector<int> indices;
      for (auto &n : info["indices"].children()) {
         indices.push_back(n.val_int());
      }
      for (auto &n : info["labels"].children()) {
         labels.push_back(n.val());
      }

      // Create the combined dataset for RooFit
      std::map<std::string, std::unique_ptr<RooAbsData>> dsMap;
      RooCategory indexCat{indexCatName.c_str(), indexCatName.c_str()};
      RooArgSet allVars{indexCat};
      for (std::size_t iChannel = 0; iChannel < labels.size(); ++iChannel) {
         auto componentName = combinedName + "_" + labels[iChannel];
         // We move the found channel data out of the "datas" vector, such that
         // the data components don't get imported anymore.
         std::unique_ptr<RooAbsData> &component =
            *std::find_if(datas.begin(), datas.end(), [&](auto &d) { return d && d->GetName() == componentName; });
         allVars.add(*component->get());
         dsMap.insert({labels[iChannel], std::move(component)});
         indexCat.defineType(labels[iChannel], indices[iChannel]);
      }

      auto combined = std::make_unique<RooDataSet>(combinedName.c_str(), combinedName.c_str(), allVars,
                                                   RooFit::Import(dsMap), RooFit::Index(indexCat));
      datas.emplace_back(std::move(combined));
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

void RooJSONFactoryWSTool::fillSeq(JSONNode &node, RooAbsCollection const &coll)
{
   node.set_seq();
   for (RooAbsArg const *arg : coll) {
      if (isLiteralConstVar(*arg))
         node.append_child() << static_cast<RooConstVar const *>(arg)->getVal();
      else
         node.append_child() << arg->GetName();
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
   if (JSONNode const *vars = getVariablesNode(*_rootnodeInput)) {
      if (auto node = vars->find(objname)) {
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
   if (auto distributionsNode = _rootnodeInput->find("distributions")) {
      if (auto child = findNamedChild(*distributionsNode, objname)) {
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
   if (auto functionNode = _rootnodeInput->find("functions")) {
      if (auto child = findNamedChild(*functionNode, objname)) {
         this->importFunction(*child, true);
         if (RooAbsReal *retval = _workspace.function(objname))
            return retval;
      }
   }
   return nullptr;
}

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

void RooJSONFactoryWSTool::exportVariables(const RooArgSet &allElems, JSONNode &n)
{
   // export a list of RooRealVar objects
   for (RooAbsArg *arg : allElems) {
      exportVariable(arg, n);
   }
}

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
      for (RooAbsArg *s : func.servers()) {
         this->exportObject(*s, exportedObjectNames);
      }

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
            for (RooAbsArg *s : func.servers()) {
               this->exportObject(*s, exportedObjectNames);
            }
         } else {
            for (RooAbsArg const *s : _serversToExport) {
               this->exportObject(*s, exportedObjectNames);
            }
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
                   "https://github.com/root-project/root/blob/master/roofit/hs3/README.md to "
                   "see how to do this!\n";
      return;
   }

   elem["type"] << dict->second.type;

   size_t nprox = func.numProxies();

   for (size_t i = 0; i < nprox; ++i) {
      RooAbsProxy *p = func.getProxy(i);

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

      if (auto l = dynamic_cast<RooListProxy *>(p)) {
         fillSeq(elem[k->second], *l);
      }
      if (auto r = dynamic_cast<RooRealProxy *>(p)) {
         if (isLiteralConstVar(r->arg()))
            elem[k->second] << r->arg().getVal();
         else
            elem[k->second] << r->arg().GetName();
      }
   }

   // export all the servers of a given RooAbsArg
   for (RooAbsArg *s : func.servers()) {
      this->exportObject(*s, exportedObjectNames);
   }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
// importing functions
void RooJSONFactoryWSTool::importFunction(const JSONNode &p, bool importAllDependants)
{
   auto const &importers = RooFit::JSONIO::importers();
   auto const &factoryExpressions = RooFit::JSONIO::importExpressions();

   // some preparations: what type of function are we dealing with here?
   std::string name(RooJSONFactoryWSTool::name(p));

   // if the RooAbsArg already exists, we don't need to do anything
   if (_workspace.arg(name)) {
      return;
   }
   // if the key we found is not a map, it's an error
   if (!p.is_map()) {
      std::stringstream ss;
      ss << "RooJSONFactoryWSTool() function node " + name + " is not a map!";
      logInputArgumentsError(std::move(ss));
      return;
   }
   std::string prefix = genPrefix(p, true);
   if (!prefix.empty())
      name = prefix + name;
   if (!p.has_child("type")) {
      std::stringstream ss;
      ss << "RooJSONFactoryWSTool() no type given for function '" << name << "', skipping." << std::endl;
      logInputArgumentsError(std::move(ss));
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
            logInputArgumentsError(std::move(ss));
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
               "https://github.com/root-project/root/blob/master/roofit/hs3/README.md to see "
               "how to do this!"
            << std::endl;
         logInputArgumentsError(std::move(ss));
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

void RooJSONFactoryWSTool::importFunction(const std::string &jsonString, bool importAllDependants)
{
   this->importFunction((JSONTree::create(jsonString))->rootnode(), importAllDependants);
}

void RooJSONFactoryWSTool::exportHisto(RooArgSet const &vars, std::size_t n, double const *contents, JSONNode &output)
{
   auto &observablesNode = output["axes"].set_seq();
   // axes have to be ordered to get consistent bin indices
   for (auto *var : static_range_cast<RooRealVar *>(vars)) {
      JSONNode &obsNode = observablesNode.append_child().set_map();
      obsNode["name"] << var->GetName();
      if (var->getBinning().isUniform()) {
         obsNode["min"] << var->getMin();
         obsNode["max"] << var->getMax();
         obsNode["nbins"] << var->getBins();
      } else {
         auto &bounds = obsNode["bounds"];
         bounds.set_seq();
         double val = var->getBinning().binLow(0);
         bounds.append_child() << val;
         for (int i = 0; i < var->getBinning().numBins(); ++i) {
            val = var->getBinning().binHigh(i);
            bounds.append_child() << val;
         }
      }
   }

   return exportArray(n, contents, output["contents"]);
}

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

void RooJSONFactoryWSTool::exportCategory(RooAbsCategory const &cat, JSONNode &node)
{
   auto &labels = node["labels"].set_seq();
   auto &indices = node["indices"].set_seq();

   for (auto const &item : cat) {
      labels.append_child() << item.first;
      indices.append_child() << item.second;
   }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
// exporting combined data
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
   auto &child = getRooFitInternal(*_rootnodeOutput, "combined_datas").set_map()[data.GetName()].set_map();
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
   std::unique_ptr<TList> dataList{simPdf ? data.split(*simPdf, true) : data.split(*cat, true)};

   int i = 0;
   for (RooAbsData *absData : static_range_cast<RooAbsData *>(*dataList)) {
      absData->SetName((std::string(data.GetName()) + "_" + absData->GetName()).c_str());
      datamap.components[cat->lookupName(i)] = absData->GetName();
      this->exportData(*absData);
      ++i;
   }
   return datamap;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
// exporting data
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
   // in ROOT 6.30: until then, the weight variable of the observerd dataset,
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
   auto *weights = data.isWeighted() ? &output["weights"].set_seq() : nullptr;
   for (int i = 0; i < data.numEntries(); ++i) {
      data.get(i);
      coords.append_child().fill_seq(variables, [](auto x) { return static_cast<RooRealVar *>(x)->getVal(); });
      if (weights)
         weights->append_child() << data.weight();
   }
}

std::unique_ptr<RooDataHist> RooJSONFactoryWSTool::readBinnedData(const JSONNode &n, const std::string &name)
{
   RooArgList varlist;

   for (JSONNode const &nd : n["axes"].children()) {
      if (nd.has_child("bounds")) {
         std::vector<double> bounds;
         for (auto const &bound : nd["bounds"].children()) {
            bounds.push_back(bound.val_double());
         }
         auto obs = std::make_unique<RooRealVar>(nd["name"].val().c_str(), nd["name"].val().c_str(), bounds[0],
                                                 bounds[bounds.size() - 1]);
         RooBinning bins(obs->getMin(), obs->getMax());
         ;
         for (auto b : bounds) {
            bins.addBoundary(b);
         }
         obs->setBinning(bins);
         varlist.addOwned(std::move(obs));
      } else {
         auto obs = std::make_unique<RooRealVar>(nd["name"].val().c_str(), nd["name"].val().c_str(),
                                                 nd["min"].val_double(), nd["max"].val_double());
         obs->setBins(nd["nbins"].val_int());
         varlist.addOwned(std::move(obs));
      }
   }

   return readBinnedData(n, name, varlist);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
// reading binned data
std::unique_ptr<RooDataHist>
RooJSONFactoryWSTool::readBinnedData(const JSONNode &n, const std::string &name, RooArgList const &varlist)
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

   auto bins = generateBinIndices(varlist);
   if (contents.num_children() != bins.size()) {
      std::stringstream errMsg;
      errMsg << "inconsistent bin numbers: contents=" << contents.num_children() << ", bins=" << bins.size();
      RooJSONFactoryWSTool::error(errMsg.str());
   }
   auto dh = std::make_unique<RooDataHist>(name.c_str(), name.c_str(), varlist);
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

///////////////////////////////////////////////////////////////////////////////////////////////////////
// importing variable
void RooJSONFactoryWSTool::importVariable(const JSONNode &p)
{
   // import a RooRealVar object
   std::string name(RooJSONFactoryWSTool::name(p));
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

///////////////////////////////////////////////////////////////////////////////////////////////////////
// import all dependants (servers) of a node
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
      RooMsgService::instance().log(nullptr, RooFit::MsgLevel::WARNING, RooFit::IO)
         << "RooFitHS3 only supports ModelConfigs with RooSimultaneous! Skipping ModelConfig.\n";
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

   analysisNode["domains"].set_seq().append_child() << "default_domain";

   analysisNode["likelihood"] << analysisName;

   auto &nllNode = appendNamedChild(rootnode["likelihoods"], analysisName);
   nllNode["distributions"].set_seq();
   nllNode["data"].set_seq();

   for (auto const &item : pdf->indexCat()) {
      nllNode["distributions"].append_child() << pdf->getPdf(item.first)->GetName();
      if (dataComponents) {
         const auto &d = dataComponents->find(item.first);
         nllNode["data"].append_child() << d->second;
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

   auto &modelConfigAux = getRooFitInternal(rootnode, "ModelConfigs", analysisName);
   modelConfigAux.set_map();
   modelConfigAux["pdfName"] << pdf->GetName();
   modelConfigAux["mcName"] << mc.GetName();
}

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
   for (RooAbsPdf *p : allpdfs) {
      this->exportObject(*p, exportedObjectNames);
   }

   // export attributes of exported objects
   for (std::string const &name : exportedObjectNames) {
      exportAttributes(_workspace.arg(name), n);
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
      if (data.components.size() > 0)
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
         if (exportedObjectNames.find(arg->GetName()) != exportedObjectNames.end())
            snapshotSorted.add(*arg);
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

bool RooJSONFactoryWSTool::importJSONfromString(const std::string &s)
{
   // import the workspace from JSON
   std::stringstream ss(s);
   return importJSON(ss);
}

bool RooJSONFactoryWSTool::importYMLfromString(const std::string &s)
{
   // import the workspace from YML
   std::stringstream ss(s);
   return importYML(ss);
}

std::string RooJSONFactoryWSTool::exportJSONtoString()
{
   // export the workspace to JSON
   std::stringstream ss;
   exportJSON(ss);
   return ss.str();
}

std::string RooJSONFactoryWSTool::exportYMLtoString()
{
   // export the workspace to YML
   std::stringstream ss;
   exportYML(ss);
   return ss.str();
}

/// Create a new JSON tree with version information.
std::unique_ptr<JSONTree> RooJSONFactoryWSTool::createNewJSONTree()
{
   std::unique_ptr<JSONTree> tree = JSONTree::create();
   JSONNode &n = tree->rootnode();
   n.set_map();
   auto &metadata = n["metadata"];
   metadata.set_map();

   // Bump to 0.2.0 once the HS3 v2 standard is final
   metadata["hs3_version"] << "0.1.90";

   // Add information about the ROOT version that was used to generate this file
   auto &rootInfo = appendNamedChild(metadata["packages"], "ROOT");
   std::string versionName = gROOT->GetVersion();
   // We want to consistently use dots such that the version name can be easily
   // digested automatically.
   std::replace(versionName.begin(), versionName.end(), '/', '.');
   rootInfo["version"] << versionName;

   return tree;
}

bool RooJSONFactoryWSTool::exportJSON(std::ostream &os)
{
   // export the workspace in JSON
   std::unique_ptr<JSONTree> tree = createNewJSONTree();
   JSONNode &n = tree->rootnode();
   this->exportAllObjects(n);
   n.writeJSON(os);
   return true;
}
bool RooJSONFactoryWSTool::exportJSON(std::string const &filename)
{
   // export the workspace in JSON
   std::ofstream out(filename.c_str());
   if (!out.is_open()) {
      std::stringstream ss;
      ss << "RooJSONFactoryWSTool() invalid output file '" << filename << "'." << std::endl;
      logInputArgumentsError(std::move(ss));
      return false;
   }
   return this->exportJSON(out);
}

bool RooJSONFactoryWSTool::exportYML(std::ostream &os)
{
   // export the workspace in YML
   std::unique_ptr<JSONTree> tree = createNewJSONTree();
   JSONNode &n = tree->rootnode();
   this->exportAllObjects(n);
   n.writeYML(os);
   return true;
}
bool RooJSONFactoryWSTool::exportYML(std::string const &filename)
{
   // export the workspace in YML
   std::ofstream out(filename.c_str());
   if (!out.is_open()) {
      std::stringstream ss;
      ss << "RooJSONFactoryWSTool() invalid output file '" << filename << "'." << std::endl;
      logInputArgumentsError(std::move(ss));
      return false;
   }
   return this->exportYML(out);
}

void RooJSONFactoryWSTool::importAllNodes(const JSONNode &n)
{
   _domains = std::make_unique<RooFit::JSONIO::Detail::Domains>();
   if (auto domains = n.find("domains"))
      _domains->readJSON(*domains);

   _rootnodeInput = &n;

   _attributesNode = findRooFitInternal(*_rootnodeInput, "attributes");

   this->importDependants(n);

   if (auto paramPointsNode = n.find("parameter_points")) {
      for (const auto &snsh : paramPointsNode->children()) {
         std::string name = RooJSONFactoryWSTool::name(snsh);
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
   // that case, we will remove the components from the "datas" vector so they
   // don't get imported.
   std::vector<std::unique_ptr<RooAbsData>> datas;
   if (auto dataNode = n.find("data")) {
      for (const auto &p : dataNode->children()) {
         datas.push_back(loadData(p, _workspace));
      }
   }

   // Now, read in analyses and likelihoods if there are any

   if (auto analysesNode = n.find("analyses")) {
      for (JSONNode const &analysisNode : analysesNode->children()) {
         importAnalysis(*_rootnodeInput, analysisNode, n["likelihoods"], _workspace, datas);
      }
   }

   combineDatasets(*_rootnodeInput, datas);

   for (auto const &d : datas) {
      if (d)
         _workspace.import(*d);
   }

   _rootnodeInput = nullptr;
   _domains.reset();
}

bool RooJSONFactoryWSTool::importJSON(std::istream &is)
{
   // import a JSON file to the workspace
   std::unique_ptr<JSONTree> tree = JSONTree::create(is);
   this->importAllNodes(tree->rootnode());
   return true;
}

bool RooJSONFactoryWSTool::importJSON(std::string const &filename)
{
   // import a JSON file to the workspace
   std::ifstream infile(filename.c_str());
   if (!infile.is_open()) {
      std::stringstream ss;
      ss << "RooJSONFactoryWSTool() invalid input file '" << filename << "'." << std::endl;
      logInputArgumentsError(std::move(ss));
      return false;
   }
   return this->importJSON(infile);
}

bool RooJSONFactoryWSTool::importYML(std::istream &is)
{
   // import a YML file to the workspace
   std::unique_ptr<JSONTree> tree = JSONTree::create(is);
   this->importAllNodes(tree->rootnode());
   return true;
}
bool RooJSONFactoryWSTool::importYML(std::string const &filename)
{
   // import a YML file to the workspace
   std::ifstream infile(filename.c_str());
   if (!infile.is_open()) {
      std::stringstream ss;
      ss << "RooJSONFactoryWSTool() invalid input file '" << filename << "'." << std::endl;
      logInputArgumentsError(std::move(ss));
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

std::ostream &RooJSONFactoryWSTool::log(int level)
{
   return RooMsgService::instance().log(nullptr, static_cast<RooFit::MsgLevel>(level), RooFit::IO);
}

void RooJSONFactoryWSTool::error(const char *s)
{
   RooMsgService::instance().log(nullptr, RooFit::MsgLevel::ERROR, RooFit::IO) << s << std::endl;
   throw std::runtime_error(s);
}
