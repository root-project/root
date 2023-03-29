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
#include <RooAbsCategory.h>
#include <RooRealProxy.h>
#include <RooListProxy.h>
#include <RooAbsProxy.h>
#include <RooCategory.h>
#include <RooDataSet.h>
#include <RooDataHist.h>
#include <RooSimultaneous.h>
#include <RooFit/ModelConfig.h>

#include "Domains.h"

#include "TH1.h"

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

using RooFit::Detail::JSONNode;
using RooFit::JSONIO::Detail::Domains;

namespace {

struct Var {
   int nbins;
   double min;
   double max;
   std::vector<double> bounds;

   Var(int n) : nbins(n), min(0), max(n) {}
   Var(const RooFit::Detail::JSONNode &val);
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
   if (val.is_map()) {
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
   } else if (val.is_seq()) {
      for (size_t i = 0; i < val.num_children(); ++i) {
         this->bounds.push_back(val[i].val_double());
      }
      this->nbins = this->bounds.size();
      this->min = this->bounds[0];
      this->max = this->bounds[this->nbins - 1];
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
         arg->setStringAttribute(RooJSONFactoryWSTool::name(attr).c_str(), attr.val().c_str());
      }
   }
   if (auto seq = node.find("tags")) {
      for (const auto &attr : seq->children()) {
         arg->setAttribute(attr.val().c_str());
      }
   }
}

bool checkRegularBins(const TAxis &ax)
{
   double w = ax.GetXmax() - ax.GetXmin();
   double bw = w / ax.GetNbins();
   for (int i = 0; i <= ax.GetNbins(); ++i) {
      if (std::abs(ax.GetBinUpEdge(i) - (ax.GetXmin() + (bw * i))) > w * 1e-6)
         return false;
   }
   return true;
}

inline void writeAxis(JSONNode &bounds, const TAxis &ax)
{
   bool regular = (!ax.IsVariableBinSize()) || checkRegularBins(ax);
   if (regular) {
      bounds.set_map();
      bounds["nbins"] << ax.GetNbins();
      bounds["min"] << ax.GetXmin();
      bounds["max"] << ax.GetXmax();
   } else {
      bounds.set_seq();
      for (int i = 0; i <= ax.GetNbins(); ++i) {
         bounds.append_child() << ax.GetBinUpEdge(i);
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
   for (auto k : ex.arguments) {
      expression << (first ? "::" + RooJSONFactoryWSTool::name(p) + "(" : ",");
      first = false;
      if (k == "true" || k == "false") {
         expression << (k == "true" ? "1" : "0");
      } else if (p[k].is_seq()) {
         bool f = true;
         for (RooAbsArg *arg : tool->requestArgList<RooAbsReal>(p, p[k].key())) {
            expression << (f ? "{" : ",") << arg->GetName();
            f = false;
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
      vars_numbins.push_back(absv->numBins());
   }
   std::vector<int> curr_comb(vars.size());
   ::genIndicesHelper(combinations, curr_comb, vars_numbins, 0);
   return combinations;
}

void exportAttributes(const RooAbsArg *arg, JSONNode &rootnode)
{
   JSONNode *node = nullptr;

   auto initializeNode = [&]() {
      if (node)
         return;

      node =
         &RooJSONFactoryWSTool::appendNamedChild(rootnode.get("misc", "ROOT_internal", "attributes"), arg->GetName());
   };

   // We have to remember if the variable was a constant RooRealVar or a
   // RooConstVar in RooFit to reconstruct the workspace correctly. The HS3
   // standard does not make this distinction.
   if (dynamic_cast<RooConstVar const *>(arg)) {
      initializeNode();
      (*node)["is_const_var"] << 1;
   }

   // export all string attributes of an object
   if (!arg->stringAttributes().empty()) {
      for (const auto &it : arg->stringAttributes()) {
         initializeNode();
         auto &dict = (*node)["dict"];
         dict.set_map();
         dict[it.first] << it.second;
      }
   }
   if (!arg->attributes().empty()) {
      initializeNode();
      (*node)["tags"].fill_seq(arg->attributes());
   }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
// read observables
std::map<std::string, Var> readObservables(const JSONNode &node)
{
   std::map<std::string, Var> vars;

   for (const auto &p : node["axes"].children()) {
      vars.emplace(RooJSONFactoryWSTool::name(p), Var(p));
   }

   return vars;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
// create several observables
void getObservables(RooWorkspace const &ws, const JSONNode &node, RooArgSet &out)
{
   auto vars = readObservables(node);
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
   if (name.empty() || !p.is_map())
      return nullptr;
   if (p.has_child("contents")) {
      // binned
      return RooJSONFactoryWSTool::readBinnedData(p, name);
   } else if (p.has_child("entries")) {
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
      for (size_t i = 0; i < coords.num_children(); ++i) {
         auto &point = coords[i];
         if (!point.is_seq()) {
            std::stringstream errMsg;
            errMsg << "coordinate point '" << i << "' is not a list!";
            RooJSONFactoryWSTool::error(errMsg.str());
         }
         if (point.num_children() != varlist.size()) {
            RooJSONFactoryWSTool::error("inconsistent number of entries and observables!");
         }
         for (size_t j = 0; j < point.num_children(); ++j) {
            auto *v = static_cast<RooRealVar *>(varlist.at(j));
            v->setVal(point[j].val_double());
         }
         data->add(vars, weights[i].val_double());
      }
      return data;
   }

   std::stringstream ss;
   ss << "RooJSONFactoryWSTool() failed to create dataset " << name << std::endl;
   logInputArgumentsError(std::move(ss));
   return nullptr;
}

void importAnalysis(const RooFit::Detail::JSONNode &rootnode, const RooFit::Detail::JSONNode &analysisNode,
                    const RooFit::Detail::JSONNode &likelihoodsNode, RooWorkspace &workspace)
{
   // if this is a toplevel pdf, also create a modelConfig for it
   std::string mcname = "ModelConfig";
   {
      RooStats::ModelConfig mc{mcname.c_str(), analysisNode["name"].val().c_str()};
      workspace.import(mc);
   }

   std::vector<std::string> nllDistNames;
   std::vector<std::string> nllDataNames;

   auto *nllNode = RooJSONFactoryWSTool::findNamedChild(likelihoodsNode, analysisNode["likelihood"].val());
   if (!nllNode) {
      throw std::runtime_error("likelihood node not found!");
   }
   for (auto &nameNode : (*nllNode)["distributions"].children()) {
      nllDistNames.push_back(nameNode.val());
   }
   for (auto &nameNode : (*nllNode)["data"].children()) {
      nllDataNames.push_back(nameNode.val());
   }

   std::string const &pdfName = analysisNode["name"].val();

   std::vector<std::string> channelNames;
   for (auto &n : rootnode.find("misc", "ROOT_internal", "channel_names", pdfName)->children()) {
      channelNames.push_back(n.val());
   }

   std::stringstream ss;
   ss << "SIMUL::" << pdfName << "(channelCat[";
   for (std::size_t iChannel = 0; iChannel < nllDistNames.size(); ++iChannel) {
      ss << channelNames[iChannel] << "=" << iChannel;
      if (iChannel < nllDistNames.size() - 1) {
         ss << ",";
      }
   }
   ss << "]";

   for (std::size_t iChannel = 0; iChannel < nllDistNames.size(); ++iChannel) {
      ss << ", " << channelNames[iChannel] << "=" << nllDistNames[iChannel];
   }
   ss << ")";
   auto pdf = static_cast<RooSimultaneous *>(workspace.factory(ss.str()));
   if (!pdf) {
      throw std::runtime_error("unable to import simultaneous pdf!");
   }

   auto &mc = *static_cast<RooStats::ModelConfig *>(workspace.obj(mcname));
   mc.SetWS(workspace);
   mc.SetPdf(*pdf);
   RooArgSet observables;
   for (auto const &child : analysisNode["variables"].children()) {
      if (auto var = workspace.var(child.val())) {
         observables.add(*var);
      }
   }
   RooArgSet nps;
   RooArgSet pois;
   for (auto const &child : analysisNode["pois"].children()) {
      if (auto var = workspace.var(child.val())) {
         pois.add(*var);
      }
   }
   RooArgSet globs;
   std::unique_ptr<RooArgSet> pdfVars{pdf->getVariables()};
   for (auto &var : workspace.allVars()) {
      if (!pdfVars->find(*var))
         continue;
      if (var->getAttribute("np")) {
         nps.add(*var, true);
      }
      if (var->getAttribute("glob")) {
         globs.add(*var, true);
      }
   }
   mc.SetObservables(observables);
   mc.SetParametersOfInterest(pois);
   mc.SetNuisanceParameters(nps);
   mc.SetGlobalObservables(globs);

   auto *mcAuxNode = rootnode.find("misc", "ROOT_internal", "ModelConfigs", analysisNode["name"].val());

   if (mcAuxNode == nullptr || !mcAuxNode->has_child("combined_data_name")) {
      throw std::runtime_error("Any imported ModelConfig must have the combined_data_name attribute (for now)!");
   }
   std::string name = (*mcAuxNode)["combined_data_name"].val();

   pdf->setStringAttribute("combined_data_name", name.c_str());
}

void combineDatasets(const RooFit::Detail::JSONNode &rootnode, std::vector<std::unique_ptr<RooAbsData>> &datas)
{
   auto *combinedDataInfoNode = rootnode.find("misc", "ROOT_internal", "combined_datasets");

   // If there is no info on combining datasets
   if (combinedDataInfoNode == nullptr) {
      return;
   }

   for (auto &info : combinedDataInfoNode->children()) {

      // parse the information
      std::string combinedName = info["name"].val();
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

} // namespace

RooJSONFactoryWSTool::RooJSONFactoryWSTool(RooWorkspace &ws) : _workspace{ws} {}

RooJSONFactoryWSTool::~RooJSONFactoryWSTool() {}

bool RooJSONFactoryWSTool::Config::stripObservables = true;

JSONNode &RooJSONFactoryWSTool::appendNamedChild(JSONNode &node, std::string const &name)
{
   node.set_seq();
   JSONNode &child = node.append_child();
   child.set_map();
   child["name"] << name;
   return child;
}

JSONNode const *RooJSONFactoryWSTool::findNamedChild(JSONNode const &node, std::string const &name)
{
   for (JSONNode const &child : node.children()) {
      if (child["name"].val() == name)
         return &child;
   }
   return nullptr;
}

JSONNode &RooJSONFactoryWSTool::makeVariablesNode(JSONNode &rootNode)
{
   JSONNode &container = appendNamedChild(rootNode["parameter_points"], "default_values");
   JSONNode &list = container["parameters"];
   list.set_seq();
   return list;
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
         this->importFunction(*child, false);
         if (RooAbsReal *retval = _workspace.function(objname))
            return retval;
      }
   }
   return nullptr;
}

void RooJSONFactoryWSTool::writeObservables(const TH1 &h, JSONNode &n, const std::vector<std::string> &varnames)
{
   auto &observables = n["axes"];
   auto &x = appendNamedChild(observables, varnames[0]);
   writeAxis(x, *h.GetXaxis());
   if (h.GetDimension() > 1) {
      auto &y = appendNamedChild(observables, varnames[1]);
      writeAxis(y, *(h.GetYaxis()));
      if (h.GetDimension() > 2) {
         auto &z = appendNamedChild(observables, varnames[2]);
         writeAxis(z, *(h.GetZaxis()));
      }
   }
}

void RooJSONFactoryWSTool::exportHistogram(const TH1 &h, JSONNode &n, const std::vector<std::string> &varnames,
                                           const TH1 *errH, bool writeObservables, bool writeErrors)
{
   n.set_map();
   auto &weights = n["contents"];
   weights.set_seq();
   JSONNode *errors = nullptr;
   if (writeErrors) {
      errors = &n["errors"];
      errors->set_seq();
   }
   if (writeObservables) {
      RooJSONFactoryWSTool::writeObservables(h, n, varnames);
   }
   const int nBins = h.GetNbinsX() * h.GetNbinsY() * h.GetNbinsZ();
   for (int i = 1; i <= nBins; ++i) {
      const double val = h.GetBinContent(i);
      weights.append_child() << val;
      if (writeErrors) {
         const double err = errH ? val * errH->GetBinContent(i) : h.GetBinError(i);
         errors->append_child() << err;
      }
   }
}

void RooJSONFactoryWSTool::exportVariable(const RooAbsArg *v, JSONNode &n)
{
   auto *cv = dynamic_cast<const RooConstVar *>(v);
   auto *rrv = dynamic_cast<const RooRealVar *>(v);
   if (!cv && !rrv)
      return;

   // for RooConstVar, if name and value are the same, we don't need to do anything
   if (cv && strcmp(cv->GetName(), TString::Format("%g", cv->getVal()).Data()) == 0) {
      return;
   }

   JSONNode &var = appendNamedChild(n, v->GetName());

   if (cv) {
      var["value"] << cv->getVal();
      var["const"] << true;
   } else if (rrv) {
      var["value"] << rrv->getVal();
      if (rrv->isConstant()) {
         var["const"] << rrv->isConstant();
      }
      if (rrv->numBins() != 100) {
         var["nbins"] << rrv->numBins();
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

void RooJSONFactoryWSTool::writeChannelNames(JSONNode &rootnode, std::string const &simPdfName,
                                             std::vector<std::string> const &channelNames)
{
   auto &categoriesinfo = rootnode.get("misc", "ROOT_internal", "channel_names");
   categoriesinfo.set_map();
   // Avoid repeated filling
   if (!categoriesinfo.has_child(simPdfName)) {
      auto &catinfo = categoriesinfo[simPdfName];
      catinfo.fill_seq(channelNames);
   }
}

JSONNode *RooJSONFactoryWSTool::exportObject(const RooAbsArg *func)
{
   if (auto simPdf = dynamic_cast<RooSimultaneous const *>(func)) {
      // RooSimultaneous is not used in the HS3 standard, we only export the
      // dependents and some ROOT internal information.
      RooJSONFactoryWSTool::exportDependants(func);

      std::vector<std::string> channelNames;
      for (auto const &item : simPdf->indexCat()) {
         channelNames.push_back(item.first);
      }
      writeChannelNames(*_rootnodeOutput, simPdf->GetName(), channelNames);

      return nullptr;
   } else if (dynamic_cast<RooAbsCategory const *>(func)) {
      // categories are created by the respective RooSimultaneous, so we're skipping the export here
      return nullptr;
   } else if (dynamic_cast<RooRealVar const *>(func) || dynamic_cast<RooConstVar const *>(func)) {
      // for variables, skip it because they are all exported in the beginning
      return nullptr;
   }

   auto &collectionNode = (*_rootnodeOutput)[dynamic_cast<RooAbsPdf const *>(func) ? "distributions" : "functions"];

   const std::string name = func->GetName();

   // if this element already exists, skip
   if (auto child = findNamedChild(collectionNode, name))
      return const_cast<JSONNode *>(child);

   auto const &exporters = RooFit::JSONIO::exporters();
   auto const &exportKeys = RooFit::JSONIO::exportKeys();

   TClass *cl = func->IsA();

   auto &elem = appendNamedChild(collectionNode, name);

   auto it = exporters.find(cl);
   if (it != exporters.end()) { // check if we have a specific exporter available
      for (auto &exp : it->second) {
         if (!exp->exportObject(this, func, elem)) {
            // The exporter might have messed with the content of the node
            // before failing. That's why we clear it and only reset the name.
            elem.clear();
            elem.set_map();
            elem["name"] << name;
            continue;
         }
         if (exp->autoExportDependants()) {
            RooJSONFactoryWSTool::exportDependants(func);
         }
         // Exporting the dependants will invalidate the iterator in "elem". So
         // instead of returning elem, we have to find again the element with
         // the right name.
         return const_cast<JSONNode *>(findNamedChild(collectionNode, name));
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
      return nullptr;
   }

   elem["type"] << dict->second.type;

   size_t nprox = func->numProxies();
   for (size_t i = 0; i < nprox; ++i) {
      RooAbsProxy *p = func->getProxy(i);

      std::string pname(p->name());
      if (pname[0] == '!')
         pname.erase(0, 1);

      auto k = dict->second.proxies.find(pname);
      if (k == dict->second.proxies.end()) {
         std::cerr << "failed to find key matching proxy '" << pname << "' for type '" << dict->second.type
                   << "', skipping" << std::endl;
         return nullptr;
      }

      if (auto l = dynamic_cast<RooListProxy *>(p)) {
         elem[k->second].fill_seq(*l, [](auto const &e) { return e->GetName(); });
      }
      if (auto r = dynamic_cast<RooRealProxy *>(p)) {
         elem[k->second] << r->arg().GetName();
      }
   }

   RooJSONFactoryWSTool::exportDependants(func);

   // Exporting the dependants will invalidate the iterator in "elem". So
   // instead of returning elem, we have to find again the element with the
   // right name.
   return const_cast<JSONNode *>(findNamedChild(collectionNode, name));
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
// importing functions
void RooJSONFactoryWSTool::importFunction(const JSONNode &p, bool isPdf)
{
   auto const &importers = RooFit::JSONIO::importers();
   auto const &pdfFactoryExpressions = RooFit::JSONIO::pdfImportExpressions();
   auto const &funcFactoryExpressions = RooFit::JSONIO::functionImportExpressions();

   // some preparations: what type of function are we dealing with here?
   std::string name(RooJSONFactoryWSTool::name(p));

   // if the function already exists, we don't need to do anything
   if ((isPdf && _workspace.pdf(name)) || _workspace.function(name)) {
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
   this->importDependants(p);

   try {
      // check for specific implementations
      auto it = importers.find(functype);
      bool ok = false;
      if (it != importers.end()) {
         for (auto &imp : it->second) {
            ok = isPdf ? imp->importPdf(this, p) : imp->importFunction(this, p);
            if (ok)
               break;
         }
      }
      if (!ok) { // generic import using the factory expressions
         auto expr = isPdf ? pdfFactoryExpressions.find(functype) : funcFactoryExpressions.find(functype);
         if (expr != (isPdf ? pdfFactoryExpressions.end() : funcFactoryExpressions.end())) {
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
   } catch (const RooJSONFactoryWSTool::DependencyMissingError &ex) {
      throw ex;
   }
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

   if (cat) {
      // this is a combined dataset
      auto &child =
         appendNamedChild(_rootnodeOutput->get("misc", "ROOT_internal", "combined_datasets"), data.GetName());
      child["index_cat"] << cat->GetName();
      auto &labels = child["labels"];
      labels.set_seq();
      auto &indices = child["indices"];
      indices.set_seq();

      std::unique_ptr<TList> dataList{data.split(*cat, true)};
      for (RooAbsData *absData : static_range_cast<RooAbsData *>(*dataList)) {
         labels.append_child() << absData->GetName();
         indices.append_child() << cat->lookupIndex(absData->GetName());
         absData->SetName((std::string(data.GetName()) + "_" + absData->GetName()).c_str());
         this->exportData(*absData);
      }

      return;
   }

   JSONNode &output = appendNamedChild((*_rootnodeOutput)["data"], data.GetName());

   // this is a binned dataset
   if (dynamic_cast<RooDataHist const *>(&data)) {

      auto &observablesNode = output["axes"];
      for (RooRealVar *var : static_range_cast<RooRealVar *>(*data.get())) {
         auto &observableNode = appendNamedChild(observablesNode, var->GetName());
         observableNode["min"] << var->getMin();
         observableNode["max"] << var->getMax();
         observableNode["nbins"] << var->numBins();
      }

      auto &weights = output["contents"];
      weights.set_seq();
      for (int i = 0; i < data.numEntries(); ++i) {
         double w = static_cast<RooDataHist const &>(data).weight(i);
         // To make sure there are no unnecessary floating points in the JSON
         if (int(w) == w) {
            weights.append_child() << int(w);
         } else {
            weights.append_child() << w;
         }
      }
      return;
   }

   // this is a regular unbinned dataset
   bool singlePoint = (data.numEntries() <= 1);
   RooArgSet reduced_obs;
   if (Config::stripObservables) {
      if (!singlePoint) {
         std::map<RooRealVar *, std::vector<double>> obs_values;
         for (int i = 0; i < data.numEntries(); ++i) {
            data.get(i);
            for (RooRealVar *rv : static_range_cast<RooRealVar *>(*data.get())) {
               obs_values[rv].push_back(rv->getVal());
            }
         }
         for (auto &obs_it : obs_values) {
            auto &vals = obs_it.second;
            double v0 = vals[0];
            bool is_const_val = std::all_of(vals.begin(), vals.end(), [v0](double v) { return v == v0; });
            if (!is_const_val)
               reduced_obs.add(*(obs_it.first), true);
         }
      }
   } else {
      reduced_obs.add(*data.get());
   }
   if (!reduced_obs.empty()) {
      exportVariables(reduced_obs, output["axes"]);
   }
   auto &weights = singlePoint && Config::stripObservables ? output["contents"] : output["weights"];
   weights.set_seq();
   for (int i = 0; i < data.numEntries(); ++i) {
      data.get(i);
      if (!(Config::stripObservables && singlePoint)) {
         auto &coords = output["entries"];
         coords.set_seq();
         coords.append_child().fill_seq(reduced_obs, [](auto x) { return static_cast<RooRealVar *>(x)->getVal(); });
      }
      weights.append_child() << data.weight();
   }
}

std::unique_ptr<RooDataHist> RooJSONFactoryWSTool::readBinnedData(const JSONNode &n, const std::string &name)
{
   RooArgList varlist;

   for (JSONNode const &nd : n["axes"].children()) {
      const std::string nm = nd["name"].val();
      auto var = std::make_unique<RooRealVar>(nm.c_str(), nm.c_str(), nd["min"].val_double(), nd["max"].val_double());
      var->setBins(nd["nbins"].val_double());
      varlist.addOwned(std::move(var));
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
   // temporarily disable dirty flag propagation when filling the RDH
   std::vector<double> initVals;
   for (auto &v : varlist) {
      v->setDirtyInhibit(true);
      initVals.push_back(static_cast<RooAbsReal const *>(v)->getVal());
   }
   for (size_t ibin = 0; ibin < bins.size(); ++ibin) {
      for (size_t i = 0; i < bins[ibin].size(); ++i) {
         static_cast<RooRealVar *>(varlist.at(i))->setBin(bins[ibin][i]);
      }
      const double err = errors ? (*errors)[ibin].val_double() : -1;
      dh->add(varlist, contents[ibin].val_double(), err > 0 ? err * err : -1);
   }
   // re-enable dirty flag propagation
   for (size_t i = 0; i < varlist.size(); ++i) {
      auto v = static_cast<RooRealVar *>(varlist.at(i));
      v->setVal(initVals[i]);
      v->setDirtyInhibit(false);
   }
   return dh;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
// importing variables
void RooJSONFactoryWSTool::importVariables(const JSONNode &n)
{
   for (const auto &p : n.children()) {
      importVariable(p);
   }
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
      if (auto *attrNode = findNamedChild(*_attributesNode, name)) {
         // We should not create RooRealVar objects for RooConstVars!
         if (attrNode->has_child("is_const_var") && (*attrNode)["is_const_var"].val_int() == 1) {
            RooConstVar v(name.c_str(), name.c_str(), p["value"].val_double());
            wsImport(v);
            return;
         }
      }
   }
   RooRealVar v(name.c_str(), name.c_str(), 1.);
   configureVariable(*_domains, p, v);
   wsImport(v);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
// export all dependants (servers) of a RooAbsArg
void RooJSONFactoryWSTool::exportDependants(const RooAbsArg *source)
{
   // export all the servers of a given RooAbsArg
   std::vector<RooAbsArg *> servers;
   for (auto &s : source->servers()) {
      servers.push_back(s);
   }
   std::sort(servers.begin(), servers.end(), [](auto const &l, auto const &r) {
      if (l->dependsOn(*r))
         return true;
      return r->dependsOn(*l) ? false : strcmp(l->GetName(), r->GetName()) < 0;
   });
   for (auto s : servers) {
      this->exportObject(s);
   }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
// import all dependants (servers) of a node
void RooJSONFactoryWSTool::importDependants(const JSONNode &n)
{
   // import all the dependants of an object
   if (JSONNode const *varsNode = getVariablesNode(n)) {
      this->importVariables(*varsNode);
   }
   if (auto seq = n.find("functions")) {
      for (const auto &p : seq->children()) {
         importFunction(p, false);
      }
   }
   if (auto seq = n.find("distributions")) {
      for (const auto &p : seq->children()) {
         this->importFunction(p, true);
      }
   }
}

std::string RooJSONFactoryWSTool::name(const JSONNode &n)
{
   return n.is_container() && n.has_child("name") ? n["name"].val() : (n.has_key() ? n.key() : n.val());
}

void RooJSONFactoryWSTool::writeCombinedDataName(JSONNode &rootnode, std::string const &pdfName,
                                                 std::string const &dataName)
{
   auto &miscinfo = rootnode["misc"];
   miscinfo.set_map();
   auto &rootinfo = miscinfo["ROOT_internal"];
   rootinfo.set_map();
   auto &modelConfigs = rootinfo["ModelConfigs"];
   modelConfigs.set_map();
   auto &modelConfigAux = modelConfigs[pdfName];
   modelConfigAux.set_map();

   modelConfigAux["combined_data_name"] << dataName;
}

void RooJSONFactoryWSTool::exportModelConfig(JSONNode &rootnode, RooStats::ModelConfig const &mc)
{
   auto pdf = dynamic_cast<RooSimultaneous const *>(mc.GetPdf());
   if (pdf == nullptr) {
      throw std::runtime_error("RooFitHS3 only supports ModelConfigs with RooSimultaneous!");
   }

   JSONNode &analysisNode = appendNamedChild(rootnode["analyses"], pdf->GetName());

   auto &analysisDomains = analysisNode["domains"];
   analysisDomains.set_seq();
   analysisDomains.append_child() << "default_domain";

   analysisNode["likelihood"] << pdf->GetName();

   std::string basename;
   if (auto s = pdf->getStringAttribute("combined_data_name")) {
      basename = s;
   } else {
      throw std::runtime_error("Any exported RooSimultaneous must have the combined_data_name attribute (for now)!");
   }

   writeCombinedDataName(rootnode, pdf->GetName(), basename);

   auto &nllNode = appendNamedChild(rootnode["likelihoods"], pdf->GetName());
   nllNode["distributions"].set_seq();
   nllNode["data"].set_seq();

   for (auto const &item : pdf->indexCat()) {
      nllNode["distributions"].append_child() << pdf->getPdf(item.first)->GetName();
      nllNode["data"].append_child() << basename + "_" + item.first;
   }

   auto writeList = [&](const char *name, RooArgSet const *args) {
      if (args)
         analysisNode[name].fill_seq(*args, [](auto const &arg) { return arg->GetName(); });
   };

   writeList("variables", mc.GetObservables());
   writeList("pois", mc.GetParametersOfInterest());
   writeList("nps", mc.GetNuisanceParameters());
   writeList("globs", mc.GetGlobalObservables());
}

void RooJSONFactoryWSTool::exportAllObjects(JSONNode &n)
{
   _domains = std::make_unique<Domains>();

   // export all attributes
   for (RooAbsArg const *arg : _workspace.components()) {
      exportAttributes(arg, n);
   }

   // export all RooRealVars and RooConstVars
   {
      JSONNode &vars = makeVariablesNode(n);
      for (RooAbsArg *arg : _workspace.components()) {
         if (dynamic_cast<RooRealVar const *>(arg) || dynamic_cast<RooConstVar const *>(arg)) {
            exportVariable(arg, vars);
         }
      }
   }

   _rootnodeOutput = &n;
   // export all datasets
   std::vector<RooAbsData *> alldata;
   for (auto &d : _workspace.allData()) {
      alldata.push_back(d);
   }
   std::sort(alldata.begin(), alldata.end(), [](auto l, auto r) { return strcmp(l->GetName(), r->GetName()) < 0; });
   for (auto &d : alldata) {
      this->exportData(*d);
   }

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
   std::sort(allpdfs.begin(), allpdfs.end(), [](auto l, auto r) { return strcmp(l->GetName(), r->GetName()) < 0; });
   for (auto &p : allpdfs) {
      this->exportObject(p);
   }

   _rootnodeOutput = &n;
   // export all ModelConfig objects and attached Pdfs
   std::vector<RooStats::ModelConfig *> mcs;
   for (TObject *obj : _workspace.allGenericObjects()) {
      if (auto mc = dynamic_cast<RooStats::ModelConfig *>(obj)) {
         mcs.push_back(mc);
         exportModelConfig(n, *mcs.back());
      }
   }

   for (auto *snsh : static_range_cast<RooArgSet const *>(_workspace.getSnapshots())) {
      std::string name(snsh->GetName());
      if (name != "default_values") {
         this->exportVariables(*snsh, appendNamedChild(n["parameter_points"], name)["parameters"]);
      }
   }
   for (const auto &mc : mcs) {
      RooJSONFactoryWSTool::exportObject(mc->GetPdf());
   }
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
   n["metadata"].set_map();
   // The currently implemented HS3 standard is version 0.1
   n["metadata"]["version"] << "0.1.90";
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

void RooJSONFactoryWSTool::importAllNodes(const RooFit::Detail::JSONNode &n)
{
   _domains = std::make_unique<Domains>();
   if (n.has_child("domains"))
      _domains->readJSON(n["domains"]);

   _rootnodeInput = &n;

   _attributesNode = _rootnodeInput->find("misc", "ROOT_internal", "attributes");
   if (_attributesNode && !_attributesNode->is_seq()) {
      _attributesNode = nullptr;
   }

   this->importDependants(n);

   _workspace.saveSnapshot("fromJSON", _workspace.allVars());
   if (n.has_child("parameter_points")) {
      for (const auto &snsh : n["parameter_points"].children()) {
         std::string name = RooJSONFactoryWSTool::name(snsh);
         if (name == "fromJSON")
            continue;
         RooArgSet vars;
         for (const auto &var : snsh.children()) {
            if (RooRealVar *rrv = _workspace.var(RooJSONFactoryWSTool::name(var))) {
               configureVariable(*_domains, var, *rrv);
               vars.add(*rrv);
            }
         }
         _workspace.saveSnapshot(name, vars);
      }
   }
   _workspace.loadSnapshot("fromJSON");

   // Import attributes
   if (_attributesNode) {
      for (const auto &elem : _attributesNode->children()) {
         if (RooAbsArg *arg = _workspace.arg(elem["name"].val()))
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

   combineDatasets(*_rootnodeInput, datas);

   // Now, read in analyses and likelihoods if there are any
   if (auto analysesNode = n.find("analyses")) {
      for (JSONNode const &analysisNode : analysesNode->children()) {
         importAnalysis(*_rootnodeInput, analysisNode, n["likelihoods"], _workspace);
      }
   }

   _rootnodeInput = nullptr;
   _domains.reset();

   for (auto const &d : datas) {
      if (d)
         _workspace.import(*d);
   }
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

std::ostream &RooJSONFactoryWSTool::log(int level)
{
   return RooMsgService::instance().log(nullptr, static_cast<RooFit::MsgLevel>(level), RooFit::IO);
}

void RooJSONFactoryWSTool::error(const char *s)
{
   RooMsgService::instance().log(nullptr, RooFit::MsgLevel::ERROR, RooFit::IO) << s << std::endl;
   ;
   throw std::runtime_error(s);
}
