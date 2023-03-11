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
#include <RooStats/ModelConfig.h>

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

const RooFit::Detail::JSONNode &dereference_helper(const RooFit::Detail::JSONNode &n,
                                                   const std::vector<std::string> &keys,
                                                   const RooFit::Detail::JSONNode &default_val, size_t idx)
{
   if (idx >= keys.size())
      return n;
   if (n.has_child(keys[idx]))
      return dereference_helper(n[keys[idx]], keys, default_val, idx + 1);
   return default_val;
}

const RooFit::Detail::JSONNode &dereference(const RooFit::Detail::JSONNode &n, const std::vector<std::string> &keys,
                                            const RooFit::Detail::JSONNode &default_val)
{
   return dereference_helper(n, keys, default_val, 0);
}

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
   if (p.has_child("value"))
      v.setVal(p["value"].val_double());
   domains.writeVariable(v);
   if (p.has_child("nbins"))
      v.setBins(p["nbins"].val_int());
   if (p.has_child("relErr"))
      v.setError(v.getVal() * p["relErr"].val_double());
   if (p.has_child("err"))
      v.setError(p["err"].val_double());
   if (p.has_child("const"))
      v.setConstant(p["const"].val_bool());
   else
      v.setConstant(false);
}

} // namespace

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

JSONNode const *RooJSONFactoryWSTool::getVariablesNode(JSONNode const &rootNode)
{
   auto out = findNamedChild(rootNode["parameter_points"], "default_values");
   if (out == nullptr)
      return nullptr;
   return &((*out)["parameters"]);
}

RooJSONFactoryWSTool::RooJSONFactoryWSTool(RooWorkspace &ws) : _workspace{ws} {}

RooJSONFactoryWSTool::~RooJSONFactoryWSTool() {}

template <>
RooRealVar *RooJSONFactoryWSTool::requestImpl<RooRealVar>(const std::string &objname)
{
   if (RooRealVar *retval = _workspace.var(objname))
      return retval;
   if (JSONNode const *vars = getVariablesNode(*_rootnodeInput)) {
      if (vars->has_child(objname)) {
         this->importVariable((*vars)[objname]);
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
   if (_rootnodeInput->has_child("distributions")) {
      if (auto child = findNamedChild((*_rootnodeInput)["distributions"], objname)) {
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
   if (_rootnodeInput->has_child("functions")) {
      if (auto child = findNamedChild((*_rootnodeInput)["functions"], objname)) {
         this->importFunction(*child, false);
         if (RooAbsReal *retval = _workspace.function(objname))
            return retval;
      }
   }
   return nullptr;
}

namespace {

void logInputArgumentsError(std::stringstream &&ss)
{
   oocoutE(nullptr, InputArguments) << ss.str() << std::endl;
}

} // namespace

RooJSONFactoryWSTool::Var::Var(const JSONNode &val)
{
   if (val.is_map()) {
      if (!val.has_child("nbins"))
         this->nbins = 1;
      else
         this->nbins = val["nbins"].val_int();
      if (!val.has_child("min"))
         this->min = 0;
      else
         this->min = val["min"].val_double();
      if (!val.has_child("max"))
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

std::string RooJSONFactoryWSTool::genPrefix(const JSONNode &p, bool trailing_underscore)
{
   std::string prefix;
   if (!p.is_map())
      return prefix;
   if (p.has_child("namespaces")) {
      for (const auto &ns : p["namespaces"].children()) {
         if (!prefix.empty())
            prefix += "_";
         prefix += ns.val();
      }
   }
   if (trailing_underscore && !prefix.empty())
      prefix += "_";
   return prefix;
}

bool RooJSONFactoryWSTool::Config::stripObservables = true;

namespace {

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

void importAttributes(RooAbsArg *arg, JSONNode const &rootnode)
{
   JSONNode const &attributesNode = dereference(rootnode, {"misc", "ROOT_internal", "attributes"}, rootnode);

   // If the attributes node was not found, it will not be a sequence node
   if (!attributesNode.is_seq())
      return;

   JSONNode const *node = RooJSONFactoryWSTool::findNamedChild(attributesNode, arg->GetName());
   if (node == nullptr)
      return;

   if (node->has_child("dict")) {
      for (const auto &attr : (*node)["dict"].children()) {
         arg->setStringAttribute(RooJSONFactoryWSTool::name(attr).c_str(), attr.val().c_str());
      }
   }
   if (node->has_child("tags")) {
      for (const auto &attr : (*node)["tags"].children()) {
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
   std::string name(RooJSONFactoryWSTool::name(p));
   std::stringstream expression;
   std::string classname(ex.tclass->GetName());
   size_t colon = classname.find_last_of(":");
   if (colon < classname.size()) {
      expression << classname.substr(colon + 1);
   } else {
      expression << classname;
   }
   expression << "::" << name << "(";
   bool first = true;
   for (auto k : ex.arguments) {
      if (!first)
         expression << ",";
      first = false;
      if (k == "true") {
         expression << "1";
         continue;
      } else if (k == "false") {
         expression << "0";
         continue;
      } else if (!p.has_child(k)) {
         std::stringstream err;
         err << "factory expression for class '" << ex.tclass->GetName() << "', which expects key '" << k
             << "' missing from input for object '" << name << "', skipping.";
         RooJSONFactoryWSTool::error(err.str());
      }
      if (p[k].is_seq()) {
         expression << "{";
         bool f = true;
         for (const auto &x : p[k].children()) {
            if (!f)
               expression << ",";
            f = false;
            std::string obj(x.val());
            tool->request<RooAbsReal>(obj, name);
            expression << obj;
         }
         expression << "}";
      } else {
         std::string obj(p[k].val());
         tool->request<RooAbsReal>(obj, name);
         expression << obj;
      }
   }
   expression << ")";
   return expression.str();
}

} // namespace

std::vector<std::vector<int>> RooJSONFactoryWSTool::generateBinIndices(const RooArgList &vars)
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

void RooJSONFactoryWSTool::writeObservables(const TH1 &h, JSONNode &n, const std::vector<std::string> &varnames)
{
   auto &observables = n["axes"];
   auto &x = appendNamedChild(observables, varnames[0]);
   writeAxis(x, *(h.GetXaxis()));
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
   if (writeErrors) {
      n["errors"].set_seq();
   }
   if (writeObservables) {
      RooJSONFactoryWSTool::writeObservables(h, n, varnames);
   }
   for (int i = 1; i <= h.GetNbinsX(); ++i) {
      if (h.GetDimension() == 1) {
         weights.append_child() << h.GetBinContent(i);
         if (writeErrors) {
            n["errors"].append_child() << (errH ? h.GetBinContent(i) * errH->GetBinContent(i) : h.GetBinError(i));
         }
      } else {
         for (int j = 1; j <= h.GetNbinsY(); ++j) {
            if (h.GetDimension() == 2) {
               weights.append_child() << h.GetBinContent(i, j);
               if (writeErrors) {
                  n["errors"].append_child()
                     << (errH ? h.GetBinContent(i, j) * errH->GetBinContent(i, j) : h.GetBinError(i, j));
               }
            } else {
               for (int k = 1; k <= h.GetNbinsZ(); ++k) {
                  weights.append_child() << h.GetBinContent(i, j, k);
                  if (writeErrors) {
                     n["errors"].append_child()
                        << (errH ? h.GetBinContent(i, j, k) * errH->GetBinContent(i, j, k) : h.GetBinError(i, j, k));
                  }
               }
            }
         }
      }
   }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
// helper namespace
///////////////////////////////////////////////////////////////////////////////////////////////////////

namespace {

bool find(const JSONNode &n, const std::string &elem)
{
   // find an attribute
   if (n.is_seq()) {
      for (const auto &t : n.children()) {
         if (t.val() == elem)
            return true;
      }
      return false;
   } else if (n.is_map()) {
      return n.has_child(elem);
   }
   return false;
}

void append(JSONNode &n, const std::string &elem)
{
   // append an attribute
   n.set_seq();
   if (!find(n, elem)) {
      n.append_child() << elem;
   }
}

void exportAttributes(const RooAbsArg *arg, JSONNode &rootnode)
{
   JSONNode *node = nullptr;

   auto initializeNode = [&]() {
      if (node)
         return;

      auto &miscinfo = rootnode["misc"];
      miscinfo.set_map();
      auto &rootinfo = miscinfo["ROOT_internal"];
      rootinfo.set_map();
      auto &attributesNode = rootinfo["attributes"];
      node = &RooJSONFactoryWSTool::appendNamedChild(attributesNode, arg->GetName());
   };

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
      for (const auto &it : arg->attributes()) {
         initializeNode();
         append((*node)["tags"], it);
      }
   }
}

} // namespace

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

JSONNode *RooJSONFactoryWSTool::exportObject(const RooAbsArg *func)
{
   if (dynamic_cast<RooSimultaneous const *>(func)) {
      // RooSimultaneous is not used in the HS3 standard, we only export the dependents
      RooJSONFactoryWSTool::exportDependants(func);
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

   auto it = exporters.find(cl);
   if (it != exporters.end()) { // check if we have a specific exporter available
      auto &elem = appendNamedChild(collectionNode, name);
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
         return &elem;
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

   RooJSONFactoryWSTool::exportDependants(func);

   auto &elem = appendNamedChild(collectionNode, name);
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

   return &elem;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
// importing functions
void RooJSONFactoryWSTool::importFunctions(const JSONNode &n)
{
   // import a list of RooAbsReal objects
   for (const auto &p : n.children()) {
      importFunction(p, false);
   }
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
   if (isPdf) {
      if (_workspace.pdf(name))
         return;
   } else {
      if (_workspace.function(name))
         return;
   }
   // if the key we found is not a map, it's an error
   if (!p.is_map()) {
      std::stringstream ss;
      ss << "RooJSONFactoryWSTool() function node " + name + " is not a map!";
      logInputArgumentsError(std::move(ss));
      return;
   }
   std::string prefix = RooJSONFactoryWSTool::genPrefix(p, true);
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
// importing data
std::map<std::string, std::unique_ptr<RooAbsData>> RooJSONFactoryWSTool::loadData(const JSONNode &rootnode)
{
   std::map<std::string, std::unique_ptr<RooAbsData>> dataMap;

   if (!rootnode.has_child("data")) {
      return dataMap;
   }

   for (const auto &p : rootnode["data"].children()) {
      std::string name(RooJSONFactoryWSTool::name(p));
      if (name.empty())
         continue;
      if (_workspace.data(name))
         continue;
      if (!p.is_map())
         continue;
      if (p.has_child("contents")) {
         // binned
         dataMap[name] = this->readBinnedData(p, name);
      } else if (p.has_child("entries")) {
         // unbinned
         RooArgSet vars;
         this->getObservables(_workspace, p, name, vars);
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
         dataMap[name] = std::move(data);
      } else {
         std::stringstream ss;
         ss << "RooJSONFactoryWSTool() failed to create dataset " << name << std::endl;
         logInputArgumentsError(std::move(ss));
      }
   }
   return dataMap;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
// exporting data
void RooJSONFactoryWSTool::exportData(RooAbsData &data)
{
   // find category observables
   RooAbsCategory *cat = nullptr;
   for (RooAbsArg *obs : *data.get()) {
      if (dynamic_cast<RooAbsCategory *>(obs)) {
         if (cat) {
            RooJSONFactoryWSTool::error("dataset '" + std::string(data.GetName()) +
                                        " has several category observables!");
         } else {
            cat = static_cast<RooAbsCategory *>(obs);
         }
      }
   }

   if (cat) {
      // this is a composite dataset
      std::unique_ptr<TList> dataList{data.split(*(cat), true)};
      if (!dataList) {
         RooJSONFactoryWSTool::error("unable to split dataset '" + std::string(data.GetName()) + "' at '" +
                                     std::string(cat->GetName()) + "'");
      }
      for (RooAbsData *absData : static_range_cast<RooAbsData *>(*dataList)) {
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

///////////////////////////////////////////////////////////////////////////////////////////////////////
// create several observables
void RooJSONFactoryWSTool::getObservables(RooWorkspace &ws, const JSONNode &node, const std::string &obsnamecomp,
                                          RooArgSet &out)
{
   auto vars = RooJSONFactoryWSTool::readObservables(node, obsnamecomp);
   for (auto v : vars) {
      std::string name(v.first);
      if (ws.var(name)) {
         out.add(*ws.var(name));
      } else {
         out.add(*RooJSONFactoryWSTool::createObservable(ws, name, v.second));
      }
   }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
// create an observable
RooRealVar *
RooJSONFactoryWSTool::createObservable(RooWorkspace &ws, const std::string &name, const RooJSONFactoryWSTool::Var &var)
{
   ws.factory(name + "[" + std::to_string(var.min) + "]");
   RooRealVar *rrv = ws.var(name);
   rrv->setMin(var.min);
   rrv->setMax(var.max);
   rrv->setConstant(true);
   rrv->setBins(var.nbins);
   rrv->setAttribute("observable");
   return rrv;
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
RooJSONFactoryWSTool::readBinnedData(const JSONNode &n, const std::string &name, RooArgList varlist)
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

   auto bins = RooJSONFactoryWSTool::generateBinIndices(varlist);
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
// read observables
std::map<std::string, RooJSONFactoryWSTool::Var>
RooJSONFactoryWSTool::readObservables(const JSONNode &node, const std::string &obsnamecomp)
{
   std::map<std::string, RooJSONFactoryWSTool::Var> vars;
   if (!node.has_child("axes")) {
      vars.emplace("obs_x_" + obsnamecomp, RooJSONFactoryWSTool::Var(node["contents"].num_children()));
      return vars;
   }

   for (const auto &p : node["axes"].children()) {
      vars.emplace(RooJSONFactoryWSTool::name(p), RooJSONFactoryWSTool::Var(p));
   }

   return vars;
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
   RooRealVar v(name.c_str(), name.c_str(), 1.);
   configureVariable(*_domains, p, v);
   wsImport(v);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
// export all dependants (servers) of a RooAbsArg
void RooJSONFactoryWSTool::exportDependants(const RooAbsArg *source)
{
   // export all the servers of a given RooAbsArg
   auto servers(source->servers());
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
   if (n.has_child("functions")) {
      this->importFunctions(n["functions"]);
   }
   if (n.has_child("distributions")) {
      for (const auto &p : n["distributions"].children()) {
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

   // export all ModelConfig objects and attached Pdfs
   std::vector<RooStats::ModelConfig *> mcs;
   std::vector<RooAbsPdf *> toplevel;
   _rootnodeOutput = &n;
   for (TObject *obj : _workspace.allGenericObjects()) {
      if (auto mc = dynamic_cast<RooStats::ModelConfig *>(obj)) {
         mcs.push_back(mc);
         exportModelConfig(n, *mcs.back());
      }
   }
   for (RooAbsArg *pdf : _workspace.allPdfs()) {

      if (!pdf->hasClients() || pdf->getAttribute("toplevel")) {
         bool hasMC = false;
         for (const auto &mc : mcs) {
            if (mc->GetPdf() == pdf)
               hasMC = true;
         }
         if (!hasMC)
            toplevel.push_back(static_cast<RooAbsPdf *>(pdf));
      }
   }
   for (auto d : _workspace.allData()) {
      this->exportData(*d);
   }
   for (auto *snsh : static_range_cast<RooArgSet const *>(_workspace.getSnapshots())) {
      this->exportVariables(*snsh, appendNamedChild(n["parameter_points"], snsh->GetName())["parameters"]);
   }
   for (const auto &mc : mcs) {
      RooJSONFactoryWSTool::exportObject(mc->GetPdf());
   }
   for (const auto &pdf : toplevel) {
      auto *pdfNode = RooJSONFactoryWSTool::exportObject(pdf);
      if (!pdfNode)
         continue;
      if (!pdf->getAttribute("toplevel"))
         append((*pdfNode)["tags"], "toplevel");
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

void RooJSONFactoryWSTool::importAnalysis(const RooFit::Detail::JSONNode &analysisNode,
                                          const RooFit::Detail::JSONNode &likelihoodsNode,
                                          const RooFit::Detail::JSONNode &mcAuxNode)
{

   // if this is a toplevel pdf, also create a modelConfig for it
   std::string mcname = "ModelConfig";
   {
      RooStats::ModelConfig mc{mcname.c_str(), analysisNode["name"].val().c_str()};
      _workspace.import(mc);
   }

   std::vector<std::string> nllDistNames;
   std::vector<std::string> nllDataNames;

   auto *nllNode = findNamedChild(likelihoodsNode, analysisNode["likelihood"].val());
   if (!nllNode) {
      throw std::runtime_error("likelihood node not found!");
   }
   for (auto &nameNode : (*nllNode)["distributions"].children()) {
      nllDistNames.push_back(nameNode.val());
   }
   for (auto &nameNode : (*nllNode)["data"].children()) {
      nllDataNames.push_back(nameNode.val());
   }

   std::stringstream ss;
   ss << "SIMUL::" << analysisNode["name"].val() << "(channelCat[";
   for (std::size_t iChannel = 0; iChannel < nllDistNames.size(); ++iChannel) {
      ss << "channel_" << iChannel << "=" << iChannel;
      if (iChannel < nllDistNames.size() - 1) {
         ss << ",";
      }
   }
   ss << "]";

   for (std::size_t iChannel = 0; iChannel < nllDistNames.size(); ++iChannel) {
      ss << ", "
         << "channel_" << iChannel << "=" << nllDistNames[iChannel];
   }
   ss << ")";
   auto pdf = static_cast<RooSimultaneous *>(_workspace.factory(ss.str()));
   if (!pdf) {
      throw std::runtime_error("unable to import simultaneous pdf!");
   }

   auto &mc = *static_cast<RooStats::ModelConfig *>(_workspace.obj(mcname));
   mc.SetWS(_workspace);
   mc.SetPdf(*pdf);
   RooArgSet observables;
   for (auto const &child : analysisNode["variables"].children()) {
      if (auto var = _workspace.var(child.val())) {
         observables.add(*var);
      }
   }
   RooArgSet nps;
   RooArgSet pois;
   for (auto const &child : analysisNode["pois"].children()) {
      pois.add(*_workspace.var(child.val()));
   }
   RooArgSet globs;
   std::unique_ptr<RooArgSet> pdfVars{pdf->getVariables()};
   for (auto &var : _workspace.allVars()) {
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

   // Create the combined dataset for RooFit
   std::map<std::string, RooAbsData *> dsMap;
   RooArgSet allVars{pdf->indexCat()};
   for (std::size_t iChannel = 0; iChannel < nllDataNames.size(); ++iChannel) {
      RooAbsData *channelData = _workspace.data(nllDataNames[iChannel]);
      dsMap.insert({"channel_" + std::to_string(iChannel), channelData});
      allVars.add(*channelData->get());
   }

   if (!mcAuxNode.has_child("combined_data_name")) {
      throw std::runtime_error("Any imported ModelConfig must have the combined_data_name attribute (for now)!");
   }
   std::string name = mcAuxNode["combined_data_name"].val();

   pdf->setStringAttribute("combined_data_name", name.c_str());

   RooDataSet obsData{name.c_str(), name.c_str(), allVars, RooFit::Import(dsMap),
                      RooFit::Index(const_cast<RooCategory &>(static_cast<RooCategory const &>(pdf->indexCat())))};
   _workspace.import(obsData);
}

void RooJSONFactoryWSTool::importAllNodes(const RooFit::Detail::JSONNode &n)
{
   _domains = std::make_unique<Domains>();
   _domains->readJSON(n["domains"]);

   _rootnodeInput = &n;
   this->importDependants(n);

   for (const auto &d : loadData(n)) {
      _workspace.import(*d.second);
   }

   _workspace.saveSnapshot("fromJSON", _workspace.allVars());
   if (n.has_child("parameter_points")) {
      for (const auto &snsh : n["parameter_points"].children()) {
         std::string name = RooJSONFactoryWSTool::name(snsh);
         if (name == "fromJSON")
            continue;
         RooArgSet vars;
         for (const auto &var : snsh.children()) {
            std::string vname = RooJSONFactoryWSTool::name(var);
            RooRealVar *rrv = _workspace.var(vname);
            if (!rrv)
               continue;
            configureVariable(*_domains, var, *rrv);
            vars.add(*rrv);
         }
         _workspace.saveSnapshot(name, vars);
      }
   }
   _workspace.loadSnapshot("fromJSON");

   // Import attributes
   for (RooAbsArg *arg : _workspace.components()) {
      importAttributes(arg, *_rootnodeInput);
   }

   _rootnodeInput = nullptr;
   _domains.reset();

   // Now, read in analyses and likelihoods if there are any
   if (n.has_child("analyses")) {
      for (JSONNode const &analysisNode : n["analyses"].children()) {
         importAnalysis(analysisNode, n["likelihoods"],
                        dereference(n, {"misc", "ROOT_internal", "ModelConfigs", analysisNode["name"].val()}, n));
      }
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
