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
#include <RooFit/Detail/JSONInterface.h>

#include <RooGlobalFunc.h>
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

#include "TROOT.h"
#include "TH1.h"

#include "RConfigure.h"

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

RooJSONFactoryWSTool::RooJSONFactoryWSTool(RooWorkspace &ws) : _workspace{ws} {}

RooJSONFactoryWSTool::~RooJSONFactoryWSTool() {}

RooFit::Detail::JSONNode &RooJSONFactoryWSTool::orootnode()
{
   if (_rootnode_output)
      return *_rootnode_output;
   throw MissingRootnodeError();
}
const RooFit::Detail::JSONNode &RooJSONFactoryWSTool::irootnode() const
{
   if (_rootnode_input)
      return *_rootnode_input;
   throw MissingRootnodeError();
}

template <>
RooRealVar *RooJSONFactoryWSTool::request<RooRealVar>(const std::string &objname, const std::string &requestAuthor)
{
   RooRealVar *retval = _workspace.var(objname);
   if (retval)
      return retval;
   if (irootnode().has_child("variables")) {
      const JSONNode &vars = irootnode()["variables"];
      if (vars.has_child(objname)) {
         this->importVariable(vars[objname]);
         retval = _workspace.var(objname);
         if (retval)
            return retval;
      }
   }
   throw DependencyMissingError(requestAuthor, objname, "RooRealVar");
}

template <>
RooAbsPdf *RooJSONFactoryWSTool::request<RooAbsPdf>(const std::string &objname, const std::string &requestAuthor)
{
   RooAbsPdf *retval = _workspace.pdf(objname);
   if (retval)
      return retval;
   if (irootnode().has_child("pdfs")) {
      const JSONNode &pdfs = irootnode()["pdfs"];
      if (pdfs.has_child(objname)) {
         this->importFunction(pdfs[objname], true);
         retval = _workspace.pdf(objname);
         if (retval)
            return retval;
      }
   }
   throw DependencyMissingError(requestAuthor, objname, "RooAbsPdf");
}

template <>
RooAbsReal *RooJSONFactoryWSTool::request<RooAbsReal>(const std::string &objname, const std::string &requestAuthor)
{
   RooAbsReal *retval = nullptr;
   retval = _workspace.pdf(objname);
   if (retval)
      return retval;
   retval = _workspace.function(objname);
   if (retval)
      return retval;
   retval = _workspace.var(objname);
   if (retval)
      return retval;
   if (isNumber(objname))
      return dynamic_cast<RooAbsReal *>(_workspace.factory(objname));
   if (irootnode().has_child("pdfs")) {
      const JSONNode &pdfs = irootnode()["pdfs"];
      if (pdfs.has_child(objname)) {
         this->importFunction(pdfs[objname], true);
         retval = _workspace.pdf(objname.c_str());
         if (retval)
            return retval;
      }
   }
   if (irootnode().has_child("variables")) {
      const JSONNode &vars = irootnode()["variables"];
      if (vars.has_child(objname)) {
         this->importVariable(vars[objname]);
         retval = _workspace.var(objname);
         if (retval)
            return retval;
      }
   }
   if (irootnode().has_child("functions")) {
      const JSONNode &funcs = irootnode()["functions"];
      if (funcs.has_child(objname)) {
         this->importFunction(funcs[objname], false);
         retval = _workspace.function(objname);
         if (retval)
            return retval;
      }
   }
   throw DependencyMissingError(requestAuthor, objname, "RooAbsReal");
}

namespace {

void logInputArgumentsError(std::stringstream &&ss)
{
   oocoutE(nullptr, InputArguments) << ss.str();
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
         if (prefix.size() > 0)
            prefix += "_";
         prefix += ns.val();
      }
   }
   if (trailing_underscore && prefix.size() > 0)
      prefix += "_";
   return prefix;
}

namespace {
// helpers for serializing / deserializing binned datasets
inline void genIndicesHelper(std::vector<std::vector<int>> &combinations, std::vector<int> &curr_comb,
                             const std::vector<int> &vars_numbins, size_t curridx)
{
   if (curridx == vars_numbins.size()) {
      // we have filled a combination. Copy it.
      combinations.push_back(std::vector<int>(curr_comb));
   } else {
      for (int i = 0; i < vars_numbins[curridx]; ++i) {
         curr_comb[curridx] = i;
         ::genIndicesHelper(combinations, curr_comb, vars_numbins, curridx + 1);
      }
   }
}

std::string containerName(RooAbsArg const *elem)
{
   std::string contname = "functions";
   if (elem->InheritsFrom(RooAbsPdf::Class()))
      contname = "pdfs";
   if (elem->InheritsFrom(RooRealVar::Class()) || elem->InheritsFrom(RooConstVar::Class()))
      contname = "variables";
   return contname;
}
} // namespace

bool RooJSONFactoryWSTool::Config::stripObservables = true;

///////////////////////////////////////////////////////////////////////////////////////////////////////
// helper functions specific to JSON
///////////////////////////////////////////////////////////////////////////////////////////////////////

namespace {

inline void importAttributes(RooAbsArg *arg, const JSONNode &n)
{
   if (!n.is_map())
      return;
   if (n.has_child("dict") && n["dict"].is_map()) {
      for (const auto &attr : n["dict"].children()) {
         arg->setStringAttribute(RooJSONFactoryWSTool::name(attr).c_str(), attr.val().c_str());
      }
   }
   if (n.has_child("tags") && n["tags"].is_seq()) {
      for (const auto &attr : n["tags"].children()) {
         arg->setAttribute(attr.val().c_str());
      }
   }
}
inline bool checkRegularBins(const TAxis &ax)
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
} // namespace

///////////////////////////////////////////////////////////////////////////////////////////////////////
// RooWSFactoryTool expression handling
///////////////////////////////////////////////////////////////////////////////////////////////////////

namespace {
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
         RooJSONFactoryWSTool::error(err.str().c_str());
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
}; // namespace

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
   auto &observables = n["observables"];
   observables.set_map();
   auto &x = observables[varnames[0]];
   writeAxis(x, *(h.GetXaxis()));
   if (h.GetDimension() > 1) {
      auto &y = observables[varnames[1]];
      writeAxis(y, *(h.GetYaxis()));
      if (h.GetDimension() > 2) {
         auto &z = observables[varnames[2]];
         writeAxis(z, *(h.GetZaxis()));
      }
   }
}

void RooJSONFactoryWSTool::exportHistogram(const TH1 &h, JSONNode &n, const std::vector<std::string> &varnames,
                                           const TH1 *errH, bool writeObservables, bool writeErrors)
{
   n.set_map();
   auto &weights = n["counts"];
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

bool RooJSONFactoryWSTool::find(const JSONNode &n, const std::string &elem)
{
   // find an attribute
   if (n.is_seq()) {
      for (const auto &t : n.children()) {
         if (t.val() == elem)
            return true;
      }
      return false;
   } else if (n.is_map()) {
      return n.has_child(elem.c_str());
   }
   return false;
}

void RooJSONFactoryWSTool::append(JSONNode &n, const std::string &elem)
{
   // append an attribute
   n.set_seq();
   if (!find(n, elem)) {
      n.append_child() << elem;
   }
}

void RooJSONFactoryWSTool::exportAttributes(const RooAbsArg *arg, JSONNode &n)
{
   // export all string attributes of an object
   if (arg->stringAttributes().size() > 0) {
      auto &dict = n["dict"];
      dict.set_map();
      for (const auto &it : arg->stringAttributes()) {
         dict[it.first] << it.second;
      }
   }
   if (arg->attributes().size() > 0) {
      auto &tags = n["tags"];
      for (const auto &it : arg->attributes()) {
         RooJSONFactoryWSTool::append(tags, it);
      }
   }
}

void RooJSONFactoryWSTool::exportVariable(const RooAbsArg *v, JSONNode &n)
{
   auto *cv = dynamic_cast<const RooConstVar *>(v);
   auto *rrv = dynamic_cast<const RooRealVar *>(v);
   if (!cv && !rrv)
      return;

   n.set_map();

   JSONNode &var = n[v->GetName()];
   var.set_map();
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
   RooJSONFactoryWSTool::exportAttributes(v, var);
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
   auto &n = orootnode()[containerName(func)];
   n.set_map();

   auto const &exporters = RooFit::JSONIO::exporters();
   auto const &exportKeys = RooFit::JSONIO::exportKeys();

   // if this element already exists, skip
   if (n.has_child(func->GetName()))
      return &n[func->GetName()];

   if (func->InheritsFrom(RooConstVar::Class()) &&
       strcmp(func->GetName(), TString::Format("%g", ((RooConstVar *)func)->getVal()).Data()) == 0) {
      // for RooConstVar, if name and value are the same, we don't need to do anything
      return nullptr;
   } else if (func->InheritsFrom(RooAbsCategory::Class())) {
      // categories are created by the respective RooSimultaneous, so we're skipping the export here
      return nullptr;
   } else if (func->InheritsFrom(RooRealVar::Class()) || func->InheritsFrom(RooConstVar::Class())) {
      // for variables, call the variable exporter
      exportVariable(func, n);
      return nullptr;
   }

   TClass *cl = TClass::GetClass(func->ClassName());

   auto it = exporters.find(cl);
   if (it != exporters.end()) { // check if we have a specific exporter available
      for (auto &exp : it->second) {
         try {
            auto &elem = n[func->GetName()];
            elem.set_map();
            if (!exp->exportObject(this, func, elem)) {
               continue;
            }
            if (exp->autoExportDependants()) {
               RooJSONFactoryWSTool::exportDependants(func);
            }
            RooJSONFactoryWSTool::exportAttributes(func, elem);
            return &elem;
         } catch (const std::exception &ex) {
            std::cerr << "error exporting " << func->Class()->GetName() << " " << func->GetName() << ": " << ex.what()
                      << ". skipping." << std::endl;
            return nullptr;
         }
      }
   }

   // generic export using the factory expressions
   const auto &dict = exportKeys.find(cl);
   if (dict == exportKeys.end()) {
      std::cerr << "unable to export class '" << cl->GetName() << "' - no export keys available!" << std::endl;
      std::cerr << "there are several possible reasons for this:" << std::endl;
      std::cerr << " 1. " << cl->GetName() << " is a custom class that you or some package you are using added."
                << std::endl;
      std::cerr << " 2. " << cl->GetName()
                << " is a ROOT class that nobody ever bothered to write a serialization definition for." << std::endl;
      std::cerr << " 3. something is wrong with your setup, e.g. you might have called "
                   "RooJSONFactoryWSTool::clearExportKeys() and/or never successfully read a file defining these "
                   "keys with RooJSONFactoryWSTool::loadExportKeys(filename)"
                << std::endl;
      std::cerr << "either way, please make sure that:" << std::endl;
      std::cerr << " 3: you are reading a file with export keys - call RooJSONFactoryWSTool::printExportKeys() to "
                   "see what is available"
                << std::endl;
      std::cerr << " 2 & 1: you might need to write a serialization definition yourself. check "
                   "https://github.com/root-project/root/blob/master/roofit/hs3/README.md to "
                   "see how to do this!"
                << std::endl;
      return nullptr;
   }

   RooJSONFactoryWSTool::exportDependants(func);

   auto &elem = n[func->GetName()];
   elem.set_map();
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
   RooJSONFactoryWSTool::exportAttributes(func, elem);

   return &elem;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
// importing functions
void RooJSONFactoryWSTool::importFunctions(const JSONNode &n)
{
   // import a list of RooAbsReal objects
   if (!n.is_map())
      return;
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
   // if it's an empty name, it's a lost cause, let's just skip it
   if (name.empty())
      return;
   // if the function already exists, we don't need to do anything
   if (isPdf) {
      if (_workspace.pdf(name.c_str()))
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
         RooJSONFactoryWSTool::error(err.str().c_str());
      } else {
         ::importAttributes(func, p);
      }
   } catch (const RooJSONFactoryWSTool::DependencyMissingError &ex) {
      throw;
   } catch (const RooJSONFactoryWSTool::MissingRootnodeError &ex) {
      throw;
   } catch (const std::exception &ex) {
      std::stringstream ss;
      ss << "RooJSONFactoryWSTool(): error importing " << name << ": " << ex.what() << ". skipping." << std::endl;
      logInputArgumentsError(std::move(ss));
   }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
// generating an unbinned dataset from a binned one

std::unique_ptr<RooDataSet> RooJSONFactoryWSTool::unbinned(RooDataHist const &hist)
{
   RooArgSet obs(*hist.get());
   auto data = std::make_unique<RooDataSet>(hist.GetName(), hist.GetTitle(), obs, RooFit::WeightVar());
   for (int i = 0; i < hist.numEntries(); ++i) {
      data->add(*hist.get(i), hist.weight(i));
   }
   return data;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
// importing data
std::map<std::string, std::unique_ptr<RooAbsData>> RooJSONFactoryWSTool::loadData(const JSONNode &n)
{
   std::map<std::string, std::unique_ptr<RooAbsData>> dataMap;
   if (!n.is_map()) {
      return dataMap;
   }
   for (const auto &p : n.children()) {
      std::string name(RooJSONFactoryWSTool::name(p));
      if (name.empty())
         continue;
      if (_workspace.data(name.c_str()))
         continue;
      if (!p.is_map())
         continue;
      if (p.has_child("counts")) {
         // binned
         dataMap[name] = this->readBinnedData(p, name);
      } else if (p.has_child("coordinates")) {
         // unbinned
         RooArgSet vars;
         this->getObservables(_workspace, p, name, vars);
         RooArgList varlist(vars);
         auto data = std::make_unique<RooDataSet>(name, name, vars, RooFit::WeightVar());
         auto &coords = p["coordinates"];
         auto &weights = p["weights"];
         if (coords.num_children() != weights.num_children()) {
            RooJSONFactoryWSTool::error("inconsistent number of coordinates and weights!");
         }
         if (!coords.is_seq()) {
            RooJSONFactoryWSTool::error("key 'coordinates' is not a list!");
         }
         for (size_t i = 0; i < coords.num_children(); ++i) {
            auto &point = coords[i];
            if (!point.is_seq()) {
               RooJSONFactoryWSTool::error(TString::Format("coordinate point '%d' is not a list!", (int)i).Data());
            }
            if (point.num_children() != varlist.size()) {
               RooJSONFactoryWSTool::error("inconsistent number of coordinates and observables!");
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
void RooJSONFactoryWSTool::exportData(RooAbsData *data, JSONNode &n)
{
   // The data might already have been exported
   if (n.has_child(data->GetName())) {
      return;
   }

   RooArgSet observables(*data->get());

   // find category observables
   RooAbsCategory *cat = nullptr;
   for (const auto &obs : observables) {
      if (obs->InheritsFrom(RooAbsCategory::Class())) {
         if (cat) {
            RooJSONFactoryWSTool::error("dataset '" + std::string(data->GetName()) +
                                        " has several category observables!");
         } else {
            cat = (RooAbsCategory *)(obs);
         }
      }
   }

   if (cat) {
      // this is a composite dataset
      RooDataSet *ds = (RooDataSet *)(data);
      std::unique_ptr<TList> dataList{ds->split(*(cat), true)};
      if (!dataList) {
         RooJSONFactoryWSTool::error("unable to split dataset '" + std::string(ds->GetName()) + "' at '" +
                                     std::string(cat->GetName()) + "'");
      }
      for (RooAbsData *absData : static_range_cast<RooAbsData *>(*dataList)) {
         absData->SetName((std::string(data->GetName()) + "_" + absData->GetName()).c_str());
         this->exportData(absData, n);
      }

      return;
   } else if (data->InheritsFrom(RooDataHist::Class())) {
      // this is a binned dataset
      auto &output = n[data->GetName()];
      output.set_map();
      RooDataHist *dh = (RooDataHist *)(data);

      auto &observablesNode = output["observables"];
      observablesNode.set_map();
      for (RooRealVar *var : static_range_cast<RooRealVar *>(observables)) {
         auto &observableNode = observablesNode[var->GetName()];
         observableNode.set_map();
         observableNode["min"] << var->getMin();
         observableNode["max"] << var->getMax();
         observableNode["nbins"] << var->numBins();
      }

      auto &weights = output["counts"];
      weights.set_seq();
      for (int i = 0; i < dh->numEntries(); ++i) {
         dh->get(i);
         weights.append_child() << dh->weight();
      }
      return;
   }
   auto &output = n[data->GetName()];
   output.set_map();

   // this is a regular unbinned dataset
   RooDataSet *ds = (RooDataSet *)(data);

   bool singlePoint = (ds->numEntries() <= 1);
   RooArgSet reduced_obs;
   if (Config::stripObservables) {
      if (!singlePoint) {
         std::map<RooRealVar *, std::vector<double>> obs_values;
         for (int i = 0; i < ds->numEntries(); ++i) {
            ds->get(i);
            for (const auto &obs : observables) {
               RooRealVar *rv = (RooRealVar *)(obs);
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
      reduced_obs.add(observables);
   }
   if (reduced_obs.size() > 0) {
      exportVariables(reduced_obs, output["observables"]);
   }
   auto &weights = singlePoint && Config::stripObservables ? output["counts"] : output["weights"];
   weights.set_seq();
   for (int i = 0; i < ds->numEntries(); ++i) {
      ds->get(i);
      if (!(Config::stripObservables && singlePoint)) {
         auto &coords = output["coordinates"];
         coords.set_seq();
         coords.append_child().fill_seq(reduced_obs, [](auto x) { return static_cast<RooRealVar *>(x)->getVal(); });
      }
      weights.append_child() << ds->weight();
   }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
// create several observables
void RooJSONFactoryWSTool::getObservables(RooWorkspace &ws, const JSONNode &n, const std::string &obsnamecomp,
                                          RooArgSet &out)
{
   auto vars = RooJSONFactoryWSTool::readObservables(n, obsnamecomp);
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

std::unique_ptr<RooDataHist> RooJSONFactoryWSTool::readBinnedData(const JSONNode &n, const std::string &namecomp)
{
   RooArgList varlist;

   for (JSONNode const &obsNode : n["observables"].children()) {
      auto var = std::make_unique<RooRealVar>(obsNode.key().c_str(), obsNode.key().c_str(), obsNode["min"].val_double(),
                                              obsNode["max"].val_double());
      var->setBins(obsNode["nbins"].val_double());
      varlist.addOwned(std::move(var));
   }

   return readBinnedData(n, namecomp, varlist);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
// reading binned data
std::unique_ptr<RooDataHist>
RooJSONFactoryWSTool::readBinnedData(const JSONNode &n, const std::string &namecomp, RooArgList varlist)
{
   if (!n.has_child("counts"))
      RooJSONFactoryWSTool::error("no counts given");
   if (!n["counts"].is_seq())
      RooJSONFactoryWSTool::error("counts are not in list form");
   auto &counts = n["counts"];

   auto bins = RooJSONFactoryWSTool::generateBinIndices(varlist);
   if (counts.num_children() != bins.size())
      RooJSONFactoryWSTool::error(TString::Format("inconsistent bin numbers: counts=%d, bins=%d",
                                                  (int)counts.num_children(), (int)(bins.size())));
   auto dh = std::make_unique<RooDataHist>(namecomp.c_str(), namecomp.c_str(), varlist);
   // temporarily disable dirty flag propagation when filling the RDH
   std::vector<double> initVals;
   for (auto &v : varlist) {
      v->setDirtyInhibit(true);
      initVals.push_back(((RooRealVar *)v)->getVal());
   }
   for (size_t ibin = 0; ibin < bins.size(); ++ibin) {
      for (size_t i = 0; i < bins[ibin].size(); ++i) {
         RooRealVar *v = (RooRealVar *)(varlist.at(i));
         v->setBin(bins[ibin][i]);
      }
      dh->add(varlist, counts[ibin].val_double());
   }
   // re-enable dirty flag propagation
   for (size_t i = 0; i < varlist.size(); ++i) {
      RooRealVar *v = (RooRealVar *)(varlist.at(i));
      v->setVal(initVals[i]);
      v->setDirtyInhibit(false);
   }
   return dh;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
// read observables
std::map<std::string, RooJSONFactoryWSTool::Var>
RooJSONFactoryWSTool::readObservables(const JSONNode &n, const std::string &obsnamecomp)
{
   std::map<std::string, RooJSONFactoryWSTool::Var> vars;
   if (!n.is_map())
      return vars;
   if (n.has_child("observables")) {
      auto &observables = n["observables"];
      if (!observables.is_map())
         return vars;
      if (observables.has_child("nbins")) {
         vars.emplace(std::make_pair("obs_x_" + obsnamecomp, RooJSONFactoryWSTool::Var(observables)));
      } else {
         for (const auto &p : observables.children()) {
            vars.emplace(std::make_pair(RooJSONFactoryWSTool::name(p), RooJSONFactoryWSTool::Var(p)));
         }
      }
   } else {
      vars.emplace(std::make_pair("obs_x_" + obsnamecomp, RooJSONFactoryWSTool::Var(n["counts"].num_children())));
   }
   return vars;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
// importing pdfs
void RooJSONFactoryWSTool::importPdfs(const JSONNode &n)
{
   // import a list of RooAbsPdf objects
   if (!n.is_map())
      return;
   for (const auto &p : n.children()) {
      this->importFunction(p, true);
   }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
// importing variables
void RooJSONFactoryWSTool::importVariables(const JSONNode &n)
{
   // import a list of RooRealVar objects
   if (!n.is_map())
      return;
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
      ss << "RooJSONFactoryWSTool() node '" << name << "' is not a map, skipping." << std::endl;
      logInputArgumentsError(std::move(ss));
      return;
   }
   RooRealVar v(name.c_str(), name.c_str(), 1.);
   configureVariable(*_domains, p, v);
   ::importAttributes(&v, p);
   _workspace.import(v, RooFit::RecycleConflictNodes(true), RooFit::Silence(true));
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
   if (n.has_child("variables")) {
      this->importVariables(n["variables"]);
   }
   if (n.has_child("functions")) {
      this->importFunctions(n["functions"]);
   }
   if (n.has_child("pdfs")) {
      this->importPdfs(n["pdfs"]);
   }
}

std::string RooJSONFactoryWSTool::name(const JSONNode &n)
{
   std::stringstream ss;
   if (n.is_container() && n.has_child("name")) {
      ss << n["name"].val();
   } else if (n.has_key()) {
      ss << n.key();
   } else {
      ss << n.val();
   }
   return ss.str();
}

void RooJSONFactoryWSTool::tagVariables(JSONNode &rootnode, RooArgSet const *args, const char *tag)
{
   if (args == nullptr)
      return;

   for (RooAbsArg *arg : *args) {
      if (auto *v = dynamic_cast<RooRealVar *>(arg)) {
         auto &vars = rootnode["variables"];
         exportVariable(v, vars);
         auto &tags = vars[v->GetName()]["tags"];
         RooJSONFactoryWSTool::append(tags, tag);
      }
   }
}

void RooJSONFactoryWSTool::exportModelConfig(JSONNode &rootnode, RooStats::ModelConfig const &mc)
{
   auto pdf = dynamic_cast<RooSimultaneous const *>(mc.GetPdf());
   if (pdf == nullptr) {
      throw std::runtime_error("RooFitHS3 only supports ModelConfigs with RooSimultaneous!");
   }

   auto &analysesNode = rootnode["analyses"];
   analysesNode.set_map();

   auto &likelihoodsNode = rootnode["likelihoods"];
   likelihoodsNode.set_map();

   auto &analysisNode = analysesNode[mc.GetPdf()->GetName()];
   analysisNode.set_map();

   auto &analysisDomains = analysisNode["domains"];
   analysisDomains.set_seq();
   analysisDomains.append_child() << "default_domain";

   auto &analysisLikelihoods = analysisNode["likelihoods"];
   analysisLikelihoods.set_seq();

   std::string basename = "obsData";
   if (auto s = pdf->getStringAttribute("combinedObservationName")) {
      basename = s;
      analysisNode["combinedObservationName"] << s;
   }

   for (auto const &item : pdf->indexCat()) {
      analysisLikelihoods.append_child() << item.first;

      auto &nllNode = likelihoodsNode[item.first];
      nllNode.set_map();
      nllNode["dist"] << pdf->getPdf(item.first.c_str())->GetName();
      nllNode["obs"] << basename + "_" + item.first;
   }

   auto writeList = [&](const char *name, RooArgSet const *args) {
      if (!args)
         return;
      auto &node = analysisNode[name];
      node.set_seq();
      for (RooAbsArg *arg : *args) {
         node.append_child() << arg->GetName();
      }
   };

   writeList("observables", mc.GetObservables());
   writeList("pois", mc.GetParametersOfInterest());
   writeList("nps", mc.GetNuisanceParameters());
   writeList("globs", mc.GetGlobalObservables());
}

void RooJSONFactoryWSTool::exportAllObjects(JSONNode &n)
{
   _domains = std::make_unique<Domains>();

   // export all ModelConfig objects and attached Pdfs
   std::vector<RooStats::ModelConfig *> mcs;
   std::vector<RooAbsPdf *> toplevel;
   _rootnode_output = &n;
   for (auto obj : _workspace.allGenericObjects()) {
      if (obj->InheritsFrom(RooStats::ModelConfig::Class())) {
         auto *mc = static_cast<RooStats::ModelConfig *>(obj);
         tagVariables(n, mc->GetObservables(), "observable");
         tagVariables(n, mc->GetParametersOfInterest(), "poi");
         tagVariables(n, mc->GetNuisanceParameters(), "np");
         tagVariables(n, mc->GetGlobalObservables(), "glob");
         mcs.push_back(mc);
         exportModelConfig(n, *mc);
      }
   }
   for (auto pdf : _workspace.allPdfs()) {
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
      auto &data = n["data"];
      data.set_map();
      this->exportData(d, data);
   }
   for (const auto *snsh_obj : _workspace.getSnapshots()) {
      const RooArgSet *snsh = static_cast<const RooArgSet *>(snsh_obj);
      auto &snapshots = n["snapshots"];
      snapshots.set_map();
      this->exportVariables(*snsh, snapshots[snsh->GetName()]);
   }
   for (const auto &mc : mcs) {
      exportTopLevelPdf(n, *mc->GetPdf(), mc->GetName());
   }
   for (const auto &pdf : toplevel) {
      exportTopLevelPdf(n, *pdf, std::string(pdf->GetName()) + "_modelConfig");
   }
   _domains->writeJSON(n["domains"]);
   _domains.reset();
   _rootnode_output = nullptr;
}

void RooJSONFactoryWSTool::exportTopLevelPdf(JSONNode &node, RooAbsPdf const &pdf, std::string const &modelConfigName)
{
   auto &pdfs = node["pdfs"];
   pdfs.set_map();
   RooJSONFactoryWSTool::exportObject(&pdf);
   auto &pdfNode = pdfs[pdf.GetName()];
   pdfNode.set_map();
   if (!pdf.getAttribute("toplevel"))
      RooJSONFactoryWSTool::append(pdfNode["tags"], "toplevel");
   auto &dict = pdfNode["dict"];
   dict.set_map();
   dict["ModelConfig"] << modelConfigName;
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
   n["metadata"]["version"] << "0.1";
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
                                          const RooFit::Detail::JSONNode &likelihoodsNode)
{

   // if this is a toplevel pdf, also create a modelConfig for it
   std::string mcname = "ModelConfig";
   {
      RooStats::ModelConfig mc{mcname.c_str(), analysisNode.key().c_str()};
      _workspace.import(mc);
   }

   std::stringstream ss;
   ss << "SIMUL::" << analysisNode.key() << "(channelCat[";
   {
      std::size_t iNLL = 0;
      std::size_t nNLL = analysisNode["likelihoods"].num_children();
      for (auto const &child : analysisNode["likelihoods"].children()) {
         ss << child.val() << "=" << iNLL;
         if (iNLL < nNLL - 1) {
            ss << ",";
         }
         ++iNLL;
      }
   }
   ss << "]";
   for (auto const &child : analysisNode["likelihoods"].children()) {
      ss << ", " << child.val() << "=" << likelihoodsNode[child.val()]["dist"].val();
   }
   ss << ")";
   RooSimultaneous &pdf = *static_cast<RooSimultaneous *>(_workspace.factory(ss.str()));

   auto &mc = *static_cast<RooStats::ModelConfig *>(_workspace.obj(mcname));
   mc.SetWS(_workspace);
   mc.SetPdf(pdf);
   RooArgSet observables;
   for (auto const &child : analysisNode["observables"].children()) {
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
   std::unique_ptr<RooArgSet> pdfVars{pdf.getVariables()};
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
   std::map<std::string, RooDataSet *> dsMap;
   std::stack<std::unique_ptr<RooDataSet>> ownedDataSets;
   RooArgSet allVars{pdf.indexCat()};
   RooDataSet *channelData = nullptr;
   for (auto const &child : analysisNode["likelihoods"].children()) {
      auto &nllNode = likelihoodsNode[child.val()];
      RooAbsData *originalChannelData = _workspace.data(nllNode["obs"].val());
      if (originalChannelData->InheritsFrom(RooDataHist::Class())) {
         ownedDataSets.push(unbinned(static_cast<RooDataHist const &>(*originalChannelData)));
         channelData = ownedDataSets.top().get();
      } else {
         channelData = static_cast<RooDataSet *>(originalChannelData);
      }
      dsMap.insert({child.val(), channelData});
      allVars.add(*channelData->get());
   }

   std::string name = "obsData"; // default name for the combined data if not specified in the JSON
   if (analysisNode.has_child("combinedObservationName")) {
      name = analysisNode["combinedObservationName"].val();
   }

   pdf.setStringAttribute("combinedObservationName", name.c_str());

   RooDataSet obsData{name.c_str(),
                      name.c_str(),
                      allVars,
                      RooFit::Import(dsMap),
                      RooFit::Index(const_cast<RooCategory &>(static_cast<RooCategory const &>(pdf.indexCat())))};
   _workspace.import(obsData);
}

void RooJSONFactoryWSTool::importAllNodes(const RooFit::Detail::JSONNode &n)
{
   _domains = std::make_unique<Domains>();
   _domains->readJSON(n["domains"]);

   _rootnode_input = &n;
   this->importDependants(n);

   if (n.has_child("data")) {
      auto data = this->loadData(n["data"]);
      for (const auto &d : data) {
         _workspace.import(*d.second);
      }
   }

   _workspace.saveSnapshot("fromJSON", _workspace.allVars());
   if (n.has_child("snapshots")) {
      for (const auto &snsh : n["snapshots"].children()) {
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
         _workspace.saveSnapshot(name.c_str(), vars);
      }
   }
   _workspace.loadSnapshot("fromJSON");

   _rootnode_input = nullptr;
   _domains.reset();

   // Now, read in analyses and likelihoods if there are any
   if (n.has_child("analyses")) {
      for (JSONNode const &analysisNode : n["analyses"].children()) {
         importAnalysis(analysisNode, n["likelihoods"]);
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
