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

#include <RooFitHS3/RooJSONFactoryWSTool.h>

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
#include <RooStats/ModelConfig.h>

#include "TROOT.h"
#include "TH1.h"

#include "RConfigure.h"

#ifdef ROOFIT_HS3_WITH_RYML
#include "RYMLParser.h"
typedef TRYMLTree tree_t;
#else
#include "JSONParser.h"
typedef TJSONTree tree_t;
#endif

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
~~~ {.py}
tool = ROOT.RooJSONFactoryWSTool(ws)
tool.printImporters()
tool.printExporters()
tool.printFactoryExpressions()
tool.printExportKeys()
~~~

Alternatively, you can generate a LaTeX version of the available importers and exporters by calling
~~~ {.py}
tool = ROOT.RooJSONFactoryWSTool(ws)
tool.writedoc("hs3.tex")
~~~

*/

using RooFit::Experimental::JSONNode;

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
} // namespace

RooFit::Experimental::JSONNode &RooJSONFactoryWSTool::orootnode()
{
   if (_rootnode_output)
      return *_rootnode_output;
   throw MissingRootnodeError();
}
const RooFit::Experimental::JSONNode &RooJSONFactoryWSTool::irootnode() const
{
   if (_rootnode_input)
      return *_rootnode_input;
   throw MissingRootnodeError();
}

template <>
RooRealVar *RooJSONFactoryWSTool::request<RooRealVar>(const std::string &objname, const std::string &requestAuthor)
{
   RooRealVar *retval = this->workspace()->var(objname.c_str());
   if (retval)
      return retval;
   if (irootnode().has_child("variables")) {
      const JSONNode &vars = irootnode()["variables"];
      if (vars.has_child(objname)) {
         this->importVariable(vars[objname]);
         retval = this->workspace()->var(objname.c_str());
         if (retval)
            return retval;
      }
   }
   throw DependencyMissingError(requestAuthor, objname, "RooRealVar");
}

template <>
RooAbsPdf *RooJSONFactoryWSTool::request<RooAbsPdf>(const std::string &objname, const std::string &requestAuthor)
{
   RooAbsPdf *retval = this->workspace()->pdf(objname.c_str());
   if (retval)
      return retval;
   if (irootnode().has_child("pdfs")) {
      const JSONNode &pdfs = irootnode()["pdfs"];
      if (pdfs.has_child(objname)) {
         this->importFunction(pdfs[objname], true);
         retval = this->workspace()->pdf(objname.c_str());
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
   retval = this->workspace()->pdf(objname.c_str());
   if (retval)
      return retval;
   retval = this->workspace()->function(objname.c_str());
   if (retval)
      return retval;
   retval = this->workspace()->var(objname.c_str());
   if (retval)
      return retval;
   if (isNumber(objname))
      return dynamic_cast<RooAbsReal *>(this->workspace()->factory(objname.c_str()));
   if (irootnode().has_child("pdfs")) {
      const JSONNode &pdfs = irootnode()["pdfs"];
      if (pdfs.has_child(objname)) {
         this->importFunction(pdfs[objname], true);
         retval = this->workspace()->pdf(objname.c_str());
         if (retval)
            return retval;
      }
   }
   if (irootnode().has_child("variables")) {
      const JSONNode &vars = irootnode()["variables"];
      if (vars.has_child(objname)) {
         this->importVariable(vars[objname]);
         retval = this->workspace()->var(objname.c_str());
         if (retval)
            return retval;
      }
   }
   if (irootnode().has_child("functions")) {
      const JSONNode &funcs = irootnode()["functions"];
      if (funcs.has_child(objname)) {
         this->importFunction(funcs[objname], false);
         retval = this->workspace()->function(objname.c_str());
         if (retval)
            return retval;
      }
   }
   throw DependencyMissingError(requestAuthor, objname, "RooAbsReal");
}

namespace {

void logInputArgumentsError(std::stringstream &&ss)
{
   oocoutE(static_cast<RooAbsArg *>(nullptr), InputArguments) << ss.str();
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
         this->min = val["min"].val_float();
      if (!val.has_child("max"))
         this->max = 1;
      else
         this->max = val["max"].val_float();
   } else if (val.is_seq()) {
      for (size_t i = 0; i < val.num_children(); ++i) {
         this->bounds.push_back(val[i].val_float());
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

std::string containerName(RooAbsArg *elem)
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

bool RooJSONFactoryWSTool::registerImporter(const std::string &key,
                                            std::unique_ptr<const RooJSONFactoryWSTool::Importer> f, bool topPriority)
{
   auto &vec = staticImporters()[key];
   vec.insert(topPriority ? vec.begin() : vec.end(), std::move(f));
   return true;
}

bool RooJSONFactoryWSTool::registerExporter(const TClass *key, std::unique_ptr<const RooJSONFactoryWSTool::Exporter> f,
                                            bool topPriority)
{
   auto &vec = staticExporters()[key];
   vec.insert(topPriority ? vec.begin() : vec.end(), std::move(f));
   return true;
}

int RooJSONFactoryWSTool::removeImporters(const std::string &needle)
{
   int n = 0;
   for (auto &element : staticImporters()) {
      for (size_t i = element.second.size(); i > 0; --i) {
         auto *imp = element.second[i - 1].get();
         std::string name(typeid(*imp).name());
         if (name.find(needle) != std::string::npos) {
            element.second.erase(element.second.begin() + i - 1);
            ++n;
         }
      }
   }
   return n;
}

int RooJSONFactoryWSTool::removeExporters(const std::string &needle)
{
   int n = 0;
   for (auto &element : staticExporters()) {
      for (size_t i = element.second.size(); i > 0; --i) {
         auto *imp = element.second[i - 1].get();
         std::string name(typeid(*imp).name());
         if (name.find(needle) != std::string::npos) {
            element.second.erase(element.second.begin() + i - 1);
            ++n;
         }
      }
   }
   return n;
}

void RooJSONFactoryWSTool::printImporters()
{
   for (const auto &x : staticImporters()) {
      for (const auto &ePtr : x.second) {
         // Passing *e directory to typeid results in clang warnings.
         auto const &e = *ePtr;
         std::cout << x.first << "\t" << typeid(e).name() << std::endl;
      }
   }
}
void RooJSONFactoryWSTool::printExporters()
{
   for (const auto &x : staticExporters()) {
      for (const auto &ePtr : x.second) {
         // Passing *e directory to typeid results in clang warnings.
         auto const &e = *ePtr;
         std::cout << x.first->GetName() << "\t" << typeid(e).name() << std::endl;
      }
   }
}

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
      if (fabs(ax.GetBinUpEdge(i) - (ax.GetXmin() + (bw * i))) > w * 1e-6)
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
std::string generate(const RooJSONFactoryWSTool::ImportExpression &ex, const JSONNode &p, RooJSONFactoryWSTool *tool)
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

void RooJSONFactoryWSTool::loadFactoryExpressions(const std::string &fname)
{
   auto &pdfFactoryExpressions = staticPdfImportExpressions();
   auto &funcFactoryExpressions = staticFunctionImportExpressions();

   // load a yml file defining the factory expressions
   std::ifstream infile(fname);
   if (!infile.is_open()) {
      std::cerr << "unable to read file '" << fname << "'" << std::endl;
      return;
   }
   try {
      tree_t p(infile);
      const JSONNode &n = p.rootnode();
      for (const auto &cl : n.children()) {
         std::string key(RooJSONFactoryWSTool::name(cl));
         if (!cl.has_child("class")) {
            std::cerr << "error in file '" << fname << "' for entry '" << key << "': 'class' key is required!"
                      << std::endl;
            continue;
         }
         std::string classname(cl["class"].val());
         TClass *c = TClass::GetClass(classname.c_str());
         if (!c) {
            std::cerr << "unable to find class " << classname << ", skipping." << std::endl;
         } else {
            ImportExpression ex;
            ex.tclass = c;
            if (!cl.has_child("arguments")) {
               std::cerr << "class " << classname << " seems to have no arguments attached, skipping" << std::endl;
               continue;
            }
            for (const auto &arg : cl["arguments"].children()) {
               ex.arguments.push_back(arg.val());
            }
            if (c->InheritsFrom(RooAbsPdf::Class())) {
               pdfFactoryExpressions[key] = ex;
            } else if (c->InheritsFrom(RooAbsReal::Class())) {
               funcFactoryExpressions[key] = ex;
            } else {
               std::cerr << "class " << classname << " seems to not inherit from any suitable class, skipping"
                         << std::endl;
            }
         }
      }
   } catch (const std::exception &ex) {
      std::cout << "caught" << std::endl;
      std::cerr << "unable to load factory expressions: " << ex.what() << std::endl;
   }
}
void RooJSONFactoryWSTool::clearFactoryExpressions()
{
   // clear all factory expressions
   staticPdfImportExpressions().clear();
   staticFunctionImportExpressions().clear();
}
void RooJSONFactoryWSTool::printFactoryExpressions()
{
   // print all factory expressions
   for (auto it : staticPdfImportExpressions()) {
      std::cout << it.first;
      std::cout << " " << it.second.tclass->GetName();
      for (auto v : it.second.arguments) {
         std::cout << " " << v;
      }
      std::cout << std::endl;
   }
   for (auto it : staticFunctionImportExpressions()) {
      std::cout << it.first;
      std::cout << " " << it.second.tclass->GetName();
      for (auto v : it.second.arguments) {
         std::cout << " " << v;
      }
      std::cout << std::endl;
   }
}

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
// RooProxy-based export handling
///////////////////////////////////////////////////////////////////////////////////////////////////////

void RooJSONFactoryWSTool::loadExportKeys(const std::string &fname)
{
   auto &exportKeys = staticExportKeys();

   // load a yml file defining the export keys
   std::ifstream infile(fname);
   if (!infile.is_open()) {
      std::cerr << "unable to read file '" << fname << "'" << std::endl;
      return;
   }
   try {
      tree_t p(infile);
      const JSONNode &n = p.rootnode();
      for (const auto &cl : n.children()) {
         std::string classname(RooJSONFactoryWSTool::name(cl));
         TClass *c = TClass::GetClass(classname.c_str());
         if (!c) {
            std::cerr << "unable to find class " << classname << ", skipping." << std::endl;
         } else {
            ExportKeys ex;
            if (!cl.has_child("type")) {
               std::cerr << "class " << classname << "has not type key set, skipping" << std::endl;
               continue;
            }
            if (!cl.has_child("proxies")) {
               std::cerr << "class " << classname << "has no proxies identified, skipping" << std::endl;
               continue;
            }
            ex.type = cl["type"].val();
            for (const auto &k : cl["proxies"].children()) {
               std::string key(RooJSONFactoryWSTool::name(k));
               std::string val(k.val());
               ex.proxies[key] = val;
            }
            exportKeys[c] = ex;
         }
      }
   } catch (const std::exception &ex) {
      std::cerr << "unable to load export keys: " << ex.what() << std::endl;
   }
}

void RooJSONFactoryWSTool::clearExportKeys()
{
   // clear all export keys
   staticExportKeys().clear();
}

void RooJSONFactoryWSTool::printExportKeys()
{
   // print all export keys
   for (const auto &it : staticExportKeys()) {
      std::cout << it.first->GetName() << ": " << it.second.type;
      for (const auto &kv : it.second.proxies) {
         std::cout << " " << kv.first << "=" << kv.second;
      }
      std::cout << std::endl;
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
      tags.set_seq();
      for (const auto &it : arg->attributes()) {
         RooJSONFactoryWSTool::append(tags, it);
      }
   }
}

void RooJSONFactoryWSTool::exportVariable(const RooAbsReal *v, JSONNode &n)
{
   auto &var = n[v->GetName()];
   const RooConstVar *cv = dynamic_cast<const RooConstVar *>(v);
   const RooRealVar *rrv = dynamic_cast<const RooRealVar *>(v);
   var.set_map();
   if (cv) {
      var["value"] << cv->getVal();
      var["const"] << true;
   } else if (rrv) {
      var["value"] << rrv->getVal();
      if (rrv->getMin() > -1e30) {
         var["min"] << rrv->getMin();
      }
      if (rrv->getMax() < 1e30) {
         var["max"] << rrv->getMax();
      }
      if (rrv->isConstant()) {
         var["const"] << rrv->isConstant();
      }
      if (rrv->numBins() != 100) {
         var["nbins"] << rrv->numBins();
      }
   }
   RooJSONFactoryWSTool::exportAttributes(v, var);
}

void RooJSONFactoryWSTool::exportVariables(const RooArgSet &allElems, JSONNode &n)
{
   // export a list of RooRealVar objects
   for (auto *arg : allElems) {
      RooAbsReal *v = dynamic_cast<RooAbsReal *>(arg);
      if (!v)
         continue;
      if (v->InheritsFrom(RooRealVar::Class()) || v->InheritsFrom(RooConstVar::Class())) {
         exportVariable(v, n);
      }
   }
}

void RooJSONFactoryWSTool::exportObject(const RooAbsArg *func, JSONNode &n)
{
   auto const &exporters = staticExporters();
   auto const &exportKeys = staticExportKeys();

   // if this element already exists, skip
   if (n.has_child(func->GetName()))
      return;

   if (func->InheritsFrom(RooConstVar::Class()) &&
       strcmp(func->GetName(), TString::Format("%g", ((RooConstVar *)func)->getVal()).Data()) == 0) {
      // for RooConstVar, if name and value are the same, we don't need to do anything
      return;
   } else if (func->InheritsFrom(RooAbsCategory::Class())) {
      // categories are created by the respective RooSimultaneous, so we're skipping the export here
      return;
   } else if (func->InheritsFrom(RooRealVar::Class()) || func->InheritsFrom(RooConstVar::Class())) {
      // for variables, call the variable exporter
      exportVariable(static_cast<const RooAbsReal *>(func), n);
      return;
   }

   TClass *cl = TClass::GetClass(func->ClassName());

   auto it = exporters.find(cl);
   bool ok = false;
   if (it != exporters.end()) { // check if we have a specific exporter available
      for (auto &exp : it->second) {
         try {
            auto &elem = n[func->GetName()];
            elem.set_map();
            if (!exp->exportObject(this, func, elem)) {
               continue;
            }
            if (exp->autoExportDependants()) {
               RooJSONFactoryWSTool::exportDependants(func, &orootnode());
            }
            RooJSONFactoryWSTool::exportAttributes(func, elem);
            ok = true;
         } catch (const std::exception &ex) {
            std::cerr << "error exporting " << func->Class()->GetName() << " " << func->GetName() << ": " << ex.what()
                      << ". skipping." << std::endl;
            return;
         }
         if (ok)
            break;
      }
   }
   if (!ok) { // generic export using the factory expressions
      const auto &dict = exportKeys.find(cl);
      if (dict == exportKeys.end()) {
         std::cerr << "unable to export class '" << cl->GetName() << "' - no export keys available!" << std::endl;
         std::cerr << "there are several possible reasons for this:" << std::endl;
         std::cerr << " 1. " << cl->GetName() << " is a custom class that you or some package you are using added."
                   << std::endl;
         std::cerr << " 2. " << cl->GetName()
                   << " is a ROOT class that nobody ever bothered to write a serialization definition for."
                   << std::endl;
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
         return;
      }

      RooJSONFactoryWSTool::exportDependants(func, &orootnode());

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
            return;
         }

         RooListProxy *l = dynamic_cast<RooListProxy *>(p);
         if (l) {
            auto &items = elem[k->second];
            items.set_seq();
            for (auto e : *l) {
               items.append_child() << e->GetName();
            }
         }
         RooRealProxy *r = dynamic_cast<RooRealProxy *>(p);
         if (r) {
            elem[k->second] << r->arg().GetName();
         }
      }
      RooJSONFactoryWSTool::exportAttributes(func, elem);
   }
}

void RooJSONFactoryWSTool::exportFunctions(const RooArgSet &allElems, JSONNode &n)
{
   // export a list of functions
   // note: this function assumes that all the dependants of these objects have already been exported
   for (auto *arg : allElems) {
      RooAbsReal *func = dynamic_cast<RooAbsReal *>(arg);
      if (!func)
         continue;
      RooJSONFactoryWSTool::exportObject(func, n);
   }
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
   auto const &importers = staticImporters();
   auto const &pdfFactoryExpressions = staticPdfImportExpressions();
   auto const &funcFactoryExpressions = staticFunctionImportExpressions();

   // some preparations: what type of function are we dealing with here?
   std::string name(RooJSONFactoryWSTool::name(p));
   // if it's an empty name, it's a lost cause, let's just skip it
   if (name.empty())
      return;
   // if the function already exists, we don't need to do anything
   if (isPdf) {
      if (this->_workspace->pdf(name.c_str()))
         return;
   } else {
      if (this->_workspace->function(name.c_str()))
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
   if (prefix.size() > 0)
      name = prefix + name;
   if (!p.has_child("type")) {
      std::stringstream ss;
      ss << "RooJSONFactoryWSTool() no type given for function '" << name << "', skipping." << std::endl;
      logInputArgumentsError(std::move(ss));
      return;
   }

   bool toplevel = false;
   if (isPdf && p.has_child("tags")) {
      toplevel = RooJSONFactoryWSTool::find(p["tags"], "toplevel");
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
            if (!this->_workspace->factory(expression.c_str())) {
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
                  "RooJSONFactoryWSTool::clearFactoryExpressions() and/or never successfully read a file defining "
                  "these expressions with RooJSONFactoryWSTool::loadFactoryExpressions(filename)\n"
               << "either way, please make sure that:\n"
               << " 3: you are reading a file with factory expressions - call "
                  "RooJSONFactoryWSTool::printFactoryExpressions() "
                  "to see what is available\n"
               << " 2 & 1: you might need to write a deserialization definition yourself. check "
                  "https://github.com/root-project/root/blob/master/roofit/hs3/README.md to see "
                  "how to do this!"
               << std::endl;
            logInputArgumentsError(std::move(ss));
            return;
         }
      }
      RooAbsReal *func = this->_workspace->function(name.c_str());
      if (!func) {
         std::stringstream err;
         err << "something went wrong importing function '" << name << "'.";
         RooJSONFactoryWSTool::error(err.str().c_str());
      } else {
         ::importAttributes(func, p);

         if (isPdf && toplevel) {
            configureToplevelPdf(p, *static_cast<RooAbsPdf *>(func));
         }
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
   RooRealVar *weight = this->getWeightVar("weight");
   obs.add(*weight, true);
   auto data = std::make_unique<RooDataSet>(hist.GetName(), hist.GetTitle(), obs, RooFit::WeightVar(*weight));
   for (int i = 0; i < hist.numEntries(); ++i) {
      data->add(*hist.get(i), hist.weight(i));
   }
   return data;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
// generating a weight variable

RooRealVar *RooJSONFactoryWSTool::getWeightVar(const char *weightName)
{
   RooRealVar *weightVar = _workspace->var(weightName);
   if (!weightVar) {
      _workspace->factory(TString::Format("%s[0.,0.,10000000]", weightName).Data());
   }
   weightVar = _workspace->var(weightName);
   return weightVar;
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
      if (this->_workspace->data(name.c_str()))
         continue;
      if (!p.is_map())
         continue;
      if (p.has_child("counts")) {
         // binned
         RooArgSet vars;
         this->getObservables(p, name, vars);
         dataMap[name] = this->readBinnedData(p, name, vars);
      } else if (p.has_child("coordinates")) {
         // unbinned
         RooArgSet vars;
         this->getObservables(p, name, vars);
         RooArgList varlist(vars);
         RooRealVar *weightVar = this->getWeightVar("weight");
         vars.add(*weightVar, true);
         auto data = std::make_unique<RooDataSet>(name, name, vars, RooFit::WeightVar(*weightVar));
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
               v->setVal(point[j].val_float());
            }
            data->add(vars, weights[i].val_float());
         }
         dataMap[name] = std::move(data);
      } else if (p.has_child("index")) {
         // combined measurement
         auto subMap = loadData(p);
         auto catname = p["index"].val();
         RooCategory *channelCat = _workspace->cat(catname.c_str());
         if (!channelCat) {
            std::stringstream ss;
            ss << "RooJSONFactoryWSTool() failed to retrieve channel category " << catname << std::endl;
            logInputArgumentsError(std::move(ss));
         } else {
            RooArgSet allVars;
            allVars.add(*channelCat, true);
            std::stack<std::unique_ptr<RooDataSet>> ownedDataSets;
            std::map<std::string, RooDataSet *> datasets;
            for (const auto &subd : subMap) {
               allVars.add(*subd.second->get(), true);
               if (subd.second->InheritsFrom(RooDataHist::Class())) {
                  ownedDataSets.push(unbinned(static_cast<RooDataHist const &>(*subd.second)));
                  datasets[subd.first] = ownedDataSets.top().get();
               } else {
                  datasets[subd.first] = static_cast<RooDataSet *>(subd.second.get());
               }
            }
            RooRealVar *weightVar = this->getWeightVar("weight");
            allVars.add(*weightVar, true);
            dataMap[name] =
               std::make_unique<RooDataSet>(name.c_str(), name.c_str(), allVars, RooFit::Index(*channelCat),
                                            RooFit::Import(datasets), RooFit::WeightVar(*weightVar));
         }
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
   RooArgSet observables(*data->get());
   auto &output = n[data->GetName()];
   output.set_map();

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
      output["index"] << cat->GetName();
      std::unique_ptr<TList> dataList{ds->split(*(cat), true)};
      if (!dataList) {
         RooJSONFactoryWSTool::error("unable to split dataset '" + std::string(ds->GetName()) + "' at '" +
                                     std::string(cat->GetName()) + "'");
      }
      for (RooAbsData *absData : static_range_cast<RooAbsData *>(*dataList)) {
         this->exportData(absData, output);
      }
   } else if (data->InheritsFrom(RooDataHist::Class())) {
      // this is a binned dataset
      RooDataHist *dh = (RooDataHist *)(data);
      auto &obs = output["observables"];
      obs.set_map();
      exportVariables(observables, obs);
      auto &weights = output["counts"];
      weights.set_seq();
      for (int i = 0; i < dh->numEntries(); ++i) {
         dh->get(i);
         weights.append_child() << dh->weight();
      }
   } else {
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
         auto &obsset = output["observables"];
         obsset.set_map();
         exportVariables(reduced_obs, obsset);
      }
      auto &weights = singlePoint && Config::stripObservables ? output["counts"] : output["weights"];
      weights.set_seq();
      for (int i = 0; i < ds->numEntries(); ++i) {
         ds->get(i);
         if (!(Config::stripObservables && singlePoint)) {
            auto &coordinates = output["coordinates"];
            coordinates.set_seq();
            auto &point = coordinates.append_child();
            point.set_seq();
            for (const auto &obs : reduced_obs) {
               RooRealVar *rv = (RooRealVar *)(obs);
               point.append_child() << rv->getVal();
            }
         }
         weights.append_child() << ds->weight();
      }
   }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
// create several observables
void RooJSONFactoryWSTool::getObservables(const JSONNode &n, const std::string &obsnamecomp, RooArgSet &out)
{
   if (!_scope.observables.empty()) {
      out.add(_scope.observables.begin(), _scope.observables.end());
      return;
   }
   auto vars = readObservables(n, obsnamecomp);
   for (auto v : vars) {
      std::string name(v.first);
      if (_workspace->var(name.c_str())) {
         out.add(*(_workspace->var(name.c_str())));
      } else {
         out.add(*RooJSONFactoryWSTool::createObservable(name, v.second));
      }
   }
}

void RooJSONFactoryWSTool::setScopeObservables(const RooArgList &args)
{
   for (auto *arg : args) {
      _scope.observables.push_back(arg);
   }
}
void RooJSONFactoryWSTool::setScopeObject(const std::string &name, RooAbsArg *obj)
{
   this->_scope.objects[name] = obj;
}
RooAbsArg *RooJSONFactoryWSTool::getScopeObject(const std::string &name)
{
   return this->_scope.objects[name];
}
void RooJSONFactoryWSTool::clearScope()
{
   this->_scope.objects.clear();
   this->_scope.observables.clear();
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
// create an observable
RooRealVar *RooJSONFactoryWSTool::createObservable(const std::string &name, const RooJSONFactoryWSTool::Var &var)
{
   this->_workspace->factory(TString::Format("%s[%f]", name.c_str(), var.min));
   RooRealVar *rrv = this->_workspace->var(name.c_str());
   rrv->setMin(var.min);
   rrv->setMax(var.max);
   rrv->setConstant(true);
   rrv->setBins(var.nbins);
   rrv->setAttribute("observable");
   return rrv;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
// reading binned data
std::unique_ptr<RooDataHist>
RooJSONFactoryWSTool::readBinnedData(const JSONNode &n, const std::string &namecomp, RooArgList varlist)
{
   if (!n.is_map())
      RooJSONFactoryWSTool::error("data is not a map");
   if (varlist.size() == 0) {
      std::string obsname = "obs_x_" + namecomp;
      varlist.add(*(this->_workspace->factory((obsname + "[0.]").c_str())));
   }
   auto bins = RooJSONFactoryWSTool::generateBinIndices(varlist);
   if (!n.has_child("counts"))
      RooJSONFactoryWSTool::error("no counts given");
   if (!n["counts"].is_seq())
      RooJSONFactoryWSTool::error("counts are not in list form");
   auto &counts = n["counts"];
   if (counts.num_children() != bins.size())
      RooJSONFactoryWSTool::error(TString::Format("inconsistent bin numbers: counts=%d, bins=%d",
                                                  (int)counts.num_children(), (int)(bins.size())));
   auto dh = std::make_unique<RooDataHist>(("dataHist_" + namecomp).c_str(), namecomp.c_str(), varlist);
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
      dh->add(varlist, counts[ibin].val_float());
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
// configure a pdf as "toplevel" by creating a modelconfig for it
void RooJSONFactoryWSTool::configureToplevelPdf(const JSONNode &p, RooAbsPdf &pdf)
{
   // if this is a toplevel pdf, also create a modelConfig for it
   std::string mcname = "ModelConfig";
   if (p.has_child("dict")) {
      if (p["dict"].has_child("ModelConfig")) {
         mcname = p["dict"]["ModelConfig"].val();
      }
   }
   {
      RooStats::ModelConfig mc{mcname.c_str(), pdf.GetName()};
      this->_workspace->import(mc);
   }
   RooStats::ModelConfig *inwsmc = dynamic_cast<RooStats::ModelConfig *>(this->_workspace->obj(mcname.c_str()));
   if (inwsmc) {
      inwsmc->SetWS(*(this->_workspace));
      inwsmc->SetPdf(pdf);
      RooArgSet observables;
      RooArgSet nps;
      RooArgSet pois;
      RooArgSet globs;
      std::unique_ptr<RooArgSet> pdfVars{pdf.getVariables()};
      for (auto &var : this->_workspace->allVars()) {
         if (!pdfVars->find(*var))
            continue;
         if (var->getAttribute("observable")) {
            observables.add(*var, true);
         }
         if (var->getAttribute("np")) {
            nps.add(*var, true);
         }
         if (var->getAttribute("poi")) {
            pois.add(*var, true);
         }
         if (var->getAttribute("glob")) {
            globs.add(*var, true);
         }
      }
      inwsmc->SetObservables(observables);
      inwsmc->SetParametersOfInterest(pois);
      inwsmc->SetNuisanceParameters(nps);
      inwsmc->SetGlobalObservables(globs);
   } else {
      std::stringstream ss;
      ss << "RooJSONFactoryWSTool() object '" << mcname << "' in workspace is not of type RooStats::ModelConfig!"
         << std::endl;
      logInputArgumentsError(std::move(ss));
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
   if (this->_workspace->var(name.c_str()))
      return;
   if (!p.is_map()) {
      std::stringstream ss;
      ss << "RooJSONFactoryWSTool() node '" << name << "' is not a map, skipping." << std::endl;
      logInputArgumentsError(std::move(ss));
      return;
   }
   RooRealVar v(name.c_str(), name.c_str(), 1.);
   configureVariable(p, v);
   ::importAttributes(&v, p);
   this->_workspace->import(v, RooFit::RecycleConflictNodes(true), RooFit::Silence(true));
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
// configuring variable
void RooJSONFactoryWSTool::configureVariable(const JSONNode &p, RooRealVar &v)
{
   if (p.has_child("value"))
      v.setVal(p["value"].val_float());
   if (p.has_child("min"))
      v.setMin(p["min"].val_float());
   if (p.has_child("max"))
      v.setMax(p["max"].val_float());
   if (p.has_child("nbins"))
      v.setBins(p["nbins"].val_int());
   if (p.has_child("relErr"))
      v.setError(v.getVal() * p["relErr"].val_float());
   if (p.has_child("err"))
      v.setError(p["err"].val_float());
   if (p.has_child("const"))
      v.setConstant(p["const"].val_bool());
   else
      v.setConstant(false);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
// export all dependants (servers) of a RooAbsArg
void RooJSONFactoryWSTool::exportDependants(const RooAbsArg *source, JSONNode *n)
{
   if (n) {
      this->exportDependants(source, *n);
   } else {
      RooJSONFactoryWSTool::error(
         "cannot export dependents without a valid root node, only call within the context of 'exportAllObjects'");
   }
}

void RooJSONFactoryWSTool::exportDependants(const RooAbsArg *source, JSONNode &n)
{
   // export all the servers of a given RooAbsArg
   auto servers(source->servers());
   for (auto s : servers) {
      auto &container = n[containerName(s)];
      container.set_map();
      this->exportObject(s, container);
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

void RooJSONFactoryWSTool::exportAllObjects(JSONNode &n)
{
   // export all ModelConfig objects and attached Pdfs
   std::vector<RooStats::ModelConfig *> mcs;
   std::vector<RooAbsPdf *> toplevel;
   this->_rootnode_output = &n;
   for (auto obj : this->_workspace->allGenericObjects()) {
      if (obj->InheritsFrom(RooStats::ModelConfig::Class())) {
         RooStats::ModelConfig *mc = static_cast<RooStats::ModelConfig *>(obj);
         auto &vars = n["variables"];
         vars.set_map();
         if (mc->GetObservables()) {
            for (auto obs : *(mc->GetObservables())) {
               RooRealVar *v = dynamic_cast<RooRealVar *>(obs);
               if (v) {
                  exportVariable(v, vars);
                  auto &tags = vars[v->GetName()]["tags"];
                  tags.set_seq();
                  RooJSONFactoryWSTool::append(tags, "observable");
               }
            }
         }
         if (mc->GetParametersOfInterest()) {
            for (auto poi : *(mc->GetParametersOfInterest())) {
               RooRealVar *v = dynamic_cast<RooRealVar *>(poi);
               if (v) {
                  exportVariable(v, vars);
                  auto &tags = vars[v->GetName()]["tags"];
                  tags.set_seq();
                  RooJSONFactoryWSTool::append(tags, "poi");
               }
            }
         }
         if (mc->GetNuisanceParameters()) {
            for (auto np : *(mc->GetNuisanceParameters())) {
               RooRealVar *v = dynamic_cast<RooRealVar *>(np);
               if (v) {
                  exportVariable(v, vars);
                  auto &tags = vars[v->GetName()]["tags"];
                  tags.set_seq();
                  RooJSONFactoryWSTool::append(tags, "np");
               }
            }
         }
         if (mc->GetGlobalObservables()) {
            for (auto *np : *mc->GetGlobalObservables()) {
               if (auto *v = dynamic_cast<RooRealVar *>(np)) {
                  exportVariable(v, vars);
                  auto &tags = vars[v->GetName()]["tags"];
                  tags.set_seq();
                  RooJSONFactoryWSTool::append(tags, "glob");
               }
            }
         }
         mcs.push_back(mc);
      }
   }
   for (auto pdf : this->_workspace->allPdfs()) {
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
   for (auto d : this->_workspace->allData()) {
      auto &data = n["data"];
      data.set_map();
      this->exportData(d, data);
   }
   for (const auto *snsh_obj : this->_workspace->getSnapshots()) {
      const RooArgSet *snsh = static_cast<const RooArgSet *>(snsh_obj);
      auto &snapshots = n["snapshots"];
      snapshots.set_map();
      auto &coll = snapshots[snsh->GetName()];
      coll.set_map();
      this->exportVariables(*snsh, coll);
   }
   for (const auto &mc : mcs) {
      auto &pdfs = n["pdfs"];
      pdfs.set_map();
      RooAbsPdf *pdf = mc->GetPdf();
      RooJSONFactoryWSTool::exportObject(pdf, pdfs);
      auto &node = pdfs[pdf->GetName()];
      node.set_map();
      auto &tags = node["tags"];
      tags.set_seq();
      if (!pdf->getAttribute("toplevel"))
         RooJSONFactoryWSTool::append(tags, "toplevel");
      auto &dict = node["dict"];
      dict.set_map();
      dict["ModelConfig"] << mc->GetName();
   }
   for (const auto &pdf : toplevel) {
      auto &pdfs = n["pdfs"];
      pdfs.set_map();
      RooJSONFactoryWSTool::exportObject(pdf, pdfs);
      auto &node = pdfs[pdf->GetName()];
      node.set_map();
      auto &tags = node["tags"];
      tags.set_seq();
      if (!pdf->getAttribute("toplevel"))
         RooJSONFactoryWSTool::append(tags, "toplevel");
      auto &dict = node["dict"];
      dict.set_map();
      if (toplevel.size() + mcs.size() == 1) {
         dict["ModelConfig"] << "ModelConfig";
      } else {
         dict["ModelConfig"] << std::string(pdf->GetName()) + "_modelConfig";
      }
   }
   this->_rootnode_output = nullptr;
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

bool RooJSONFactoryWSTool::exportJSON(std::ostream &os)
{
   // export the workspace in JSON
   tree_t p;
   JSONNode &n = p.rootnode();
   n.set_map();
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
   tree_t p;
   JSONNode &n = p.rootnode();
   n.set_map();
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

void RooJSONFactoryWSTool::importAllNodes(const RooFit::Experimental::JSONNode &n)
{
   this->_rootnode_input = &n;
   gROOT->ProcessLine("using namespace RooStats::HistFactory;");
   this->importDependants(n);

   if (n.has_child("data")) {
      auto data = this->loadData(n["data"]);
      for (const auto &d : data) {
         this->_workspace->import(*d.second);
      }
   }

   this->_workspace->saveSnapshot("fromJSON", this->_workspace->allVars());
   if (n.has_child("snapshots")) {
      for (const auto &snsh : n["snapshots"].children()) {
         std::string name = RooJSONFactoryWSTool::name(snsh);
         if (name == "fromJSON")
            continue;
         for (const auto &var : snsh.children()) {
            std::string vname = RooJSONFactoryWSTool::name(var);
            RooRealVar *rrv = this->_workspace->var(vname.c_str());
            if (!rrv)
               continue;
            this->configureVariable(var, *rrv);
         }
      }
   }
   this->_workspace->loadSnapshot("fromJSON");

   this->_rootnode_input = nullptr;
}

bool RooJSONFactoryWSTool::importJSON(std::istream &is)
{
   // import a JSON file to the workspace
   try {
      tree_t p(is);
      this->importAllNodes(p.rootnode());
   } catch (const std::exception &ex) {
      std::cerr << "unable to import JSON: " << ex.what() << std::endl;
      return false;
   }
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
   try {
      tree_t p(is);
      this->importAllNodes(p.rootnode());
   } catch (const std::exception &ex) {
      std::cerr << "unable to import JSON: " << ex.what() << std::endl;
      return false;
   }
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

std::ostream &RooJSONFactoryWSTool::log(int level) const
{
   return RooMsgService::instance().log(static_cast<TObject *>(nullptr), static_cast<RooFit::MsgLevel>(level),
                                        RooFit::IO);
}

RooJSONFactoryWSTool::ImportMap &RooJSONFactoryWSTool::staticImporters()
{
   static ImportMap _importers;
   return _importers;
}

RooJSONFactoryWSTool::ExportMap &RooJSONFactoryWSTool::staticExporters()
{
   static ExportMap _exporters;
   return _exporters;
}

RooJSONFactoryWSTool::ImportExpressionMap &RooJSONFactoryWSTool::staticPdfImportExpressions()
{
   static ImportExpressionMap _pdfFactoryExpressions;
   return _pdfFactoryExpressions;
}

RooJSONFactoryWSTool::ImportExpressionMap &RooJSONFactoryWSTool::staticFunctionImportExpressions()
{
   static ImportExpressionMap _funcFactoryExpressions;
   return _funcFactoryExpressions;
}

RooJSONFactoryWSTool::ExportKeysMap &RooJSONFactoryWSTool::staticExportKeys()
{
   static ExportKeysMap _exportKeys;
   return _exportKeys;
}
