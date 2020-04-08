#include <RooFitHS3/RooJSONFactoryWSTool.h>

#include <iostream>
#include <fstream>
#include <stdexcept>

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

using RooFit::Detail::JSONNode;

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
         RooJSONFactoryWSTool::error("no nbins given");
      if (!val.has_child("min"))
         RooJSONFactoryWSTool::error("no min given");
      if (!val.has_child("max"))
         RooJSONFactoryWSTool::error("no max given");
      this->nbins = val["nbins"].val_int();
      this->min = val["min"].val_float();
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
inline void genIndicesHelper(std::vector<std::vector<int>> &combinations, const RooArgList &vars, size_t curridx)
{
   if (curridx == vars.size()) {
      std::vector<int> indices(curridx);
      for (size_t i = 0; i < curridx; ++i) {
         RooRealVar *v = (RooRealVar *)(vars.at(i));
         indices[i] = v->getBinning().binNumber(v->getVal());
      }
      combinations.push_back(indices);
   } else {
      RooRealVar *v = (RooRealVar *)(vars.at(curridx));
      for (int i = 0; i < v->numBins(); ++i) {
         v->setVal(v->getBinning().binCenter(i));
         ::genIndicesHelper(combinations, vars, curridx + 1);
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

// maps to hold the importers and exporters for runtime lookup
RooJSONFactoryWSTool::ImportMap RooJSONFactoryWSTool::_importers = RooJSONFactoryWSTool::ImportMap();
RooJSONFactoryWSTool::ExportMap RooJSONFactoryWSTool::_exporters = RooJSONFactoryWSTool::ExportMap();

bool RooJSONFactoryWSTool::registerImporter(const std::string &key, const RooJSONFactoryWSTool::Importer *f)
{
   if (RooJSONFactoryWSTool::_importers.find(key) != RooJSONFactoryWSTool::_importers.end())
      return false;
   RooJSONFactoryWSTool::_importers.insert(std::make_pair(key, f));
   return true;
}
bool RooJSONFactoryWSTool::registerExporter(const TClass *key, const RooJSONFactoryWSTool::Exporter *f)
{
   if (RooJSONFactoryWSTool::_exporters.find(key) != RooJSONFactoryWSTool::_exporters.end())
      return false;
   RooJSONFactoryWSTool::_exporters.insert(std::make_pair(key, f));
   return true;
}

void RooJSONFactoryWSTool::printImporters()
{
   for (const auto &x : RooJSONFactoryWSTool::_importers) {
      std::cout << x.first << "\t" << typeid(*x.second).name() << std::endl;
   }
}
void RooJSONFactoryWSTool::printExporters()
{
   for (const auto &x : RooJSONFactoryWSTool::_exporters) {
      std::cout << x.first << "\t" << typeid(*x.second).name() << std::endl;
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
struct JSON_Factory_Expression {
   TClass *tclass;
   std::vector<std::string> arguments;
   std::string generate(const JSONNode &p)
   {
      std::string name(RooJSONFactoryWSTool::name(p));
      std::stringstream expression;
      std::string classname(this->tclass->GetName());
      size_t colon = classname.find_last_of(":");
      if (colon < classname.size()) {
         expression << classname.substr(colon + 1);
      } else {
         expression << classname;
      }
      expression << "::" << name << "(";
      bool first = true;
      for (auto k : this->arguments) {
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
            err << "factory expression for class '" << this->tclass->GetName() << "', which expects key '" << k
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
               expression << x.val();
            }
            expression << "}";
         } else {
            expression << p[k].val();
         }
      }
      expression << ")";
      return expression.str();
   }
};
std::map<std::string, JSON_Factory_Expression> _pdfFactoryExpressions;
std::map<std::string, JSON_Factory_Expression> _funcFactoryExpressions;
} // namespace

void RooJSONFactoryWSTool::loadFactoryExpressions(const std::string &fname)
{
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
            JSON_Factory_Expression ex;
            ex.tclass = c;
            if (!cl.has_child("arguments")) {
               std::cerr << "class " << classname << " seems to have no arguments attached, skipping" << std::endl;
               continue;
            }
            for (const auto &arg : cl["arguments"].children()) {
               ex.arguments.push_back(arg.val());
            }
            if (c->InheritsFrom(RooAbsPdf::Class())) {
               _pdfFactoryExpressions[key] = ex;
            } else if (c->InheritsFrom(RooAbsReal::Class())) {
               _funcFactoryExpressions[key] = ex;
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
   _pdfFactoryExpressions.clear();
   _funcFactoryExpressions.clear();
}
void RooJSONFactoryWSTool::printFactoryExpressions()
{
   // print all factory expressions
   for (auto it : _pdfFactoryExpressions) {
      std::cout << it.first;
      std::cout << " " << it.second.tclass->GetName();
      for (auto v : it.second.arguments) {
         std::cout << " " << v;
      }
      std::cout << std::endl;
   }
   for (auto it : _funcFactoryExpressions) {
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
   ::genIndicesHelper(combinations, vars, 0);
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

namespace {
struct JSON_Export_Keys {
   std::string type;
   std::map<std::string, std::string> proxies;
};
std::map<TClass *, JSON_Export_Keys> _exportKeys;
} // namespace
void RooJSONFactoryWSTool::loadExportKeys(const std::string &fname)
{
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
            JSON_Export_Keys ex;
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
            _exportKeys[c] = ex;
         }
      }
   } catch (const std::exception &ex) {
      std::cerr << "unable to load export keys: " << ex.what() << std::endl;
   }
}

void RooJSONFactoryWSTool::clearExportKeys()
{
   // clear all export keys
   _exportKeys.clear();
}

void RooJSONFactoryWSTool::printExportKeys()
{
   // print all export keys
   for (const auto &it : _exportKeys) {
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

   auto it = _exporters.find(cl);
   if (it != _exporters.end()) { // check if we have a specific exporter available
      if (it->second->autoExportDependants())
         RooJSONFactoryWSTool::exportDependants(func, n);
      auto &elem = n[func->GetName()];
      elem.set_map();
      try {
         if (!it->second->exportObject(this, func, elem)) {
            std::cerr << "exporter for type " << cl->GetName() << " does not export objects!" << std::endl;
         }
         RooJSONFactoryWSTool::exportAttributes(func, elem);
      } catch (const std::exception &ex) {
         std::cerr << ex.what() << ". skipping." << std::endl;
         return;
      }
   } else { // generic export using the factory expressions
      const auto &dict = _exportKeys.find(cl);
      if (dict == _exportKeys.end()) {
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
         std::cerr << " 2 & 1: you might need to write a serialization definition yourself. check INSERTLINKHERE to "
                      "see how to do this!"
                   << std::endl;
         return;
      }

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
      // some preparations: what type of function are we dealing with here?
      std::string name(RooJSONFactoryWSTool::name(p));
      if (name.empty())
         continue;
      if (this->_workspace->pdf(name.c_str()))
         continue;
      if (!p.is_map())
         continue;
      std::string prefix = RooJSONFactoryWSTool::genPrefix(p, true);
      if (prefix.size() > 0)
         name = prefix + name;
      if (!p.has_child("type")) {
         std::stringstream ss;
         ss << "RooJSONFactoryWSTool() no type given for '" << name << "', skipping." << std::endl;
         logInputArgumentsError(std::move(ss));
         continue;
      }
      std::string functype(p["type"].val());
      this->importDependants(p);
      // check for specific implementations
      auto it = _importers.find(functype);
      if (it != _importers.end()) {
         try {
            if (!it->second->importFunction(this, p)) {
               std::stringstream ss;
               ss << "RooJSONFactoryWSTool() importer for type " << functype << " does not import functions!"
                  << std::endl;
               logInputArgumentsError(std::move(ss));
            }
         } catch (const std::exception &ex) {
            std::stringstream ss;
            ss << "RooJSONFactoryWSTool() " << ex.what() << ". skipping." << std::endl;
            logInputArgumentsError(std::move(ss));
         }
      } else { // generic import using the factory expressions
         auto expr = _funcFactoryExpressions.find(functype);
         if (expr != _funcFactoryExpressions.end()) {
            std::string expression = expr->second.generate(p);
            if (!this->_workspace->factory(expression.c_str())) {
               std::stringstream ss;
               ss << "RooJSONFactoryWSTool() failed to create " << expr->second.tclass->GetName() << " '" << name
                  << "', skipping. expression was\n"
                  << expression << std::endl;
               logInputArgumentsError(std::move(ss));
            }
         } else {
            std::stringstream ss;
            ss << "RooJSONFactoryWSTool() no handling for functype '" << functype << "' implemented, skipping."
               << "\n"
               << "there are several possible reasons for this:\n"
               << " 1. " << functype << " is a custom type that is not available in RooFit.\n"
               << " 2. " << functype
               << " is a ROOT class that nobody ever bothered to write a deserialization definition for.\n"
               << " 3. something is wrong with your setup, e.g. you might have called "
                  "RooJSONFactoryWSTool::clearFactoryExpressions() and/or never successfully read a file defining "
                  "these expressions with RooJSONFactoryWSTool::loadFactoryExpressions(filename)\n"
               << "either way, please make sure that:\n"
               << " 3: you are reading a file with export keys - call RooJSONFactoryWSTool::printFactoryExpressions() "
                  "to see what is available\n"
               << " 2 & 1: you might need to write a serialization definition yourself. check INSERTLINKHERE to see "
                  "how to do this!"
               << std::endl;
            logInputArgumentsError(std::move(ss));
            continue;
         }
      }
      RooAbsReal *func = this->_workspace->function(name.c_str());
      if (!func) {
         std::stringstream ss;
         ss << "RooJSONFactoryWSTool() something went wrong importing function '" << name << "'." << std::endl;
         logInputArgumentsError(std::move(ss));
      } else {
         ::importAttributes(func, p);
      }
   }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
// generating an unbinned dataset from a binned one

RooDataSet *RooJSONFactoryWSTool::unbinned(RooDataHist *hist)
{
   RooArgSet obs(*hist->get());
   RooRealVar *weight = this->getWeightVar("weight");
   obs.add(*weight);
   RooDataSet *data = new RooDataSet(hist->GetName(), hist->GetTitle(), obs, RooFit::WeightVar(*weight));
   for (Int_t i = 0; i < hist->numEntries(); ++i) {
      data->add(*hist->get(i), hist->weight(i));
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
std::map<std::string, RooAbsData *> RooJSONFactoryWSTool::loadData(const JSONNode &n)
{
   std::map<std::string, RooAbsData *> dataMap;
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
         auto vars = this->getObservables(p, name);
         dataMap[name] = this->readBinnedData(p, name, vars);
      } else if (p.has_child("coordinates")) {
         // unbinned
         auto vars = this->getObservables(p, name);
         RooArgList varlist(vars);
         RooRealVar *weightVar = this->getWeightVar("weight");
         vars.add(*weightVar);
         RooDataSet *data = new RooDataSet(name, name, vars, RooFit::WeightVar(*weightVar));
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
               RooRealVar *v = (RooRealVar *)(varlist.at(j));
               v->setVal(point[j].val_float());
            }
            data->add(vars, weights[i].val_float());
         }
         dataMap[name] = data;
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
            allVars.add(*channelCat);
            std::map<std::string, RooDataSet *> datasets;
            for (const auto &subd : subMap) {
               allVars.add(*subd.second->get());
               if (subd.second->InheritsFrom(RooDataHist::Class())) {
                  datasets[subd.first] = unbinned(((RooDataHist *)subd.second));
               } else {
                  datasets[subd.first] = ((RooDataSet *)subd.second);
               }
            }
            RooRealVar *weightVar = this->getWeightVar("weight");
            allVars.add(*weightVar);
            RooDataSet *data = new RooDataSet(name.c_str(), name.c_str(), allVars, RooFit::Index(*channelCat),
                                              RooFit::Import(datasets), RooFit::WeightVar(*weightVar));
            dataMap[name] = data;
         }
      } else {
         std::stringstream ss;
         ss << "RooJSONFactoryWSTool() failed to create dataset" << name << std::endl;
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
   RooAbsCategory *cat = NULL;
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
      TList *dataList = ds->split(*(cat), true);
      if (!dataList) {
         RooJSONFactoryWSTool::error("unable to split dataset '" + std::string(ds->GetName()) + "' at '" +
                                     std::string(cat->GetName()) + "'");
      }
      for (RooAbsData *absData : static_range_cast<RooAbsData *>(*dataList)) {
         this->exportData(absData, output);
      }
      delete dataList;
   } else if (data->InheritsFrom(RooDataHist::Class())) {
      // this is a binned dataset
      RooDataHist *dh = (RooDataHist *)(data);
      auto &obs = output["observables"];
      obs.set_map();
      exportVariables(observables, obs);
      auto &weights = output["counts"];
      weights.set_seq();
      for (Int_t i = 0; i < dh->numEntries(); ++i) {
         dh->get(i);
         weights.append_child() << dh->weight();
      }
   } else {
      // this is a regular unbinned dataset
      RooDataSet *ds = (RooDataSet *)(data);
      RooArgSet reduced_obs;
      for (Int_t i = 0; i < ds->numEntries(); ++i) {
         ds->get(i);
         for (const auto &obs : observables) {
            RooRealVar *rv = (RooRealVar *)(obs);
            if (rv->getVal() != 0)
               reduced_obs.add(*rv);
         }
      }
      auto &obsset = output["observables"];
      obsset.set_map();
      exportVariables(reduced_obs, obsset);
      auto &coordinates = output["coordinates"];
      coordinates.set_seq();
      auto &weights = output["weights"];
      weights.set_seq();
      for (Int_t i = 0; i < ds->numEntries(); ++i) {
         auto &point = coordinates.append_child();
         ds->get(i);
         point.set_seq();
         for (const auto &obs : reduced_obs) {
            RooRealVar *rv = (RooRealVar *)(obs);
            point.append_child() << rv->getVal();
         }
         weights.append_child() << ds->weight();
      }
   }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
// create several observables
RooArgSet RooJSONFactoryWSTool::getObservables(const JSONNode &n, const std::string &obsnamecomp)
{
   if (this->_scope.observables.size() > 0) {
      return this->_scope.observables;
   }
   auto vars = readObservables(n, obsnamecomp);
   RooArgList varlist;
   for (auto v : vars) {
      std::string name(v.first);
      if (_workspace->var(name.c_str())) {
         varlist.add(*(_workspace->var(name.c_str())));
      } else {
         varlist.add(*RooJSONFactoryWSTool::createObservable(name, v.second));
      }
   }
   return varlist;
}

void RooJSONFactoryWSTool::setScopeObservables(const RooArgList &args)
{
   this->_scope.observables.add(args);
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
   this->_scope = Scope();
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
RooDataHist *
RooJSONFactoryWSTool::readBinnedData(const JSONNode &n, const std::string &namecomp, const RooArgList &varlist)
{
   if (!n.is_map())
      throw "data is not a map";
   auto bins = RooJSONFactoryWSTool::generateBinIndices(varlist);
   if (!n.has_child("counts"))
      RooJSONFactoryWSTool::error("no counts given");
   if (!n["counts"].is_seq())
      RooJSONFactoryWSTool::error("counts are not in list form");
   auto &counts = n["counts"];
   if (counts.num_children() != bins.size())
      RooJSONFactoryWSTool::error(TString::Format("inconsistent bin numbers: counts=%d, bins=%d",
                                                  (int)counts.num_children(), (int)(bins.size())));
   RooDataHist *dh = new RooDataHist(("dataHist_" + namecomp).c_str(), namecomp.c_str(), varlist);
   for (size_t ibin = 0; ibin < bins.size(); ++ibin) {
      for (size_t i = 0; i < bins[ibin].size(); ++i) {
         RooRealVar *v = (RooRealVar *)(varlist.at(i));
         v->setVal(v->getBinning().binCenter(bins[ibin][i]));
      }
      dh->add(varlist, counts[ibin].val_float());
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
      // general preparations: what type of pdf should we build?
      std::string name(RooJSONFactoryWSTool::name(p));
      if (name.empty())
         continue;
      if (this->_workspace->pdf(name.c_str()))
         continue;
      if (!p.is_map())
         continue;
      std::string prefix = RooJSONFactoryWSTool::genPrefix(p, true);
      if (prefix.size() > 0)
         name = prefix + name;
      if (!p.has_child("type")) {
         std::stringstream ss;
         ss << "RooJSONFactoryWSTool() no type given for '" << name << "', skipping." << std::endl;
         logInputArgumentsError(std::move(ss));
         continue;
      }
      bool toplevel = false;
      if (p.has_child("tags")) {
         toplevel = RooJSONFactoryWSTool::find(p["tags"], "toplevel");
      }
      std::string pdftype(p["type"].val());
      this->importDependants(p);

      // check for specific implementations
      auto it = _importers.find(pdftype);
      if (it != _importers.end()) {
         try {
            if (!it->second->importPdf(this, p)) {
               std::stringstream ss;
               ss << "RooJSONFactoryWSTool() importer for type " << pdftype << " does not import pdfs!" << std::endl;
               logInputArgumentsError(std::move(ss));
            }
         } catch (const std::exception &ex) {
            std::stringstream ss;
            ss << "RooJSONFactoryWSTool() " << ex.what() << ". skipping." << std::endl;
            logInputArgumentsError(std::move(ss));
         }
      } else { // default implementation using the factory expressions
         auto expr = _pdfFactoryExpressions.find(pdftype);
         if (expr != _pdfFactoryExpressions.end()) {
            std::string expression = expr->second.generate(p);
            if (!this->_workspace->factory(expression.c_str())) {
               std::stringstream ss;
               ss << "RooJSONFactoryWSTool() failed to create " << expr->second.tclass->GetName() << " '" << name
                  << "', skipping. expression was\n"
                  << expression << std::endl;
               logInputArgumentsError(std::move(ss));
            }
         } else {
            std::stringstream ss;
            ss << "RooJSONFactoryWSTool() no handling for pdftype '" << pdftype << "' implemented, skipping."
               << "\n"
               << "there are several possible reasons for this:\n"
               << " 1. " << pdftype << " is a custom type that is not available in RooFit.\n"
               << " 2. " << pdftype
               << " is a ROOT class that nobody ever bothered to write a deserialization definition for.\n"
               << " 3. something is wrong with your setup, e.g. you might have called "
                  "RooJSONFactoryWSTool::clearFactoryExpressions() and/or never successfully read a file defining "
                  "these expressions with RooJSONFactoryWSTool::loadFactoryExpressions(filename)\n"
               << "either way, please make sure that:\n"
               << " 3: you are reading a file with export keys - call RooJSONFactoryWSTool::printFactoryExpressions() "
                  "to see what is available\n"
               << " 2 & 1: you might need to write a serialization definition yourself. check INSERTLINKHERE to see "
                  "how to do this!"
               << std::endl;
            logInputArgumentsError(std::move(ss));
            continue;
         }
      }
      // post-processing: make sure that the pdf has been created, and attach needed attributes
      RooAbsPdf *pdf = this->_workspace->pdf(name.c_str());
      if (!pdf) {

         std::stringstream ss;
         ss << "RooJSONFactoryWSTool() something went wrong importing pdf '" << name << "'." << std::endl;
         logInputArgumentsError(std::move(ss));
      } else {
         ::importAttributes(pdf, p);
         if (toplevel) {
            // if this is a toplevel pdf, also cereate a modelConfig for it
            std::string mcname = name + "_modelConfig";
            RooStats::ModelConfig *mc = new RooStats::ModelConfig(mcname.c_str(), name.c_str());
            this->_workspace->import(*mc);
            RooStats::ModelConfig *inwsmc =
               dynamic_cast<RooStats::ModelConfig *>(this->_workspace->obj(mcname.c_str()));
            if (inwsmc) {
               inwsmc->SetWS(*(this->_workspace));
               inwsmc->SetPdf(*pdf);
               RooArgSet observables;
               RooArgSet nps;
               RooArgSet pois;
               RooArgSet globs;
               for (auto var : this->_workspace->allVars()) {
                  if (!pdf->dependsOn(*var))
                     continue;
                  if (var->getAttribute("observable")) {
                     observables.add(*var);
                  }
                  if (var->getAttribute("np")) {
                     nps.add(*var);
                  }
                  if (var->getAttribute("poi")) {
                     pois.add(*var);
                  }
                  if (var->getAttribute("glob")) {
                     globs.add(*var);
                  }
               }
               inwsmc->SetObservables(observables);
               inwsmc->SetParametersOfInterest(pois);
               inwsmc->SetNuisanceParameters(nps);
               inwsmc->SetGlobalObservables(globs);
            } else {
               std::stringstream ss;
               ss << "RooJSONFactoryWSTool() object '" << mcname
                  << "' in workspace is not of type RooStats::ModelConfig!" << std::endl;
               logInputArgumentsError(std::move(ss));
            }
         }
      }
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
      std::string name(RooJSONFactoryWSTool::name(p));
      if (this->_workspace->var(name.c_str()))
         continue;
      if (!p.is_map()) {
         std::stringstream ss;
         ss << "RooJSONFactoryWSTool() node '" << name << "' is not a map, skipping." << std::endl;
         logInputArgumentsError(std::move(ss));
         continue;
      }
      double val(p.has_child("value") ? p["value"].val_float() : 1.);
      RooRealVar v(name.c_str(), name.c_str(), val);
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
      ::importAttributes(&v, p);
      this->_workspace->import(v);
   }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
// export all dependants (servers) of a RooAbsArg
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

void RooJSONFactoryWSTool::exportAll(JSONNode &n)
{
   // export all ModelConfig objects and attached Pdfs
   RooArgSet main;
   for (auto obj : this->_workspace->allGenericObjects()) {
      if (obj->InheritsFrom(RooStats::ModelConfig::Class())) {
         RooStats::ModelConfig *mc = static_cast<RooStats::ModelConfig *>(obj);
         RooAbsPdf *pdf = mc->GetPdf();
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
         main.add(*pdf);
      }
   }
   for (auto obj : this->_workspace->allPdfs()) {
      RooAbsPdf *pdf = dynamic_cast<RooAbsPdf *>(obj);
      if (!pdf)
         continue;
      if ((pdf->getAttribute("toplevel") || pdf->clients().size() == 0) && !main.find(*pdf)) {
         this->exportDependants(pdf, n);
         main.add(*pdf);
      }
   }
   for (auto d : this->_workspace->allData()) {
      auto &data = n["data"];
      data.set_map();
      this->exportData(d, data);
   }
   if (main.size() > 0) {
      auto &pdfs = n["pdfs"];
      pdfs.set_map();
      RooJSONFactoryWSTool::RooJSONFactoryWSTool::exportFunctions(main, pdfs);
      for (auto &pdf : main) {
         auto &node = pdfs[pdf->GetName()];
         node.set_map();
         auto &tags = node["tags"];
         RooJSONFactoryWSTool::append(tags, "toplevel");
      }
   } else {
      std::cerr << "no ModelConfig found in workspace and no pdf identified as toplevel by 'toplevel' attribute or an "
                   "empty client list. nothing exported!"
                << std::endl;
   }
}

Bool_t RooJSONFactoryWSTool::importJSONfromString(const std::string &s)
{
   // import the workspace from JSON
   std::stringstream ss(s);
   return importJSON(ss);
}

Bool_t RooJSONFactoryWSTool::importYMLfromString(const std::string &s)
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

Bool_t RooJSONFactoryWSTool::exportJSON(std::ostream &os)
{
   // export the workspace in JSON
   tree_t p;
   JSONNode &n = p.rootnode();
   n.set_map();
   this->exportAll(n);
   n.writeJSON(os);
   return true;
}
Bool_t RooJSONFactoryWSTool::exportJSON(std::string const& filename)
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

Bool_t RooJSONFactoryWSTool::exportYML(std::ostream &os)
{
   // export the workspace in YML
   tree_t p;
   JSONNode &n = p.rootnode();
   n.set_map();
   this->exportAll(n);
   n.writeYML(os);
   return true;
}
Bool_t RooJSONFactoryWSTool::exportYML(std::string const& filename)
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

void RooJSONFactoryWSTool::prepare()
{
   gROOT->ProcessLine("using namespace RooStats::HistFactory;");
}

Bool_t RooJSONFactoryWSTool::importJSON(std::istream &is)
{
   // import a JSON file to the workspace
   try {
      tree_t p(is);
      JSONNode &n = p.rootnode();
      this->prepare();
      this->importDependants(n);
      if (n.has_child("data")) {
         auto data = this->loadData(n["data"]);
         for (const auto &d : data) {
            this->_workspace->import(*d.second);
         }
      }
   } catch (const std::exception &ex) {
      std::cerr << "unable to import JSON: " << ex.what() << std::endl;
      return false;
   }
   return true;
}
Bool_t RooJSONFactoryWSTool::importJSON(std::string const& filename)
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

Bool_t RooJSONFactoryWSTool::importYML(std::istream &is)
{
   // import a YML file to the workspace
   try {
      tree_t p(is);
      JSONNode &n = p.rootnode();
      this->prepare();
      this->importDependants(n);
   } catch (const std::exception &ex) {
      std::cerr << "unable to import JSON: " << ex.what() << std::endl;
      return false;
   }
   return true;
}
Bool_t RooJSONFactoryWSTool::importYML(std::string const& filename)
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
