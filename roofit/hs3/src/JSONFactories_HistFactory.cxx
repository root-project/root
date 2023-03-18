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

#include <RooFitHS3/HistFactoryJSONTool.h>
#include <RooFitHS3/RooJSONFactoryWSTool.h>
#include <RooFitHS3/JSONIO.h>
#include <RooFit/Detail/JSONInterface.h>

#include <RooStats/HistFactory/ParamHistFunc.h>
#include <RooStats/HistFactory/PiecewiseInterpolation.h>
#include <RooStats/HistFactory/FlexibleInterpVar.h>
#include <RooConstVar.h>
#include <RooCategory.h>
#include <RooRealVar.h>
#include <RooDataHist.h>
#include <RooHistFunc.h>
#include <RooRealSumPdf.h>
#include <RooBinWidthFunction.h>
#include <RooProdPdf.h>
#include <RooPoisson.h>
#include <RooGaussian.h>
#include <RooProduct.h>
#include <RooWorkspace.h>

#include <TH1.h>

#include <stack>

#include "static_execute.h"

using RooFit::Detail::JSONNode;

namespace {

double round_prec(double d, int nSig)
{
   if (d == 0.0)
      return 0.0;
   int ndigits = floor(log10(std::abs(d))) + 1 - nSig;
   double sf = pow(10, ndigits);
   if (std::abs(d / sf) < 2)
      ndigits--;
   return sf * round(d / sf);
}

// To avoid repeating the same string literals that can potentially get out of
// sync.
namespace Literals {
constexpr auto staterror = "staterror";
}

static bool startsWith(std::string_view str, std::string_view prefix)
{
   return str.size() >= prefix.size() && 0 == str.compare(0, prefix.size(), prefix);
}
static bool endsWith(std::string_view str, std::string_view suffix)
{
   return str.size() >= suffix.size() && 0 == str.compare(str.size() - suffix.size(), suffix.size(), suffix);
}

class Scope {
public:
   void setObservables(const RooArgList &args) { _observables.add(args); }
   void setObject(const std::string &name, RooAbsArg *obj) { _objects[name] = obj; }
   RooAbsArg *getObject(const std::string &name) const
   {
      auto f = _objects.find(name);
      return f == _objects.end() ? nullptr : f->second;
   }

   void getObservables(RooArgSet &out) const { out.add(_observables); }

private:
   RooArgList _observables;
   std::map<std::string, RooAbsArg *> _objects;
};

std::unique_ptr<TH1> histFunc2TH1(const RooHistFunc *hf)
{
   if (!hf)
      RooJSONFactoryWSTool::error("null pointer passed to histFunc2TH1");
   const RooDataHist &dh = hf->dataHist();
   std::unique_ptr<RooArgSet> vars{hf->getVariables()};
   std::unique_ptr<TH1> hist{hf->createHistogram(RooJSONFactoryWSTool::concat(vars.get()))};
   hist->SetDirectory(nullptr);
   auto volumes = dh.binVolumes(0, dh.numEntries());
   for (size_t i = 0; i < volumes.size(); ++i) {
      hist->SetBinContent(i + 1, hist->GetBinContent(i + 1) / volumes[i]);
      hist->SetBinError(i + 1, std::sqrt(hist->GetBinContent(i + 1)));
   }
   return hist;
}

template <class T>
T *findClient(RooAbsArg *gamma)
{
   for (const auto &client : gamma->clients()) {
      if (auto casted = dynamic_cast<T *>(client)) {
         return casted;
      } else {
         T *c = findClient<T>(client);
         if (c)
            return c;
      }
   }
   return nullptr;
}

template <class Arg_t, class... Params_t>
Arg_t &getOrCreate(RooWorkspace &ws, std::string const &name, Params_t &&...params)
{
   Arg_t *arg = static_cast<Arg_t *>(ws.obj(name));
   if (arg)
      return *arg;
   Arg_t newArg(name.c_str(), name.c_str(), std::forward<Params_t>(params)...);
   ws.import(newArg, RooFit::RecycleConflictNodes(true), RooFit::Silence(true));
   return *static_cast<Arg_t *>(ws.obj(name));
}

RooRealVar &getNP(RooWorkspace &ws, std::string const &parname)
{
   RooRealVar &par = getOrCreate<RooRealVar>(ws, parname, 0., -5, 5);
   par.setAttribute("np");
   std::string globname = "nom_" + parname;
   RooRealVar &nom = getOrCreate<RooRealVar>(ws, globname, 0.);
   nom.setAttribute("glob");
   nom.setRange(-10, 10);
   nom.setConstant(true);
   return par;
}
RooAbsPdf &getConstraint(RooWorkspace &ws, const std::string &sysname, const std::string &pname)
{
   return getOrCreate<RooGaussian>(ws, sysname + "_constraint", *ws.var(pname), *ws.var("nom_" + pname), 1.);
}

/// Convenient alternative to std::make_unique if you construct a RooFit
/// argument where name and title are the same.
template <class Obj_t, typename... Args_t>
std::unique_ptr<Obj_t> makeUnique(RooStringView name, Args_t &&...args)
{
   return std::make_unique<Obj_t>(name, name, std::forward<Args_t>(args)...);
}

std::unique_ptr<ParamHistFunc>
createPHF(const std::string &sysname, const std::string &phfname, const std::vector<double> &vals,
          RooJSONFactoryWSTool &tool, RooArgList &constraints, const RooArgSet &observables,
          const std::string &constraintType, RooArgList &gammas, double gamma_min, double gamma_max)
{
   RooWorkspace &ws = *tool.workspace();

   std::string funcParams = "gamma_" + sysname;
   gammas.add(ParamHistFunc::createParamSet(ws, funcParams.c_str(), observables, gamma_min, gamma_max));
   auto phf = makeUnique<ParamHistFunc>(phfname, observables, gammas);
   for (auto &g : gammas) {
      g->setAttribute("np");
   }

   if (constraintType == "Gauss") {
      for (size_t i = 0; i < vals.size(); ++i) {
         std::string basename = gammas[i].GetName();
         std::string nomname = "nom_" + basename;
         std::string poisname = basename + "_constraint";
         std::string sname = basename + "_sigma";
         auto nom = makeUnique<RooRealVar>(nomname, 1);
         nom->setAttribute("glob");
         nom->setConstant(true);
         nom->setRange(0, std::max(10., gamma_max));
         auto sigma = makeUnique<RooConstVar>(sname, vals[i]);
         auto g = static_cast<RooRealVar *>(gammas.at(i));
         auto gaus = makeUnique<RooGaussian>(poisname, *nom, *g, *sigma);
         gaus->addOwnedComponents(std::move(nom), std::move(sigma));
         tool.wsImport(*gaus);
         constraints.add(*ws.pdf(gaus->GetName()), true);
      }
   } else if (constraintType == "Poisson") {
      for (size_t i = 0; i < vals.size(); ++i) {
         double tau_float = vals[i];
         std::string basename = gammas[i].GetName();
         std::string tname = basename + "_tau";
         std::string nomname = "nom_" + basename;
         std::string prodname = basename + "_poisMean";
         std::string poisname = basename + "_constraint";
         auto tau = makeUnique<RooConstVar>(tname, tau_float);
         auto nom = makeUnique<RooRealVar>(nomname, tau_float);
         nom->setAttribute("glob");
         nom->setConstant(true);
         nom->setMin(0);
         RooArgSet elems{gammas[i], *tau};
         auto prod = makeUnique<RooProduct>(prodname, elems);
         auto pois = makeUnique<RooPoisson>(poisname, *nom, *prod);
         pois->addOwnedComponents(std::move(tau), std::move(nom), std::move(prod));
         pois->setNoRounding(true);
         tool.wsImport(*pois);
         constraints.add(*ws.pdf(pois->GetName()), true);
      }
   } else {
      RooJSONFactoryWSTool::error("unknown constraint type " + constraintType);
   }
   for (auto &g : gammas) {
      for (auto client : g->clients()) {
         if (dynamic_cast<RooAbsPdf *>(client) && !constraints.find(*client)) {
            constraints.add(*client);
         }
      }
   }
   phf->recursiveRedirectServers(observables);

   return phf;
}

std::unique_ptr<ParamHistFunc> createPHFMCStat(std::string name, const std::vector<double> &sumW,
                                               const std::vector<double> &sumW2, RooJSONFactoryWSTool &tool,
                                               RooArgList &constraints, const RooArgSet &observables,
                                               double statErrorThreshold, const std::string &statErrorType)
{
   if (sumW.empty())
      return nullptr;

   if (startsWith(name, "model_")) {
      name.erase(0, 6);
   }

   RooArgList gammas;
   std::string phfname = std::string("mc_stat_") + name;
   std::string sysname = std::string("stat_") + name;
   std::vector<double> vals(sumW.size());
   std::vector<double> errs(sumW.size());

   for (size_t i = 0; i < sumW.size(); ++i) {
      errs[i] = std::sqrt(sumW2[i]) / sumW[i];
      if (statErrorType == "Gauss") {
         vals[i] = std::max(errs[i], 0.); // avoid negative sigma. This NP will be set constant anyway later
      } else if (statErrorType == "Poisson") {
         vals[i] = sumW[i] * sumW[i] / sumW2[i];
      }
   }

   auto phf = createPHF(sysname, phfname, vals, tool, constraints, observables, statErrorType, gammas, 0, 10);

   // set constant NPs which are below the MC stat threshold, and remove them from the np list
   for (size_t i = 0; i < sumW.size(); ++i) {
      auto g = static_cast<RooRealVar *>(gammas.at(i));
      g->setError(errs[i]);
      if (errs[i] < statErrorThreshold) {
         g->setConstant(true); // all negative errs are set constant
      }
   }

   return phf;
}

bool hasStaterror(const JSONNode &comp)
{
   if (!comp.has_child("modifiers"))
      return false;
   for (const auto &mod : comp["modifiers"].children()) {
      if (mod["type"].val() == ::Literals::staterror)
         return true;
   }
   return false;
}

bool importHistSample(RooJSONFactoryWSTool &tool, RooDataHist &dh, Scope &scope, const std::string &fprefix,
                      const JSONNode &p, RooArgList &constraints)
{
   RooWorkspace &ws = *tool.workspace();

   std::string name = p["name"].val();
   std::string prefixedName = fprefix + "_" + name;

   if (!p.has_child("data")) {
      RooJSONFactoryWSTool::error("sample '" + name + "' does not define a 'data' key");
   }

   std::stack<std::unique_ptr<RooAbsArg>> ownedArgsStack;

   RooArgSet varlist;
   scope.getObservables(varlist);

   auto hf = std::make_unique<RooHistFunc>(("hist_" + prefixedName).c_str(), RooJSONFactoryWSTool::name(p).c_str(),
                                           varlist, dh);

   RooArgList shapeElems;
   RooArgList normElems;

   auto bwf = makeUnique<RooBinWidthFunction>(prefixedName + "_binWidth", *hf, true);
   shapeElems.add(*bwf);

   if (hasStaterror(p)) {
      if (RooAbsArg *phf = scope.getObject("mcstat")) {
         shapeElems.add(*phf);
      } else {
         RooJSONFactoryWSTool::error("sample '" + name +
                                     "' has 'staterror' active, but no element called 'mcstat' in scope!");
      }
   }

   if (p.has_child("modifiers")) {
      RooArgList overall_nps;
      std::vector<double> overall_low;
      std::vector<double> overall_high;

      RooArgList histNps;
      RooArgList histoLo;
      RooArgList histoHi;

      for (const auto &mod : p["modifiers"].children()) {
         std::string modtype = mod["type"].val();
         if (modtype == "normfactor") {
            normElems.add(getOrCreate<RooRealVar>(ws, mod["name"].val(), 1., -3, 5));
         } else if (modtype == "normsys") {
            std::string sysname(mod["name"].val());
            std::string parname(mod.has_child("parameter") ? RooJSONFactoryWSTool::name(mod["parameter"])
                                                           : "alpha_" + sysname);
            overall_nps.add(::getNP(ws, parname));
            auto &data = mod["data"];
            overall_low.push_back(data["lo"].val_double());
            overall_high.push_back(data["hi"].val_double());
            constraints.add(getConstraint(ws, sysname, parname));
         } else if (modtype == "histosys") {
            std::string sysname(mod["name"].val());
            std::string parname(mod.has_child("parameter") ? RooJSONFactoryWSTool::name(mod["parameter"])
                                                           : "alpha_" + sysname);
            histNps.add(::getNP(ws, parname));
            auto &data = mod["data"];
            histoLo.addOwned(makeUnique<RooHistFunc>(
               sysname + "Low_" + prefixedName, varlist,
               RooJSONFactoryWSTool::readBinnedData(data["lo"], sysname + "Low_" + prefixedName, varlist)));
            histoHi.addOwned(makeUnique<RooHistFunc>(
               sysname + "High_" + prefixedName, varlist,
               RooJSONFactoryWSTool::readBinnedData(data["hi"], sysname + "High_" + prefixedName, varlist)));
            constraints.add(getConstraint(ws, sysname, parname));
         } else if (modtype == "shapesys") {
            std::string sysname(mod["name"].val());
            std::string funcName = prefixedName + "_" + sysname + "_ShapeSys";
            std::vector<double> vals;
            for (const auto &v : mod["data"]["vals"].children()) {
               vals.push_back(v.val_double());
            }
            RooArgList gammas;
            tool.wsImport(*createPHF(sysname, funcName, vals, tool, constraints, varlist, mod["constraint"].val(),
                                     gammas, 0, 1000));
            auto phf = ws.function(funcName);
            shapeElems.add(*phf);
         }
      }

      if (!overall_nps.empty()) {
         auto v = makeUnique<RooStats::HistFactory::FlexibleInterpVar>("overallSys_" + prefixedName, overall_nps, 1.,
                                                                       overall_low, overall_high);
         v->setAllInterpCodes(4); // default HistFactory interpCode
         normElems.add(*v);
         ownedArgsStack.push(std::move(v));
      }
      if (!histNps.empty()) {
         auto v = makeUnique<PiecewiseInterpolation>("histoSys_" + prefixedName, *hf, histoLo, histoHi, histNps, false);
         v->setAllInterpCodes(4); // default interpCode for HistFactory
         shapeElems.add(*v);
         ownedArgsStack.push(std::move(v));
      } else {
         shapeElems.add(*hf);
      }
   }

   tool.wsEmplace<RooProduct>(prefixedName + "_shapes", prefixedName + "_shapes", shapeElems);
   if (!normElems.empty()) {
      tool.wsEmplace<RooProduct>(prefixedName + "_scaleFactors", prefixedName + "_scaleFactors", normElems);
   } else {
      ws.factory("RooConstVar::" + prefixedName + "_scaleFactors(1.)");
   }

   return true;
}

class HistFactoryImporter : public RooFit::JSONIO::Importer {
public:
   bool importPdf(RooJSONFactoryWSTool *tool, const JSONNode &p) const override
   {
      RooWorkspace &ws = *tool->workspace();
      Scope scope;

      std::string name = p["name"].val();
      RooArgList funcs;
      RooArgList coefs;
      RooArgList constraints;
      if (!p.has_child("samples")) {
         RooJSONFactoryWSTool::error("no samples in '" + name + "', skipping.");
      }
      std::vector<std::string> usesStatError;
      double statErrorThreshold = 0;
      std::string statErrorType = "Poisson";
      if (p.has_child(::Literals::staterror)) {
         auto &staterr = p[::Literals::staterror];
         if (staterr.has_child("relThreshold"))
            statErrorThreshold = staterr["relThreshold"].val_double();
         if (staterr.has_child("constraint"))
            statErrorType = staterr["constraint"].val();
      }
      std::vector<double> sumW;
      std::vector<double> sumW2;
      std::vector<std::string> funcnames;
      std::vector<std::string> coefnames;
      RooArgSet observables;
      for (auto const &obsNode : p["axes"].children()) {
         RooRealVar &obs = getOrCreate<RooRealVar>(ws, obsNode["name"].val(), obsNode["min"].val_double(),
                                                   obsNode["max"].val_double());
         obs.setBins(obsNode["nbins"].val_int());
         observables.add(obs);
      }
      scope.setObservables(observables);

      std::string fprefix = name;

      std::vector<std::unique_ptr<RooDataHist>> data;
      for (const auto &comp : p["samples"].children()) {
         RooArgSet varlist;
         scope.getObservables(varlist);
         std::unique_ptr<RooDataHist> dh = RooJSONFactoryWSTool::readBinnedData(
            comp["data"], fprefix + "_" + comp["name"].val() + "_dataHist", varlist);
         size_t nbins = dh->numEntries();

         if (hasStaterror(comp)) {
            if (sumW.empty()) {
               sumW.resize(nbins);
               sumW2.resize(nbins);
            }
            for (size_t i = 0; i < nbins; ++i) {
               sumW[i] += dh->weight(i);
               sumW2[i] += dh->weightSquared(i);
            }
         }
         data.emplace_back(std::move(dh));
      }

      if (!sumW.empty()) {
         auto phf =
            createPHFMCStat(name, sumW, sumW2, *tool, constraints, observables, statErrorThreshold, statErrorType);
         if (phf) {
            tool->wsImport(*phf);
            scope.setObject("mcstat", ws.function(phf->GetName()));
         }
      }

      int idx = 0;
      for (const auto &comp : p["samples"].children()) {
         std::string fname(fprefix + "_" + comp["name"].val() + "_shapes");
         std::string coefname(fprefix + "_" + comp["name"].val() + "_scaleFactors");

         importHistSample(*tool, *data[idx], scope, fprefix, comp, constraints);
         ++idx;

         funcs.add(*tool->request<RooAbsReal>(fname, name));
         coefs.add(*tool->request<RooAbsReal>(coefname, name));
      }

      if (constraints.empty()) {
         auto sum = makeUnique<RooRealSumPdf>(name, funcs, coefs, true);
         sum->setAttribute("BinnedLikelihood");
         tool->wsImport(*sum);
      } else {
         auto sum = std::make_unique<RooRealSumPdf>((name + "_model").c_str(), name.c_str(), funcs, coefs, true);
         sum->setAttribute("BinnedLikelihood");
         tool->wsImport(*sum);
         tool->wsEmplace<RooProdPdf>(name, name, constraints, RooFit::Conditional(*sum, observables));
      }
      return true;
   }
};

class FlexibleInterpVarStreamer : public RooFit::JSONIO::Exporter {
public:
   std::string const &key() const override
   {
      static const std::string keystring = "interpolation0d";
      return keystring;
   }
   bool exportObject(RooJSONFactoryWSTool *, const RooAbsArg *func, JSONNode &elem) const override
   {
      auto fip = static_cast<const RooStats::HistFactory::FlexibleInterpVar *>(func);
      elem["type"] << key();
      elem["vars"].fill_seq(fip->variables(), [](auto const &item) { return item->GetName(); });
      elem["interpolationCodes"].fill_seq(fip->interpolationCodes());
      elem["nom"] << fip->nominal();
      elem["high"].fill_seq(fip->high());
      elem["low"].fill_seq(fip->low());
      return true;
   }
};

class PiecewiseInterpolationStreamer : public RooFit::JSONIO::Exporter {
public:
   std::string const &key() const override
   {
      static const std::string keystring = "interpolation";
      return keystring;
   }
   bool exportObject(RooJSONFactoryWSTool *, const RooAbsArg *func, JSONNode &elem) const override
   {
      const PiecewiseInterpolation *pip = static_cast<const PiecewiseInterpolation *>(func);
      elem["type"] << key();
      elem["interpolationCodes"].fill_seq(pip->interpolationCodes());
      elem["positiveDefinite"] << pip->positiveDefinite();
      elem["vars"].fill_seq(pip->paramList(), [](auto const &item) { return item->GetName(); });
      auto &nom = elem["nom"];
      nom << pip->nominalHist()->GetName();

      elem["high"].fill_seq(pip->highList(), [](auto const &item) { return item->GetName(); });
      elem["low"].fill_seq(pip->lowList(), [](auto const &item) { return item->GetName(); });
      return true;
   }
};

class PiecewiseInterpolationFactory : public RooFit::JSONIO::Importer {
public:
   bool importFunction(RooJSONFactoryWSTool *tool, const JSONNode &p) const override
   {
      std::string name(RooJSONFactoryWSTool::name(p));

      RooArgList vars{tool->requestArgList<RooRealVar>(p, "vars")};

      PiecewiseInterpolation pip(name.c_str(), name.c_str(), *tool->requestArg<RooAbsReal>(p, "nom"),
                                 tool->requestArgList<RooAbsReal>(p, "low"),
                                 tool->requestArgList<RooAbsReal>(p, "high"), vars);

      pip.setPositiveDefinite(p["positiveDefinite"].val_bool());

      if (p.has_child("interpolationCodes")) {
         for (size_t i = 0; i < vars.size(); ++i) {
            pip.setInterpCode(*static_cast<RooAbsReal *>(vars.at(i)), p["interpolationCodes"][i].val_int(), true);
         }
      }

      tool->wsImport(pip);
      return true;
   }
};

class FlexibleInterpVarFactory : public RooFit::JSONIO::Importer {
public:
   bool importFunction(RooJSONFactoryWSTool *tool, const JSONNode &p) const override
   {
      std::string name(RooJSONFactoryWSTool::name(p));
      if (!p.has_child("high")) {
         RooJSONFactoryWSTool::error("no high variations of '" + name + "'");
      }
      if (!p.has_child("low")) {
         RooJSONFactoryWSTool::error("no low variations of '" + name + "'");
      }
      if (!p.has_child("nom")) {
         RooJSONFactoryWSTool::error("no nominal variation of '" + name + "'");
      }

      double nom(p["nom"].val_double());

      RooArgList vars{tool->requestArgList<RooRealVar>(p, "vars")};

      std::vector<double> high;
      high << p["high"];

      std::vector<double> low;
      low << p["low"];

      if (vars.size() != low.size() || vars.size() != high.size()) {
         RooJSONFactoryWSTool::error("FlexibleInterpVar '" + name +
                                     "' has non-matching lengths of 'vars', 'high' and 'low'!");
      }

      auto fip = makeUnique<RooStats::HistFactory::FlexibleInterpVar>(name, vars, nom, low, high);

      if (p.has_child("interpolationCodes")) {
         for (size_t i = 0; i < vars.size(); ++i) {
            fip->setInterpCode(*static_cast<RooAbsReal *>(vars.at(i)), p["interpolationCodes"][i].val_int());
         }
      }

      tool->wsImport(*fip);
      return true;
   }
};

void collectElements(RooArgSet &elems, RooAbsArg *arg)
{
   if (auto prod = dynamic_cast<RooProduct *>(arg)) {
      for (const auto &e : prod->components()) {
         collectElements(elems, e);
      }
   } else {
      elems.add(*arg);
   }
}

bool tryExportHistFactory(const std::string &pdfname, const std::string chname, const RooRealSumPdf *sumpdf,
                          JSONNode &elem)
{
   if (!sumpdf)
      return false;

   for (RooAbsArg *sample : sumpdf->funcList()) {
      if (!dynamic_cast<RooProduct *>(sample) && !dynamic_cast<RooRealSumPdf *>(sample)) {
         return false;
      }
   }

   std::map<int, double> tot_yield;
   std::map<int, double> tot_yield2;
   std::map<int, double> rel_errors;
   std::map<std::string, std::unique_ptr<TH1>> bb_histograms;
   std::map<std::string, std::unique_ptr<TH1>> nonbb_histograms;
   std::vector<std::string> varnames;

   struct NormSys {
      std::string name;
      double low;
      double high;
      NormSys(const std::string &n, double h, double l) : name(n), low(l), high(h) {}
   };
   struct HistoSys {
      std::string name;
      std::unique_ptr<TH1> high;
      std::unique_ptr<TH1> low;
      HistoSys(const std::string &n, RooHistFunc *h, RooHistFunc *l)
         : name(n), high(histFunc2TH1(h)), low(histFunc2TH1(l)){};
   };
   struct ShapeSys {
      std::string name;
      std::vector<double> constraints;
      bool isPoisson = false;
      ShapeSys(const std::string &n) : name(n){};
   };
   struct Sample {
      std::string name;
      std::unique_ptr<TH1> hist = nullptr;
      std::vector<const RooAbsArg *> norms;
      std::vector<NormSys> normsys;
      std::vector<HistoSys> histosys;
      std::vector<ShapeSys> shapesys;
      bool dobb = false;
      bool bbpoisson = true;
      Sample(const std::string &n) : name(n){};
   };
   std::vector<Sample> samples;

   for (size_t sampleidx = 0; sampleidx < sumpdf->funcList().size(); ++sampleidx) {
      PiecewiseInterpolation *pip = nullptr;
      RooStats::HistFactory::FlexibleInterpVar *fip = nullptr;
      std::vector<ParamHistFunc *> phfs;

      const auto func = sumpdf->funcList().at(sampleidx);
      const auto coef = sumpdf->coefList().at(sampleidx);
      Sample sample(func->GetName());
      if (startsWith(sample.name, "L_x_"))
         sample.name = sample.name.substr(4);
      if (endsWith(sample.name, "_shapes"))
         sample.name = sample.name.substr(0, sample.name.size() - 7);
      if (endsWith(sample.name, "_" + chname))
         sample.name = sample.name.substr(0, sample.name.size() - chname.size() - 1);
      if (startsWith(sample.name, pdfname + "_"))
         sample.name = sample.name.substr(pdfname.size() + 1);
      RooArgSet elems;
      collectElements(elems, func);
      collectElements(elems, coef);

      for (RooAbsArg *e : elems) {
         if (auto constVar = dynamic_cast<RooConstVar *>(e)) {
            if (constVar->getVal() == 1.)
               continue;
            sample.norms.push_back(e);
         } else if (dynamic_cast<RooRealVar *>(e)) {
            sample.norms.push_back(e);
         } else if (auto hf = dynamic_cast<const RooHistFunc *>(e)) {
            if (varnames.empty()) {
               varnames = RooJSONFactoryWSTool::names(*hf->dataHist().get());
            }
            if (!sample.hist) {
               sample.hist = histFunc2TH1(hf);
            }
         } else if (auto phf = dynamic_cast<ParamHistFunc *>(e)) {
            phfs.push_back(phf);
         } else {
            if (!fip) {
               fip = dynamic_cast<RooStats::HistFactory::FlexibleInterpVar *>(e);
            }
            if (!pip) {
               pip = dynamic_cast<PiecewiseInterpolation *>(e);
            }
         }
      }

      // see if we can get the varnames
      if (pip) {
         if (auto nh = dynamic_cast<RooHistFunc const *>(pip->nominalHist())) {
            if (!sample.hist)
               sample.hist = histFunc2TH1(nh);
            if (varnames.empty())
               varnames = RooJSONFactoryWSTool::names(*nh->dataHist().get());
         }
      }

      // sort and configure norms
      std::sort(sample.norms.begin(), sample.norms.end(),
                [](auto &l, auto &r) { return strcmp(l->GetName(), r->GetName()) < 0; });

      // sort and configure the normsys
      if (fip) {
         for (size_t i = 0; i < fip->variables().size(); ++i) {
            std::string sysname(fip->variables().at(i)->GetName());
            if (sysname.find("alpha_") == 0) {
               sysname = sysname.substr(6);
            }
            sample.normsys.emplace_back(NormSys(sysname, fip->high()[i], fip->low()[i]));
         }
         std::sort(sample.normsys.begin(), sample.normsys.end(), [](auto &l, auto &r) { return l.name < r.name; });
      }

      // sort and configure the histosys
      if (pip) {
         std::vector<HistoSys> histosys;
         for (size_t i = 0; i < pip->paramList().size(); ++i) {
            std::string sysname(pip->paramList().at(i)->GetName());
            if (sysname.find("alpha_") == 0) {
               sysname = sysname.substr(6);
            }
            if (auto lo = dynamic_cast<RooHistFunc *>(pip->lowList().at(i))) {
               if (auto hi = dynamic_cast<RooHistFunc *>(pip->highList().at(i))) {
                  histosys.emplace_back(HistoSys(sysname, lo, hi));
               }
            }
         }
         std::sort(sample.histosys.begin(), sample.histosys.end(),
                   [](auto const &l, auto const &r) { return l.name < r.name; });
      }

      // check if we have everything
      if (!sample.hist) {
         std::cout << "unable to find hist" << std::endl;
         return false;
      }

      for (ParamHistFunc *phf : phfs) {
         if (startsWith(std::string(phf->GetName()), "mc_stat_")) { // MC stat uncertainty
            int idx = 0;
            for (const auto &g : phf->paramList()) {
               ++idx;
               RooPoisson *constraint_p = findClient<RooPoisson>(g);
               RooGaussian *constraint_g = findClient<RooGaussian>(g);
               if (tot_yield.find(idx) == tot_yield.end()) {
                  tot_yield[idx] = 0;
                  tot_yield2[idx] = 0;
               }
               tot_yield[idx] += sample.hist->GetBinContent(idx);
               tot_yield2[idx] += (sample.hist->GetBinContent(idx) * sample.hist->GetBinContent(idx));
               if (constraint_p) {
                  double erel = 1. / std::sqrt(constraint_p->getX().getVal());
                  rel_errors[idx] = erel;
                  sample.bbpoisson = true;
               } else if (constraint_g) {
                  double erel = constraint_g->getSigma().getVal() / constraint_g->getMean().getVal();
                  rel_errors[idx] = erel;
                  sample.bbpoisson = false;
               }
            }
            sample.dobb = true;
         } else { // other ShapeSys
            ShapeSys sys(phf->GetName());
            if (endsWith(sys.name, "_ShapeSys")) {
               sys.name.erase(sys.name.size() - 9);
            }
            if (startsWith(sys.name, "model_" + chname + "_")) {
               sys.name.erase(0, chname.size() + 7);
            }
            if (startsWith(sys.name, chname + "_")) {
               sys.name.erase(0, chname.size() + 1);
            }
            if (startsWith(sys.name, sample.name + "_")) {
               sys.name.erase(0, sample.name.size() + 1);
            }
            if (endsWith(sys.name, "_" + chname)) {
               sys.name.erase(sys.name.size() - chname.size() - 1);
            }
            if (endsWith(sys.name, "_" + sample.name)) {
               sys.name.erase(sys.name.size() - sample.name.size() - 1);
            }

            for (const auto &g : phf->paramList()) {
               if (RooPoisson *constraint_p = findClient<RooPoisson>(g)) {
                  sys.isPoisson = true;
                  sys.constraints.push_back(constraint_p->getX().getVal());
               } else if (RooGaussian *constraint_g = findClient<RooGaussian>(g)) {
                  sys.isPoisson = false;
                  sys.constraints.push_back(constraint_g->getSigma().getVal() / constraint_g->getMean().getVal());
               }
            }
            sample.shapesys.emplace_back(std::move(sys));
         }
      }
      std::sort(sample.shapesys.begin(), sample.shapesys.end(), [](auto &l, auto &r) { return l.name < r.name; });

      // add the sample
      samples.emplace_back(std::move(sample));
   }

   std::sort(samples.begin(), samples.end(), [](auto const &l, auto const &r) { return l.name < r.name; });

   for (const auto &sample : samples) {
      if (sample.dobb) {
         for (auto bin : rel_errors) {
            // reverse engineering the correct partial error
            // the (arbitrary) convention used here is that all samples should have the same relative error
            const int i = bin.first;
            const double relerr_tot = bin.second;
            const double count = sample.hist->GetBinContent(i);
            // this reconstruction is inherently unprecise, so we truncate it at some decimal places to make sure that
            // we don't carry around too many useless digits
            sample.hist->SetBinError(i, round_prec(relerr_tot * tot_yield[i] / std::sqrt(tot_yield2[i]) * count, 7));
         }
      }
   }

   bool observablesWritten = false;
   for (const auto &sample : samples) {

      elem["type"] << "histfactory_dist";

      auto &s = RooJSONFactoryWSTool::appendNamedChild(elem["samples"], sample.name);

      auto &modifiers = s["modifiers"];
      modifiers.set_seq();

      for (const auto &nf : sample.norms) {
         RooJSONFactoryWSTool::appendNamedChild(modifiers, nf->GetName())["type"] << "normfactor";
      }

      for (const auto &sys : sample.normsys) {
         auto &mod = RooJSONFactoryWSTool::appendNamedChild(modifiers, sys.name);
         mod["type"] << "normsys";
         auto &data = mod["data"];
         data.set_map();
         data["lo"] << sys.low;
         data["hi"] << sys.high;
      }

      for (const auto &sys : sample.histosys) {
         auto &mod = RooJSONFactoryWSTool::appendNamedChild(modifiers, sys.name);
         mod["type"] << "histosys";
         auto &data = mod["data"];
         data.set_map();
         RooJSONFactoryWSTool::exportHistogram(*sys.low, data["lo"], varnames, nullptr, false, false);
         RooJSONFactoryWSTool::exportHistogram(*sys.high, data["hi"], varnames, nullptr, false, false);
      }

      for (const auto &sys : sample.shapesys) {
         auto &mod = RooJSONFactoryWSTool::appendNamedChild(modifiers, sys.name);
         mod["type"] << "shapesys";
         auto &data = mod["data"];
         data.set_map();
         auto &vals = data["vals"];
         vals.fill_seq(sys.constraints);
         mod["constraint"] << (sys.isPoisson ? "Poisson" : "Gauss");
      }

      if (sample.dobb) {
         auto &mod = RooJSONFactoryWSTool::appendNamedChild(modifiers, ::Literals::staterror);
         mod["type"] << ::Literals::staterror;
         mod["constraint"] << (sample.bbpoisson ? "Poisson" : "Gauss");
      }

      if (!observablesWritten) {
         RooJSONFactoryWSTool::writeObservables(*sample.hist, elem, varnames);
         observablesWritten = true;
      }
      auto &data = s["data"];
      RooJSONFactoryWSTool::exportHistogram(*sample.hist, data, varnames, nullptr, false, sample.dobb);
   }
   return true;
}

class HistFactoryStreamer_ProdPdf : public RooFit::JSONIO::Exporter {
public:
   bool autoExportDependants() const override { return false; }
   bool tryExport(const RooProdPdf *prodpdf, JSONNode &elem) const
   {
      RooRealSumPdf *sumpdf = nullptr;
      for (RooAbsArg *v : prodpdf->pdfList()) {
         sumpdf = dynamic_cast<RooRealSumPdf *>(v);
      }
      if (!sumpdf)
         return false;
      std::string chname(prodpdf->GetName());
      if (startsWith(chname, "model_")) {
         chname = chname.substr(6);
      }
      if (endsWith(chname, "_model")) {
         chname = chname.substr(0, chname.size() - 6);
      }

      return tryExportHistFactory(prodpdf->GetName(), chname, sumpdf, elem);
   }
   std::string const &key() const override
   {
      static const std::string keystring = "histfactory_dist";
      return keystring;
   }
   bool exportObject(RooJSONFactoryWSTool *, const RooAbsArg *p, JSONNode &elem) const override
   {
      return tryExport(static_cast<const RooProdPdf *>(p), elem);
   }
};

class HistFactoryStreamer_SumPdf : public RooFit::JSONIO::Exporter {
public:
   bool autoExportDependants() const override { return false; }
   bool tryExport(const RooRealSumPdf *sumpdf, JSONNode &elem) const
   {
      if (!sumpdf)
         return false;
      std::string chname(sumpdf->GetName());
      if (startsWith(chname, "model_")) {
         chname = chname.substr(6);
      }
      if (endsWith(chname, "_model")) {
         chname = chname.substr(0, chname.size() - 6);
      }
      return tryExportHistFactory(sumpdf->GetName(), chname, sumpdf, elem);
   }
   std::string const &key() const override
   {
      static const std::string keystring = "histfactory_dist";
      return keystring;
   }
   bool exportObject(RooJSONFactoryWSTool *, const RooAbsArg *p, JSONNode &elem) const override
   {
      return tryExport(static_cast<const RooRealSumPdf *>(p), elem);
   }
};

STATIC_EXECUTE([]() {
   using namespace RooFit::JSONIO;

   registerImporter<HistFactoryImporter>("histfactory_dist", true);
   registerImporter<PiecewiseInterpolationFactory>("interpolation", true);
   registerImporter<FlexibleInterpVarFactory>("interpolation0d", true);
   registerExporter<FlexibleInterpVarStreamer>(RooStats::HistFactory::FlexibleInterpVar::Class(), true);
   registerExporter<PiecewiseInterpolationStreamer>(PiecewiseInterpolation::Class(), true);
   registerExporter<HistFactoryStreamer_ProdPdf>(RooProdPdf::Class(), true);
   registerExporter<HistFactoryStreamer_SumPdf>(RooRealSumPdf::Class(), true);
});

} // namespace
