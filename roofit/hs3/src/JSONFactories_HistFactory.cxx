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
   void setObservables(const RooArgList &args)
   {
      for (auto *arg : args) {
         _observables.push_back(arg);
      }
   }
   void setObject(const std::string &name, RooAbsArg *obj) { _objects[name] = obj; }
   RooAbsArg *getObject(const std::string &name) const { return _objects.at(name); }

   void getObservables(RooArgSet &out) const { out.add(_observables.begin(), _observables.end()); }

private:
   std::vector<RooAbsArg *> _observables;
   std::map<std::string, RooAbsArg *> _objects;
};

inline void collectNames(const JSONNode &n, std::vector<std::string> &names)
{
   for (const auto &c : n.children()) {
      names.push_back(RooJSONFactoryWSTool::name(c));
   }
}

inline void stackError(const JSONNode &n, std::vector<double> &sumW, std::vector<double> &sumW2)
{
   if (!n.is_map())
      return;
   if (!n.has_child("contents"))
      throw std::invalid_argument("no contents given");
   JSONNode const &contents = n["contents"];
   if (!contents.is_seq())
      throw std::invalid_argument("contents are not in list form");
   if (!n.has_child("errors"))
      throw std::invalid_argument("no errors given");
   if (!n["errors"].is_seq())
      throw std::invalid_argument("errors are not in list form");
   if (contents.num_children() != n["errors"].num_children()) {
      throw std::invalid_argument("inconsistent bin numbers");
   }
   const size_t nbins = contents.num_children();
   for (size_t ibin = 0; ibin < nbins; ++ibin) {
      double w = contents[ibin].val_double();
      double e = n["errors"][ibin].val_double();
      if (ibin < sumW.size())
         sumW[ibin] += w;
      else
         sumW.push_back(w);
      if (ibin < sumW2.size())
         sumW2[ibin] += e * e;
      else
         sumW2.push_back(e * e);
   }
}

std::vector<std::string> getVarnames(const RooHistFunc *hf)
{
   const RooDataHist &dh = hf->dataHist();
   RooArgList vars(*dh.get());
   return RooJSONFactoryWSTool::names(&vars);
}

std::unique_ptr<TH1> histFunc2TH1(const RooHistFunc *hf)
{
   if (!hf)
      RooJSONFactoryWSTool::error("null pointer passed to histFunc2TH1");
   const RooDataHist &dh = hf->dataHist();
   RooArgSet *vars = hf->getVariables();
   auto varnames = RooJSONFactoryWSTool::names(vars);
   std::unique_ptr<TH1> hist{hf->createHistogram(RooJSONFactoryWSTool::concat(vars).c_str())};
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
      if (client->InheritsFrom(T::Class())) {
         return static_cast<T *>(client);
      } else {
         T *c = findClient<T>(client);
         if (c)
            return c;
      }
   }
   return nullptr;
}

RooRealVar *getNP(RooWorkspace &ws, const char *parname)
{
   RooRealVar *par = ws.var(parname);
   if (!ws.var(parname)) {
      par = static_cast<RooRealVar *>(ws.factory(std::string(parname) + "[0.,-5,5]"));
   }
   if (par) {
      par->setAttribute("np");
   }
   std::string globname = std::string("nom_") + parname;
   RooRealVar *nom = ws.var(globname);
   if (!nom) {
      nom = static_cast<RooRealVar *>(ws.factory(globname + "[0.]"));
   }
   if (nom) {
      nom->setAttribute("glob");
      nom->setRange(-5, 5);
      nom->setConstant(true);
   }
   std::string constrname = std::string("sigma_") + parname;
   RooRealVar *sigma = ws.var(constrname);
   if (!sigma) {
      sigma = static_cast<RooRealVar *>(ws.factory(constrname + "[1.]"));
   }
   if (sigma) {
      sigma->setRange(sigma->getVal(), sigma->getVal());
      sigma->setConstant(true);
   }
   if (!par)
      RooJSONFactoryWSTool::error(TString::Format("unable to find nuisance parameter '%s'", parname));
   return par;
}
RooAbsPdf *getConstraint(RooWorkspace &ws, const std::string &sysname, const std::string &parname)
{
   RooAbsPdf *pdf = ws.pdf((sysname + "_constraint").c_str());
   if (!pdf) {
      pdf = static_cast<RooAbsPdf *>(
         ws.factory(TString::Format("RooGaussian::%s_constraint(%s,nom_%s,sigma_%s)", sysname.c_str(), parname.c_str(),
                                    parname.c_str(), parname.c_str())
                       .Data()));
   }
   if (!pdf) {
      RooJSONFactoryWSTool::error(TString::Format("unable to find constraint term '%s'", sysname.c_str()));
   }
   return pdf;
}

std::unique_ptr<ParamHistFunc> createPHF(const std::string &sysname, const std::string &phfname,
                                         const std::vector<double> &vals, RooWorkspace &w, RooArgList &constraints,
                                         const RooArgSet &observables, const std::string &constraintType,
                                         RooArgList &gammas, double gamma_min, double gamma_max)
{
   RooArgList ownedComponents;

   std::string funcParams = "gamma_" + sysname;
   gammas.add(ParamHistFunc::createParamSet(w, funcParams.c_str(), observables, gamma_min, gamma_max));
   auto phf = std::make_unique<ParamHistFunc>(phfname.c_str(), phfname.c_str(), observables, gammas);
   for (auto &g : gammas) {
      g->setAttribute("np");
   }

   if (constraintType == "Gauss") {
      for (size_t i = 0; i < vals.size(); ++i) {
         TString nomname = TString::Format("nom_%s", gammas[i].GetName());
         TString poisname = TString::Format("%s_constraint", gammas[i].GetName());
         TString sname = TString::Format("%s_sigma", gammas[i].GetName());
         auto nom = std::make_unique<RooRealVar>(nomname.Data(), nomname.Data(), 1);
         nom->setAttribute("glob");
         nom->setConstant(true);
         nom->setRange(0, std::max(10., gamma_max));
         auto sigma = std::make_unique<RooConstVar>(sname.Data(), sname.Data(), vals[i]);
         auto g = static_cast<RooRealVar *>(gammas.at(i));
         auto gaus = std::make_unique<RooGaussian>(poisname.Data(), poisname.Data(), *nom, *g, *sigma);
         gaus->addOwnedComponents(std::move(nom), std::move(sigma));
         constraints.add(*gaus, true);
         ownedComponents.addOwned(std::move(gaus), true);
      }
   } else if (constraintType == "Poisson") {
      for (size_t i = 0; i < vals.size(); ++i) {
         double tau_float = vals[i];
         TString tname = TString::Format("%s_tau", gammas[i].GetName());
         TString nomname = TString::Format("nom_%s", gammas[i].GetName());
         TString prodname = TString::Format("%s_poisMean", gammas[i].GetName());
         TString poisname = TString::Format("%s_constraint", gammas[i].GetName());
         auto tau = std::make_unique<RooConstVar>(tname.Data(), tname.Data(), tau_float);
         auto nom = std::make_unique<RooRealVar>(nomname.Data(), nomname.Data(), tau_float);
         nom->setAttribute("glob");
         nom->setConstant(true);
         nom->setMin(0);
         RooArgSet elems{gammas[i], *tau};
         auto prod = std::make_unique<RooProduct>(prodname.Data(), prodname.Data(), elems);
         auto pois = std::make_unique<RooPoisson>(poisname.Data(), poisname.Data(), *nom, *prod);
         pois->addOwnedComponents(std::move(tau), std::move(nom), std::move(prod));
         pois->setNoRounding(true);
         constraints.add(*pois, true);
         ownedComponents.addOwned(std::move(pois), true);
      }
   } else {
      RooJSONFactoryWSTool::error("unknown constraint type " + constraintType);
   }
   for (auto &g : gammas) {
      for (auto client : g->clients()) {
         if (client->InheritsFrom(RooAbsPdf::Class()) && !constraints.find(*client)) {
            constraints.add(*client);
         }
      }
   }
   phf->recursiveRedirectServers(observables);
   // Transfer ownership of gammas and owned constraints to the ParamHistFunc
   phf->addOwnedComponents(std::move(ownedComponents));

   return phf;
}

std::unique_ptr<ParamHistFunc> createPHFMCStat(const std::string &name, const std::vector<double> &sumW,
                                               const std::vector<double> &sumW2, RooWorkspace &w,
                                               RooArgList &constraints, const RooArgSet &observables,
                                               double statErrorThreshold, const std::string &statErrorType)
{
   if (sumW.empty())
      return nullptr;

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

   auto phf = createPHF(sysname, phfname, vals, w, constraints, observables, statErrorType, gammas, 0, 10);

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

std::unique_ptr<ParamHistFunc> createPHFShapeSys(const JSONNode &p, const std::string &phfname, RooWorkspace &w,
                                                 RooArgList &constraints, const RooArgSet &observables)
{
   std::string sysname(RooJSONFactoryWSTool::name(p));
   std::vector<double> vals;
   for (const auto &v : p["data"]["vals"].children()) {
      vals.push_back(v.val_double());
   }
   RooArgList gammas;
   return createPHF(sysname, phfname, vals, w, constraints, observables, p["constraint"].val(), gammas, 0, 1000);
}

bool importHistSample(RooWorkspace &ws, Scope &scope, const JSONNode &p, RooArgList &constraints)
{
   std::string name(RooJSONFactoryWSTool::name(p));
   std::string prefix = RooJSONFactoryWSTool::genPrefix(p, true);
   if (prefix.size() > 0)
      name = prefix + name;
   if (!p.has_child("data")) {
      RooJSONFactoryWSTool::error("function '" + name + "' is of histogram type, but does not define a 'data' key");
   }

   std::stack<std::unique_ptr<RooAbsArg>> ownedArgsStack;
   RooArgSet shapeElems;
   RooArgSet normElems;
   RooArgSet varlist;
   scope.getObservables(varlist);

   auto getBinnedData = [&](std::string const &binnedDataName) -> RooDataHist & {
      auto *dh = dynamic_cast<RooDataHist *>(ws.embeddedData(binnedDataName));
      if (!dh) {
         auto dhForImport = RooJSONFactoryWSTool::readBinnedData(p["data"], binnedDataName, varlist);
         ws.import(*dhForImport, RooFit::Silence(true), RooFit::Embedded());
         dh = static_cast<RooDataHist *>(ws.embeddedData(dhForImport->GetName()));
      }
      return *dh;
   };

   RooDataHist &dh = getBinnedData(name);
   auto hf =
      std::make_unique<RooHistFunc>(("hist_" + name).c_str(), RooJSONFactoryWSTool::name(p).c_str(), *(dh.get()), dh);
   ownedArgsStack.push(std::make_unique<RooBinWidthFunction>(
      TString::Format("%s_binWidth", (!prefix.empty() ? prefix : name).c_str()).Data(),
      TString::Format("%s_binWidth", (!prefix.empty() ? prefix : name).c_str()).Data(), *hf, true));
   shapeElems.add(*ownedArgsStack.top());

   if (p.has_child("statError") && p["statError"].val_bool()) {
      RooAbsArg *phf = scope.getObject("mcstat");
      if (phf) {
         shapeElems.add(*phf);
      } else {
         RooJSONFactoryWSTool::error("function '" + name +
                                     "' has 'statError' active, but no element called 'mcstat' in scope!");
      }
   }

   if (p.has_child("modifiers")) {
      RooArgList overall_nps;
      std::vector<double> overall_low;
      std::vector<double> overall_high;

      RooArgList histo_nps;
      RooArgList histo_low;
      RooArgList histo_high;

      for (const auto &mod : p["modifiers"].children()) {
         std::string modtype = mod["type"].val();
         if (modtype == "normfactor") {
            std::string nfname(mod["name"].val());
            if (RooAbsReal *r = ws.var(nfname)) {
               normElems.add(*r);
            } else {
               normElems.add(*static_cast<RooRealVar *>(ws.factory(nfname + "[1.]")));
            }
         } else if (modtype == "normsys") {
            std::string sysname(mod["name"].val());
            std::string parname(mod.has_child("parameter") ? RooJSONFactoryWSTool::name(mod["parameter"])
                                                           : "alpha_" + sysname);
            if (RooRealVar *par = ::getNP(ws, parname.c_str())) {
               overall_nps.add(*par);
               auto &data = mod["data"];
               overall_low.push_back(data["lo"].val_double());
               overall_high.push_back(data["hi"].val_double());
               constraints.add(*getConstraint(ws, sysname, parname));
            } else {
               RooJSONFactoryWSTool::error("overall systematic '" + sysname + "' doesn't have a valid parameter!");
            }
         } else if (modtype == "histosys") {
            std::string sysname(mod["name"].val());
            std::string parname(mod.has_child("parameter") ? RooJSONFactoryWSTool::name(mod["parameter"])
                                                           : "alpha_" + sysname);
            RooAbsReal *par = ::getNP(ws, parname.c_str());
            histo_nps.add(*par);
            RooDataHist &dh_low = getBinnedData(sysname + "Low_" + name);
            ownedArgsStack.push(std::make_unique<RooHistFunc>(
               (sysname + "Low_" + name).c_str(), RooJSONFactoryWSTool::name(p).c_str(), *(dh_low.get()), dh_low));
            histo_low.add(*ownedArgsStack.top());
            RooDataHist &dh_high = getBinnedData(sysname + "High_" + name);
            ownedArgsStack.push(std::make_unique<RooHistFunc>(
               (sysname + "High_" + name).c_str(), RooJSONFactoryWSTool::name(p).c_str(), *(dh_high.get()), dh_high));
            histo_high.add(*ownedArgsStack.top());
            constraints.add(*getConstraint(ws, sysname, parname));
         } else if (modtype == "shapesys") {
            std::string sysname(mod["name"].val());
            std::string funcName = prefix + sysname + "_ShapeSys";
            RooAbsArg *phf = scope.getObject(funcName);
            if (!phf) {
               auto newphf = createPHFShapeSys(mod, funcName, ws, constraints, varlist);
               ws.import(*newphf, RooFit::RecycleConflictNodes(), RooFit::Silence(true));
               scope.setObject(funcName, ws.function(funcName.c_str()));
            }
            if (!phf) {
               RooJSONFactoryWSTool::error("PHF '" + funcName +
                                           "' should have been created but cannot be found in scope.");
            }
            shapeElems.add(*phf);
         }
      }

      if (overall_nps.size() > 0) {
         auto v = std::make_unique<RooStats::HistFactory::FlexibleInterpVar>(
            ("overallSys_" + name).c_str(), ("overallSys_" + name).c_str(), overall_nps, 1., overall_low, overall_high);
         v->setAllInterpCodes(4); // default HistFactory interpCode
         normElems.add(*v);
         ownedArgsStack.push(std::move(v));
      }
      if (histo_nps.size() > 0) {
         auto v = std::make_unique<PiecewiseInterpolation>(("histoSys_" + name).c_str(), ("histoSys_" + name).c_str(),
                                                           *hf, histo_low, histo_high, histo_nps, false);
         v->setAllInterpCodes(4); // default interpCode for HistFactory
         shapeElems.add(*v);
         ownedArgsStack.push(std::move(v));
      } else {
         shapeElems.add(*hf);
         ownedArgsStack.push(std::move(hf));
      }
   }

   RooProduct shape((name + "_shapes").c_str(), (name + "_shapes").c_str(), shapeElems);
   ws.import(shape, RooFit::RecycleConflictNodes(true), RooFit::Silence(true));
   if (normElems.size() > 0) {
      RooProduct norm((name + "_scaleFactors").c_str(), (name + "_scaleFactors").c_str(), normElems);
      ws.import(norm, RooFit::RecycleConflictNodes(true), RooFit::Silence(true));
   } else {
      ws.factory("RooConstVar::" + name + "_scaleFactors(1.)");
   }

   return true;
}

class HistFactoryImporter : public RooFit::JSONIO::Importer {
public:
   bool importPdf(RooJSONFactoryWSTool *tool, const JSONNode &p) const override
   {
      RooWorkspace &ws = *tool->workspace();

      Scope scope;

      std::string name(RooJSONFactoryWSTool::name(p));
      RooArgList funcs;
      RooArgList coefs;
      RooArgList constraints;
      if (!p.has_child("samples")) {
         RooJSONFactoryWSTool::error("no samples in '" + name + "', skipping.");
      }
      std::vector<std::string> usesStatError;
      double statErrorThreshold = 0;
      std::string statErrorType = "Poisson";
      if (p.has_child("statError")) {
         auto &staterr = p["statError"];
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
      if (p.has_child("variables")) {
         RooJSONFactoryWSTool::getObservables(ws, p, name, observables);
         scope.setObservables(observables);
      }
      for (const auto &comp : p["samples"].children()) {
         std::string fname(comp["name"].val() + "_shapes");
         std::string coefname(comp["name"].val() + "_scaleFactors");
         std::string fprefix = RooJSONFactoryWSTool::genPrefix(comp, true);

         if (observables.empty()) {
            RooJSONFactoryWSTool::getObservables(ws, comp["data"], fprefix, observables);
            scope.setObservables(observables);
         }

         for (const auto &sampleNode : p["samples"].children()) {
            importHistSample(ws, scope, sampleNode, constraints);
         }

         auto phf = createPHFMCStat(name, sumW, sumW2, ws, constraints, observables, statErrorThreshold, statErrorType);
         if (phf) {
            ws.import(*phf, RooFit::RecycleConflictNodes(), RooFit::Silence(true));
            scope.setObject("mcstat", ws.function(phf->GetName()));
         }

         RooAbsReal *func = tool->request<RooAbsReal>(fname.c_str(), name);
         funcs.add(*func);

         RooAbsReal *coef = tool->request<RooAbsReal>(coefname.c_str(), name);
         coefs.add(*coef);
      }

      if (constraints.empty()) {
         RooRealSumPdf sum(name.c_str(), name.c_str(), funcs, coefs, true);
         sum.setAttribute("BinnedLikelihood");
         ws.import(sum, RooFit::RecycleConflictNodes(true), RooFit::Silence(true));
      } else {
         RooRealSumPdf sum((name + "_model").c_str(), name.c_str(), funcs, coefs, true);
         sum.setAttribute("BinnedLikelihood");
         ws.import(sum, RooFit::RecycleConflictNodes(true), RooFit::Silence(true));
         RooArgList lhelems;
         lhelems.add(sum);
         RooProdPdf prod(name.c_str(), name.c_str(), RooArgSet(constraints), RooFit::Conditional(lhelems, observables));
         ws.import(prod, RooFit::RecycleConflictNodes(true), RooFit::Silence(true));
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
      RooWorkspace &ws = *tool->workspace();

      std::string name(RooJSONFactoryWSTool::name(p));
      if (!p.has_child("vars")) {
         RooJSONFactoryWSTool::error("no vars of '" + name + "'");
      }
      if (!p.has_child("high")) {
         RooJSONFactoryWSTool::error("no high variations of '" + name + "'");
      }
      if (!p.has_child("low")) {
         RooJSONFactoryWSTool::error("no low variations of '" + name + "'");
      }
      if (!p.has_child("nom")) {
         RooJSONFactoryWSTool::error("no nominal variation of '" + name + "'");
      }

      std::string nomname(p["nom"].val());
      RooAbsReal *nominal = tool->request<RooAbsReal>(nomname, name);

      RooArgList vars;
      for (const auto &d : p["vars"].children()) {
         std::string objname(RooJSONFactoryWSTool::name(d));
         RooRealVar *obj = tool->request<RooRealVar>(objname, name);
         vars.add(*obj);
      }

      RooArgList high;
      for (const auto &d : p["high"].children()) {
         std::string objname(RooJSONFactoryWSTool::name(d));
         RooAbsReal *obj = tool->request<RooAbsReal>(objname, name);
         high.add(*obj);
      }

      RooArgList low;
      for (const auto &d : p["low"].children()) {
         std::string objname(RooJSONFactoryWSTool::name(d));
         RooAbsReal *obj = tool->request<RooAbsReal>(objname, name);
         low.add(*obj);
      }

      PiecewiseInterpolation pip(name.c_str(), name.c_str(), *nominal, low, high, vars);

      pip.setPositiveDefinite(p["positiveDefinite"].val_bool());

      if (p.has_child("interpolationCodes")) {
         for (size_t i = 0; i < vars.size(); ++i) {
            pip.setInterpCode(*static_cast<RooAbsReal *>(vars.at(i)), p["interpolationCodes"][i].val_int(), true);
         }
      }

      ws.import(pip, RooFit::RecycleConflictNodes(true), RooFit::Silence(true));
      return true;
   }
};

class FlexibleInterpVarFactory : public RooFit::JSONIO::Importer {
public:
   bool importFunction(RooJSONFactoryWSTool *tool, const JSONNode &p) const override
   {
      std::string name(RooJSONFactoryWSTool::name(p));
      if (!p.has_child("vars")) {
         RooJSONFactoryWSTool::error("no vars of '" + name + "'");
      }
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

      RooArgList vars;
      for (const auto &d : p["vars"].children()) {
         std::string objname(RooJSONFactoryWSTool::name(d));
         RooRealVar *obj = tool->request<RooRealVar>(objname, name);
         vars.add(*obj);
      }

      std::vector<double> high;
      high << p["high"];

      std::vector<double> low;
      high << p["low"];

      RooStats::HistFactory::FlexibleInterpVar fip(name.c_str(), name.c_str(), vars, nom, low, high);

      if (p.has_child("interpolationCodes")) {
         for (size_t i = 0; i < vars.size(); ++i) {
            fip.setInterpCode(*static_cast<RooAbsReal *>(vars.at(i)), p["interpolationCodes"][i].val_int());
         }
      }

      tool->workspace()->import(fip, RooFit::RecycleConflictNodes(true), RooFit::Silence(true));
      return true;
   }
};

void collectElements(RooArgSet &elems, RooProduct *prod)
{
   for (const auto &e : prod->components()) {
      if (e->InheritsFrom(RooProduct::Class())) {
         collectElements(elems, (RooProduct *)e);
      } else {
         elems.add(*e);
      }
   }
}

bool tryExportHistFactory(const std::string &pdfname, const std::string chname, const RooRealSumPdf *sumpdf,
                          JSONNode &elem)
{
   if (!sumpdf)
      return false;
   for (const auto &sample : sumpdf->funcList()) {
      if (!sample->InheritsFrom(RooProduct::Class()) && !sample->InheritsFrom(RooRealSumPdf::Class()))
         return false;
   }

   bool has_poisson_constraints = false;
   bool has_gauss_constraints = false;
   std::map<int, double> tot_yield;
   std::map<int, double> tot_yield2;
   std::map<int, double> rel_errors;
   std::map<int, std::unique_ptr<TH1>> bb_histograms;
   std::map<int, std::unique_ptr<TH1>> nonbb_histograms;
   std::vector<std::string> varnames;

   for (size_t sampleidx = 0; sampleidx < sumpdf->funcList().size(); ++sampleidx) {
      const auto func = sumpdf->funcList().at(sampleidx);
      const auto coef = sumpdf->coefList().at(sampleidx);
      std::string samplename = func->GetName();
      if (startsWith(samplename, "L_x_"))
         samplename = samplename.substr(4);
      if (endsWith(samplename, "_shapes"))
         samplename = samplename.substr(0, samplename.size() - 7);
      if (endsWith(samplename, "_" + chname))
         samplename = samplename.substr(0, samplename.size() - chname.size() - 1);

      RooArgSet elems;
      if (func->InheritsFrom(RooProduct::Class())) {
         collectElements(elems, (RooProduct *)func);
      } else {
         elems.add(*func);
      }
      if (coef->InheritsFrom(RooProduct::Class())) {
         collectElements(elems, (RooProduct *)coef);
      } else {
         elems.add(*coef);
      }
      std::unique_ptr<TH1> hist;
      std::vector<ParamHistFunc *> phfs;
      PiecewiseInterpolation *pip = nullptr;
      std::vector<const RooAbsArg *> norms;

      RooStats::HistFactory::FlexibleInterpVar *fip = nullptr;
      for (const auto &e : elems) {
         if (e->InheritsFrom(RooConstVar::Class())) {
            if (((RooConstVar *)e)->getVal() == 1.)
               continue;
            norms.push_back(e);
         } else if (e->InheritsFrom(RooRealVar::Class())) {
            norms.push_back(e);
         } else if (e->InheritsFrom(RooHistFunc::Class())) {
            const RooHistFunc *hf = static_cast<const RooHistFunc *>(e);
            if (varnames.empty()) {
               varnames = getVarnames(hf);
            }
            if (!hist) {
               hist = histFunc2TH1(hf);
            }
         } else if (e->InheritsFrom(RooStats::HistFactory::FlexibleInterpVar::Class())) {
            fip = static_cast<RooStats::HistFactory::FlexibleInterpVar *>(e);
         } else if (e->InheritsFrom(PiecewiseInterpolation::Class())) {
            pip = static_cast<PiecewiseInterpolation *>(e);
         } else if (e->InheritsFrom(ParamHistFunc::Class())) {
            phfs.push_back((ParamHistFunc *)e);
         }
      }
      if (pip) {
         if (!hist && pip->nominalHist()->InheritsFrom(RooHistFunc::Class())) {
            hist = histFunc2TH1(static_cast<const RooHistFunc *>(pip->nominalHist()));
         }
         if (varnames.empty() && pip->nominalHist()->InheritsFrom(RooHistFunc::Class())) {
            varnames = getVarnames(dynamic_cast<const RooHistFunc *>(pip->nominalHist()));
         }
      }
      if (!hist) {
         return false;
      }

      elem["name"] << pdfname;
      elem["type"] << "histfactory_dist";

      auto &samples = elem["samples"];
      samples.set_seq();

      auto &s = samples.append_child();
      s.set_map();
      s["name"] << samplename;

      auto &modifiers = s["modifiers"];
      modifiers.set_seq();

      for (const auto &nf : norms) {
         auto &mod = modifiers.append_child();
         mod.set_map();
         mod["name"] << nf->GetName();
         mod["type"] << "normfactor";
      }

      if (pip) {
         for (size_t i = 0; i < pip->paramList().size(); ++i) {
            auto &mod = modifiers.append_child();
            mod.set_map();
            mod["type"] << "histosys";
            std::string sysname(pip->paramList().at(i)->GetName());
            if (sysname.find("alpha_") == 0) {
               sysname = sysname.substr(6);
            }
            mod["name"] << sysname;
            auto &data = mod["data"];
            data.set_map();
            auto &dataLow = data["lo"];
            if (pip->lowList().at(i)->InheritsFrom(RooHistFunc::Class())) {
               auto histLow = histFunc2TH1(static_cast<RooHistFunc *>(pip->lowList().at(i)));
               RooJSONFactoryWSTool::exportHistogram(*histLow, dataLow, varnames, 0, false, false);
            }
            auto &dataHigh = data["hi"];
            if (pip->highList().at(i)->InheritsFrom(RooHistFunc::Class())) {
               auto histHigh = histFunc2TH1(static_cast<RooHistFunc *>(pip->highList().at(i)));
               RooJSONFactoryWSTool::exportHistogram(*histHigh, dataHigh, varnames, 0, false, false);
            }
         }
      }

      if (fip) {
         for (size_t i = 0; i < fip->variables().size(); ++i) {
            auto &mod = modifiers.append_child();
            mod.set_map();
            mod["type"] << "normsys";
            std::string sysname(fip->variables().at(i)->GetName());
            if (sysname.find("alpha_") == 0) {
               sysname = sysname.substr(6);
            }
            mod["name"] << sysname;
            auto &data = mod["data"];
            data.set_map();
            data["lo"] << fip->low()[i];
            data["hi"] << fip->high()[i];
         }
      }
      bool has_mc_stat = false;
      for (auto phf : phfs) {
         if (TString(phf->GetName()).BeginsWith("mc_stat_")) { // MC stat uncertainty
            has_mc_stat = true;
            s["statError"] << 1;
            int idx = 0;
            for (const auto &g : phf->paramList()) {
               ++idx;
               RooPoisson *constraint_p = findClient<RooPoisson>(g);
               RooGaussian *constraint_g = findClient<RooGaussian>(g);
               if (tot_yield.find(idx) == tot_yield.end()) {
                  tot_yield[idx] = 0;
                  tot_yield2[idx] = 0;
               }
               tot_yield[idx] += hist->GetBinContent(idx);
               tot_yield2[idx] += (hist->GetBinContent(idx) * hist->GetBinContent(idx));
               if (constraint_p) {
                  double erel = 1. / std::sqrt(constraint_p->getX().getVal());
                  rel_errors[idx] = erel;
                  has_poisson_constraints = true;
               } else if (constraint_g) {
                  double erel = constraint_g->getSigma().getVal() / constraint_g->getMean().getVal();
                  rel_errors[idx] = erel;
                  has_gauss_constraints = true;
               }
            }
            bb_histograms[sampleidx] = std::move(hist);
         } else { // other ShapeSys
            auto &mod = modifiers.append_child();
            mod.set_map();

            // Getting the name of the syst is tricky.
            TString sysName(phf->GetName());
            sysName.Remove(sysName.Index("_ShapeSys"));
            sysName.Remove(0, chname.size() + 1);
            mod["name"] << sysName.Data();
            mod["type"] << "shapesys";
            auto &data = mod["data"];
            data.set_map();
            auto &vals = data["vals"];
            bool is_poisson = false;
            for (const auto &g : phf->paramList()) {
               RooPoisson *constraint_p = findClient<RooPoisson>(g);
               RooGaussian *constraint_g = findClient<RooGaussian>(g);
               if (constraint_p) {
                  is_poisson = true;
                  vals.append_child() << constraint_p->getX().getVal();
               } else if (constraint_g) {
                  is_poisson = false;
                  vals.append_child() << constraint_g->getSigma().getVal() / constraint_g->getMean().getVal();
               }
            }
            if (is_poisson) {
               mod["constraint"] << "Poisson";
            } else {
               mod["constraint"] << "Gauss";
            }
         }
      }
      if (!has_mc_stat) {
         nonbb_histograms[sampleidx] = std::move(hist);
         s["statError"] << 0;
      }
   }

   auto &samples = elem["samples"];
   for (const auto &hist : nonbb_histograms) {
      auto &s = samples[hist.first];
      auto &data = s["data"];
      RooJSONFactoryWSTool::writeObservables(*hist.second, elem, varnames);
      RooJSONFactoryWSTool::exportHistogram(*hist.second, data, varnames, 0, false, false);
   }
   for (const auto &hist : bb_histograms) {
      auto &s = samples[hist.first];
      for (auto bin : rel_errors) {
         // reverse engineering the correct partial error
         // the (arbitrary) convention used here is that all samples should have the same relative error
         const int i = bin.first;
         const double relerr_tot = bin.second;
         const double count = hist.second->GetBinContent(i);
         hist.second->SetBinError(i, relerr_tot * tot_yield[i] / std::sqrt(tot_yield2[i]) * count);
      }
      auto &data = s["data"];
      RooJSONFactoryWSTool::writeObservables(*hist.second, elem, varnames);
      RooJSONFactoryWSTool::exportHistogram(*hist.second, data, varnames, 0, false, true);
   }
   auto &statError = elem["statError"];
   statError.set_map();
   if (has_poisson_constraints) {
      statError["constraint"] << "Poisson";
   } else if (has_gauss_constraints) {
      statError["constraint"] << "Gauss";
   }
   return true;
}

class HistFactoryStreamer_ProdPdf : public RooFit::JSONIO::Exporter {
public:
   bool autoExportDependants() const override { return false; }
   bool tryExport(const RooProdPdf *prodpdf, JSONNode &elem) const
   {
      RooRealSumPdf *sumpdf = nullptr;
      for (const auto &v : prodpdf->pdfList()) {
         if (v->InheritsFrom(RooRealSumPdf::Class())) {
            sumpdf = static_cast<RooRealSumPdf *>(v);
         }
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
      const RooProdPdf *prodpdf = static_cast<const RooProdPdf *>(p);
      if (tryExport(prodpdf, elem)) {
         return true;
      }
      return false;
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
      const RooRealSumPdf *sumpdf = static_cast<const RooRealSumPdf *>(p);
      if (tryExport(sumpdf, elem)) {
         return true;
      }
      return false;
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
