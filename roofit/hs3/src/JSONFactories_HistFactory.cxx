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

#include <RooStats/HistFactory/Detail/HistFactoryImpl.h>
#include <RooStats/HistFactory/ParamHistFunc.h>
#include <RooStats/HistFactory/PiecewiseInterpolation.h>
#include <RooStats/HistFactory/FlexibleInterpVar.h>
#include <RooConstVar.h>
#include <RooRealVar.h>
#include <RooDataHist.h>
#include <RooHistFunc.h>
#include <RooRealSumPdf.h>
#include <RooBinWidthFunction.h>
#include <RooProdPdf.h>
#include <RooPoisson.h>
#include <RooLognormal.h>
#include <RooGaussian.h>
#include <RooBinning.h>
#include <RooProduct.h>
#include <RooWorkspace.h>

#include "static_execute.h"
#include "JSONIOUtils.h"

using RooFit::Detail::JSONNode;

namespace {

inline void writeAxis(JSONNode &axis, RooRealVar const &obs)
{
   auto &binning = obs.getBinning();
   if (binning.isUniform()) {
      axis["nbins"] << obs.numBins();
      axis["min"] << obs.getMin();
      axis["max"] << obs.getMax();
   } else {
      auto &bounds = axis["bounds"];
      bounds.set_seq();
      double val = binning.binLow(0);
      bounds.append_child() << val;
      for (int i = 0; i < binning.numBins(); ++i) {
         val = binning.binHigh(i);
         bounds.append_child() << val;
      }
   }
}

double round_prec(double d, int nSig)
{
   if (d == 0.0)
      return 0.0;
   int ndigits = std::floor(std::log10(std::abs(d))) + 1 - nSig;
   double sf = std::pow(10, ndigits);
   if (std::abs(d / sf) < 2)
      ndigits--;
   return sf * std::round(d / sf);
}

// To avoid repeating the same string literals that can potentially get out of
// sync.
namespace Literals {
constexpr auto staterror = "staterror";
}

void erasePrefix(std::string &str, std::string_view prefix)
{
   if (startsWith(str, prefix)) {
      str.erase(0, prefix.size());
   }
}

void eraseSuffix(std::string &str, std::string_view suffix)
{
   if (endsWith(str, suffix)) {
      str.erase(str.size() - suffix.size());
   }
}

template <class Coll>
void sortByName(Coll &coll)
{
   std::sort(coll.begin(), coll.end(), [](auto &l, auto &r) { return l.name < r.name; });
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

RooAbsPdf *findConstraint(RooAbsArg *g)
{
   RooPoisson *constraint_p = findClient<RooPoisson>(g);
   if (constraint_p)
      return constraint_p;
   RooGaussian *constraint_g = findClient<RooGaussian>(g);
   if (constraint_g)
      return constraint_g;
   RooLognormal *constraint_l = findClient<RooLognormal>(g);
   if (constraint_l)
      return constraint_l;
   return nullptr;
}

std::string toString(TClass *c)
{
   if (!c) {
      return "Const";
   }
   if (c == RooPoisson::Class()) {
      return "Poisson";
   }
   if (c == RooGaussian::Class()) {
      return "Gauss";
   }
   if (c == RooLognormal::Class()) {
      return "Lognormal";
   }
   return "unknown";
}

using namespace RooStats::HistFactory::Detail;

RooRealVar &createNominal(RooWorkspace &ws, std::string const &parname, double val, double min, double max)
{
   RooRealVar &nom = getOrCreate<RooRealVar>(ws, "nom_" + parname, val, min, max);
   nom.setConstant(true);
   nom.setAttribute("globs");
   return nom;
}

/// Get the conventional name of the constraint pdf for a constrained
/// parameter.
std::string constraintName(std::string const &sysname)
{
   return sysname + "_constraint";
}

RooAbsPdf &getConstraint(RooWorkspace &ws, const std::string &sysname, const std::string &pname)
{
   return getOrCreate<RooGaussian>(ws, constraintName(sysname), *ws.var(pname), *ws.var("nom_" + pname),
                                   RooFit::RooConst(1.));
}

void setMCStatGammaRanges(RooArgList const &gammas, std::vector<double> const &errs, double statErrThresh)
{
   // Set a reasonable range for gamma and set constant NPs which are below the
   // MC stat threshold, and remove them from the np list.
   for (size_t i = 0; i < errs.size(); ++i) {
      auto g = static_cast<RooRealVar *>(gammas.at(i));
      g->setMax(1 + 5 * errs[i]);
      g->setError(errs[i]);
      if (errs[i] < statErrThresh) {
         g->setConstant(true); // all negative errs are set constant
      }
   }
}

ParamHistFunc &createPHF(const std::string &sysname, const std::string &phfname, const std::vector<double> &vals,
                         RooJSONFactoryWSTool &tool, RooArgList &constraints, const RooArgSet &observables,
                         const std::string &constraintType, RooArgList &gammas, double gamma_min, double gamma_max)
{
   RooWorkspace &ws = *tool.workspace();

   gammas.add(ParamHistFunc::createParamSet(ws, ("gamma_" + sysname).c_str(), observables, gamma_min, gamma_max));
   auto &phf = tool.wsEmplace<ParamHistFunc>(phfname, observables, gammas);
   for (size_t i = 0; i < gammas.size(); ++i) {
      RooRealVar *v = dynamic_cast<RooRealVar *>(&gammas[i]);
      if (!v)
         continue;
      std::string basename = v->GetName();
      v->setConstant(false);
      if (constraintType == "Const" || vals[i] == 0.) {
         v->setConstant(true);
      } else if (constraintType == "Gauss") {
         auto &nom = createNominal(ws, basename, 1.0, 0, std::max(10., gamma_max));
         auto &sigma = tool.wsEmplace<RooConstVar>(basename + "_sigma", vals[i]);
         constraints.add(tool.wsEmplace<RooGaussian>(constraintName(basename), nom, *v, sigma), true);
      } else if (constraintType == "Poisson") {
         double tau_float = vals[i];
         auto &tau = tool.wsEmplace<RooConstVar>(basename + "_tau", tau_float);
         auto &nom = createNominal(ws, basename, tau_float, 0, RooNumber::infinity());
         auto &prod = tool.wsEmplace<RooProduct>(basename + "_poisMean", *v, tau);
         auto &pois = tool.wsEmplace<RooPoisson>(constraintName(basename), nom, prod);
         pois.setNoRounding(true);
         constraints.add(pois, true);
      } else {
         RooJSONFactoryWSTool::error("unknown constraint type " + constraintType);
      }
   }
   for (auto &g : gammas) {
      for (auto client : g->clients()) {
         if (dynamic_cast<RooAbsPdf *>(client) && !constraints.find(*client)) {
            constraints.add(*client);
         }
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

bool importHistSample(RooJSONFactoryWSTool &tool, RooDataHist &dh, RooArgSet const &varlist,
                      RooAbsArg const *mcStatObject, const std::string &fprefix, const JSONNode &p,
                      RooArgList &constraints)
{
   RooWorkspace &ws = *tool.workspace();

   std::string name = RooJSONFactoryWSTool::name(p);
   std::string prefixedName = fprefix + "_" + name;

   if (!p.has_child("data")) {
      RooJSONFactoryWSTool::error("sample '" + name + "' does not define a 'data' key");
   }

   auto &hf = tool.wsEmplace<RooHistFunc>("hist_" + prefixedName, varlist, dh);
   hf.SetTitle(RooJSONFactoryWSTool::name(p).c_str());

   RooArgList shapeElems;
   RooArgList normElems;

   shapeElems.add(tool.wsEmplace<RooBinWidthFunction>(prefixedName + "_binWidth", hf, true));

   if (hasStaterror(p)) {
      shapeElems.add(*mcStatObject);
   }

   if (p.has_child("modifiers")) {
      RooArgList overall_nps;
      std::vector<double> overall_low;
      std::vector<double> overall_high;

      RooArgList histNps;
      RooArgList histoLo;
      RooArgList histoHi;

      for (const auto &mod : p["modifiers"].children()) {
         std::string const &modtype = mod["type"].val();
         std::string const &sysname = RooJSONFactoryWSTool::name(mod);
         if (modtype == "staterror") {
            // this is dealt with at a different place, ignore it for now
         } else if (modtype == "normfactor") {
            normElems.add(getOrCreate<RooRealVar>(ws, sysname, 1., -3, 5));
            if (auto constrInfo = mod.find("constraint_name")) {
               constraints.add(*tool.request<RooAbsReal>(constrInfo->val(), name));
            }
         } else if (modtype == "normsys") {
            auto *parameter = mod.find("parameter");
            std::string parname(parameter ? parameter->val() : "alpha_" + sysname);
            createNominal(ws, parname, 0.0, -10, 10);
            overall_nps.add(getOrCreate<RooRealVar>(ws, parname, 0., -5, 5));
            auto &data = mod["data"];
            // the below contains a a hack to cut off variations that go below 0
            // this is needed because with interpolation code 4, which is the default, interpolation is done in
            // log-space. hence, values <= 0 result in NaN which propagate throughout the model and cause evaluations to
            // fail if you know a nicer way to solve this, please go ahead and fix the lines below
            overall_low.push_back(data["lo"].val_double() > 0 ? data["lo"].val_double()
                                                              : std::numeric_limits<double>::epsilon());
            overall_high.push_back(data["hi"].val_double() > 0 ? data["hi"].val_double()
                                                               : std::numeric_limits<double>::epsilon());
            constraints.add(getConstraint(ws, sysname, parname));
         } else if (modtype == "histosys") {
            auto *parameter = mod.find("parameter");
            std::string parname(parameter ? parameter->val() : "alpha_" + sysname);
            createNominal(ws, parname, 0.0, -10, 10);
            histNps.add(getOrCreate<RooRealVar>(ws, parname, 0., -5, 5));
            auto &data = mod["data"];
            histoLo.add(tool.wsEmplace<RooHistFunc>(
               sysname + "Low_" + prefixedName, varlist,
               RooJSONFactoryWSTool::readBinnedData(data["lo"], sysname + "Low_" + prefixedName, varlist)));
            histoHi.add(tool.wsEmplace<RooHistFunc>(
               sysname + "High_" + prefixedName, varlist,
               RooJSONFactoryWSTool::readBinnedData(data["hi"], sysname + "High_" + prefixedName, varlist)));
            constraints.add(getConstraint(ws, sysname, parname));
         } else if (modtype == "shapesys") {
            std::string funcName = prefixedName + "_" + sysname + "_" + prefixedName + "_ShapeSys";
            std::vector<double> vals;
            for (const auto &v : mod["data"]["vals"].children()) {
               vals.push_back(v.val_double());
            }
            RooArgList gammas;
            std::string constraint(mod["constraint"].val());
            shapeElems.add(createPHF(sysname, funcName, vals, tool, constraints, varlist, constraint, gammas, 0, 1000));
         } else if (modtype == "custom") {
            RooAbsReal *obj = ws.function(sysname);
            if (!obj) {
               RooJSONFactoryWSTool::error("unable to find custom modifier '" + sysname + "'");
            }
            if (obj->dependsOn(varlist)) {
               shapeElems.add(*obj);
            } else {
               normElems.add(*obj);
            }
         } else {
            RooJSONFactoryWSTool::error("modifier '" + sysname + "' of unknown type '" + modtype + "'");
         }
      }

      if (!overall_nps.empty()) {
         auto &v = tool.wsEmplace<RooStats::HistFactory::FlexibleInterpVar>("overallSys_" + prefixedName, overall_nps,
                                                                            1., overall_low, overall_high);
         v.setAllInterpCodes(4); // default HistFactory interpCode
         normElems.add(v);
      }
      if (!histNps.empty()) {
         auto &v = tool.wsEmplace<PiecewiseInterpolation>("histoSys_" + prefixedName, hf, histoLo, histoHi, histNps);
         v.setPositiveDefinite();
         v.setAllInterpCodes(4); // default interpCode for HistFactory
         shapeElems.add(v);
      } else {
         shapeElems.add(hf);
      }
   }

   tool.wsEmplace<RooProduct>(prefixedName + "_shapes", shapeElems);
   if (!normElems.empty()) {
      tool.wsEmplace<RooProduct>(prefixedName + "_scaleFactors", normElems);
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

      std::string name = RooJSONFactoryWSTool::name(p);
      if (!p.has_child("samples")) {
         RooJSONFactoryWSTool::error("no samples in '" + name + "', skipping.");
      }
      double statErrThresh = 0;
      std::string statErrType = "Poisson";
      if (p.has_child(::Literals::staterror)) {
         auto &staterr = p[::Literals::staterror];
         if (staterr.has_child("relThreshold"))
            statErrThresh = staterr["relThreshold"].val_double();
         if (staterr.has_child("constraint"))
            statErrType = staterr["constraint"].val();
      }
      std::vector<double> sumW;
      std::vector<double> sumW2;
      RooArgSet observables;
      for (auto const &obsNode : p["axes"].children()) {
         if (obsNode.has_child("bounds")) {
            std::vector<double> bounds;
            for (auto const &bound : obsNode["bounds"].children()) {
               bounds.push_back(bound.val_double());
            }
            RooRealVar &obs = getOrCreate<RooRealVar>(ws, obsNode["name"].val(), bounds[0], bounds[bounds.size() - 1]);
            RooBinning bins(obs.getMin(), obs.getMax());
            ;
            for (auto b : bounds) {
               bins.addBoundary(b);
            }
            obs.setBinning(bins);
            observables.add(obs);
         } else {
            RooRealVar &obs = getOrCreate<RooRealVar>(ws, obsNode["name"].val(), obsNode["min"].val_double(),
                                                      obsNode["max"].val_double());
            obs.setBins(obsNode["nbins"].val_int());
            observables.add(obs);
         }
      }

      std::string fprefix = name;

      std::vector<std::unique_ptr<RooDataHist>> data;
      for (const auto &comp : p["samples"].children()) {
         std::unique_ptr<RooDataHist> dh = RooJSONFactoryWSTool::readBinnedData(
            comp["data"], fprefix + "_" + RooJSONFactoryWSTool::name(comp) + "_dataHist", observables);
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

      RooAbsArg *mcStatObject = nullptr;
      RooArgList constraints;
      if (!sumW.empty()) {
         std::string phfName = name;
         erasePrefix(phfName, "model_");

         std::vector<double> vals(sumW.size());
         std::vector<double> errs(sumW.size());

         for (size_t i = 0; i < sumW.size(); ++i) {
            errs[i] = std::sqrt(sumW2[i]) / sumW[i];
            if (statErrType == "Gauss") {
               vals[i] = std::max(errs[i], 0.); // avoid negative sigma. This NP will be set constant anyway later
            } else if (statErrType == "Poisson") {
               vals[i] = sumW[i] * sumW[i] / sumW2[i];
            }
         }

         RooArgList gammas;
         mcStatObject = &createPHF("stat_" + phfName, "mc_stat_" + phfName, vals, *tool, constraints, observables,
                                   statErrType, gammas, 0, 10);
         setMCStatGammaRanges(gammas, errs, statErrThresh);
      }

      int idx = 0;
      RooArgList funcs;
      RooArgList coefs;
      for (const auto &comp : p["samples"].children()) {
         importHistSample(*tool, *data[idx], observables, mcStatObject, fprefix, comp, constraints);
         ++idx;

         std::string const &compName = RooJSONFactoryWSTool::name(comp);
         funcs.add(*tool->request<RooAbsReal>(fprefix + "_" + compName + "_shapes", name));
         coefs.add(*tool->request<RooAbsReal>(fprefix + "_" + compName + "_scaleFactors", name));
      }

      if (constraints.empty()) {
         auto &sum = tool->wsEmplace<RooRealSumPdf>(name, funcs, coefs, true);
         sum.setAttribute("BinnedLikelihood");
      } else {
         std::string sumName = name + "_model";
         erasePrefix(sumName, "model_");
         auto &sum = tool->wsEmplace<RooRealSumPdf>(sumName, funcs, coefs, true);
         sum.SetTitle(name.c_str());
         sum.setAttribute("BinnedLikelihood");
         tool->wsEmplace<RooProdPdf>(name, constraints, RooFit::Conditional(sum, observables));
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
      RooJSONFactoryWSTool::fillSeq(elem["vars"], fip->variables());
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
      RooJSONFactoryWSTool::fillSeq(elem["vars"], pip->paramList());
      elem["nom"] << pip->nominalHist()->GetName();
      RooJSONFactoryWSTool::fillSeq(elem["high"], pip->highList());
      RooJSONFactoryWSTool::fillSeq(elem["low"], pip->lowList());
      return true;
   }
};

class PiecewiseInterpolationFactory : public RooFit::JSONIO::Importer {
public:
   bool importFunction(RooJSONFactoryWSTool *tool, const JSONNode &p) const override
   {
      std::string name(RooJSONFactoryWSTool::name(p));

      RooArgList vars{tool->requestArgList<RooRealVar>(p, "vars")};

      auto &pip = tool->wsEmplace<PiecewiseInterpolation>(name, *tool->requestArg<RooAbsReal>(p, "nom"),
                                                          tool->requestArgList<RooAbsReal>(p, "low"),
                                                          tool->requestArgList<RooAbsReal>(p, "high"), vars);

      pip.setPositiveDefinite(p["positiveDefinite"].val_bool());

      if (p.has_child("interpolationCodes")) {
         std::size_t i = 0;
         for (auto const &node : p["interpolationCodes"].children()) {
            pip.setInterpCode(*static_cast<RooAbsReal *>(vars.at(i)), node.val_int(), true);
            ++i;
         }
      }

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

      auto &fip = tool->wsEmplace<RooStats::HistFactory::FlexibleInterpVar>(name, vars, nom, low, high);

      if (p.has_child("interpolationCodes")) {
         size_t i = 0;
         for (auto const &node : p["interpolationCodes"].children()) {
            fip.setInterpCode(*static_cast<RooAbsReal *>(vars.at(i)), node.val_int());
            ++i;
         }
      }

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

struct NormFactor {
   std::string name;
   RooAbsArg const *param = nullptr;
   RooAbsPdf const *constraint = nullptr;
   NormFactor(RooAbsArg const &par, RooAbsPdf const *constr = nullptr)
      : name{par.GetName()}, param{&par}, constraint{constr}
   {
   }
};

struct NormSys {
   std::string name;
   RooAbsArg const *param = nullptr;
   double low;
   double high;
   TClass *constraint = RooGaussian::Class();
   NormSys(const std::string &n, RooAbsArg *const p, double h, double l, TClass *c)
      : name(n), param(p), low(l), high(h), constraint(c)
   {
   }
};
struct HistoSys {
   std::string name;
   RooAbsArg const *param = nullptr;
   std::vector<double> low;
   std::vector<double> high;
   TClass *constraint = RooGaussian::Class();
   HistoSys(const std::string &n, RooAbsArg *const p, RooHistFunc *l, RooHistFunc *h, TClass *c)
      : name(n), param(p), constraint(c)
   {
      low.assign(l->dataHist().weightArray(), l->dataHist().weightArray() + l->dataHist().numEntries());
      high.assign(h->dataHist().weightArray(), h->dataHist().weightArray() + h->dataHist().numEntries());
   }
};
struct ShapeSys {
   std::string name;
   std::vector<double> constraints;
   TClass *constraint = nullptr;
   ShapeSys(const std::string &n) : name{n} {}
};
struct Sample {
   std::string name;
   std::vector<double> hist;
   std::vector<double> histError;
   std::vector<NormFactor> normfactors;
   std::vector<NormSys> normsys;
   std::vector<HistoSys> histosys;
   std::vector<ShapeSys> shapesys;
   std::vector<RooAbsReal *> otherElements;
   bool useBarlowBeestonLight = false;
   TClass *barlowBeestonLightConstraint = RooPoisson::Class();
   Sample(const std::string &n) : name{n} {}
};

void addNormFactor(RooRealVar const *par, Sample &sample, RooWorkspace *ws)
{
   std::string parname = par->GetName();
   bool isConstrained = false;
   for (RooAbsArg const *pdf : ws->allPdfs()) {
      if (auto gauss = dynamic_cast<RooGaussian const *>(pdf)) {
         if (parname == gauss->getX().GetName()) {
            sample.normfactors.emplace_back(*par, gauss);
            isConstrained = true;
         }
      }
   }
   if (!isConstrained)
      sample.normfactors.emplace_back(*par);
}

bool tryExportHistFactory(RooJSONFactoryWSTool *tool, const std::string &pdfname, const RooRealSumPdf *sumpdf,
                          JSONNode &elem)
{
   RooWorkspace *ws = tool->workspace();
   RooArgSet customModifiers;

   if (!sumpdf)
      return false;

   std::string chname = pdfname;
   erasePrefix(chname, "model_");
   eraseSuffix(chname, "_model");

   for (RooAbsArg *sample : sumpdf->funcList()) {
      if (!dynamic_cast<RooProduct *>(sample) && !dynamic_cast<RooRealSumPdf *>(sample)) {
         return false;
      }
   }

   std::map<int, double> tot_yield;
   std::map<int, double> tot_yield2;
   std::map<int, double> rel_errors;
   RooArgSet const *varSet = nullptr;
   long unsigned int nBins = 0;

   std::vector<Sample> samples;

   for (size_t sampleidx = 0; sampleidx < sumpdf->funcList().size(); ++sampleidx) {
      PiecewiseInterpolation *pip = nullptr;
      RooStats::HistFactory::FlexibleInterpVar *fip = nullptr;
      std::vector<ParamHistFunc *> phfs;

      const auto func = sumpdf->funcList().at(sampleidx);
      Sample sample(func->GetName());
      erasePrefix(sample.name, "L_x_");
      eraseSuffix(sample.name, "_shapes");
      eraseSuffix(sample.name, "_" + chname);
      erasePrefix(sample.name, pdfname + "_");
      RooArgSet elems;
      collectElements(elems, func);
      collectElements(elems, sumpdf->coefList().at(sampleidx));

      auto updateObservables = [&](RooDataHist const &dataHist) {
         if (varSet == nullptr) {
            varSet = dataHist.get();
            nBins = dataHist.numEntries();
         }
         if (sample.hist.empty()) {
            auto *w = dataHist.weightArray();
            sample.hist.assign(w, w + dataHist.numEntries());
         }
      };

      for (RooAbsArg *e : elems) {
         if (auto constVar = dynamic_cast<RooConstVar *>(e)) {
            if (constVar->getVal() != 1.) {
               sample.normfactors.emplace_back(*e);
            }
         } else if (auto par = dynamic_cast<RooRealVar *>(e)) {
            addNormFactor(par, sample, ws);
         } else if (auto hf = dynamic_cast<const RooHistFunc *>(e)) {
            updateObservables(hf->dataHist());
         } else if (auto phf = dynamic_cast<ParamHistFunc *>(e)) {
            phfs.push_back(phf);
         } else if (!fip && (fip = dynamic_cast<RooStats::HistFactory::FlexibleInterpVar *>(e))) {
         } else if (!pip && (pip = dynamic_cast<PiecewiseInterpolation *>(e))) {
         } else if (auto real = dynamic_cast<RooAbsReal *>(e)) {
            if (!dynamic_cast<RooBinWidthFunction *>(real)) {
               sample.otherElements.push_back(real);
            }
         }
      }

      // see if we can get the observables
      if (pip) {
         if (auto nh = dynamic_cast<RooHistFunc const *>(pip->nominalHist())) {
            updateObservables(nh->dataHist());
         }
      }

      // sort and configure norms
      sortByName(sample.normfactors);

      // sort and configure the normsys
      if (fip) {
         for (size_t i = 0; i < fip->variables().size(); ++i) {
            RooAbsArg *var = fip->variables().at(i);
            std::string sysname(var->GetName());
            erasePrefix(sysname, "alpha_");
            sample.normsys.emplace_back(sysname, var, fip->high()[i], fip->low()[i], findConstraint(var)->IsA());
         }
         sortByName(sample.normsys);
      }

      // sort and configure the histosys
      if (pip) {
         for (size_t i = 0; i < pip->paramList().size(); ++i) {
            RooAbsArg *var = pip->paramList().at(i);
            std::string sysname(var->GetName());
            erasePrefix(sysname, "alpha_");
            if (auto lo = dynamic_cast<RooHistFunc *>(pip->lowList().at(i))) {
               if (auto hi = dynamic_cast<RooHistFunc *>(pip->highList().at(i))) {
                  sample.histosys.emplace_back(sysname, var, lo, hi, findConstraint(var)->IsA());
               }
            }
         }
         sortByName(sample.histosys);
      }

      for (ParamHistFunc *phf : phfs) {
         if (startsWith(std::string(phf->GetName()), "mc_stat_")) { // MC stat uncertainty
            int idx = 0;
            for (const auto &g : phf->paramList()) {
               ++idx;
               RooAbsPdf *constraint = findConstraint(g);
               if (tot_yield.find(idx) == tot_yield.end()) {
                  tot_yield[idx] = 0;
                  tot_yield2[idx] = 0;
               }
               tot_yield[idx] += sample.hist[idx - 1];
               tot_yield2[idx] += (sample.hist[idx - 1] * sample.hist[idx - 1]);
               if (constraint) {
                  sample.barlowBeestonLightConstraint = constraint->IsA();
                  if (RooPoisson *constraint_p = dynamic_cast<RooPoisson *>(constraint)) {
                     double erel = 1. / std::sqrt(constraint_p->getX().getVal());
                     rel_errors[idx] = erel;
                  } else if (RooGaussian *constraint_g = dynamic_cast<RooGaussian *>(constraint)) {
                     double erel = constraint_g->getSigma().getVal() / constraint_g->getMean().getVal();
                     rel_errors[idx] = erel;
                  } else {
                     RooJSONFactoryWSTool::error(
                        "currently, only RooPoisson and RooGaussian are supported as constraint types");
                  }
               }
            }
            sample.useBarlowBeestonLight = true;
         } else { // other ShapeSys
            ShapeSys sys(phf->GetName());
            erasePrefix(sys.name, "model_" + chname + "_");
            erasePrefix(sys.name, chname + "_");
            erasePrefix(sys.name, sample.name + "_");
            eraseSuffix(sys.name, "_ShapeSys");
            eraseSuffix(sys.name, "_" + sample.name);
            eraseSuffix(sys.name, "_model_" + chname);
            eraseSuffix(sys.name, "_" + chname);
            eraseSuffix(sys.name, "_" + sample.name);

            for (const auto &g : phf->paramList()) {
               RooAbsPdf *constraint = findConstraint(g);
               if (!constraint)
                  constraint = ws->pdf(constraintName(g->GetName()));
               if (!constraint && !g->isConstant())
                  RooJSONFactoryWSTool::error("cannot find constraint for " + std::string(g->GetName()));
               else if (!constraint) {
                  sys.constraints.push_back(0.0);
               } else if (auto constraint_p = dynamic_cast<RooPoisson *>(constraint)) {
                  sys.constraints.push_back(constraint_p->getX().getVal());
                  if (!sys.constraint) {
                     sys.constraint = RooPoisson::Class();
                  }
               } else if (auto constraint_g = dynamic_cast<RooGaussian *>(constraint)) {
                  sys.constraints.push_back(constraint_g->getSigma().getVal() / constraint_g->getMean().getVal());
                  if (!sys.constraint) {
                     sys.constraint = RooGaussian::Class();
                  }
               }
            }
            sample.shapesys.emplace_back(std::move(sys));
         }
      }
      sortByName(sample.shapesys);

      // add the sample
      samples.emplace_back(std::move(sample));
   }

   sortByName(samples);

   for (auto &sample : samples) {
      if (sample.hist.empty()) {
         return false;
      }
      if (sample.useBarlowBeestonLight) {
         sample.histError.resize(sample.hist.size());
         for (auto bin : rel_errors) {
            // reverse engineering the correct partial error
            // the (arbitrary) convention used here is that all samples should have the same relative error
            const int i = bin.first;
            const double relerr_tot = bin.second;
            const double count = sample.hist[i - 1];
            // this reconstruction is inherently unprecise, so we truncate it at some decimal places to make sure that
            // we don't carry around too many useless digits
            sample.histError[i - 1] = round_prec(relerr_tot * tot_yield[i] / std::sqrt(tot_yield2[i]) * count, 7);
         }
      }
   }

   bool observablesWritten = false;
   for (const auto &sample : samples) {

      elem["type"] << "histfactory_dist";

      auto &s = RooJSONFactoryWSTool::appendNamedChild(elem["samples"], sample.name);

      auto &modifiers = s["modifiers"];

      for (const auto &nf : sample.normfactors) {
         auto &mod = RooJSONFactoryWSTool::appendNamedChild(modifiers, nf.name);
         mod["parameter"] << nf.param->GetName();
         mod["type"] << "normfactor";
         if (nf.constraint) {
            mod["constraint_name"] << nf.constraint->GetName();
            tool->queueExport(*nf.constraint);
         }
      }

      for (const auto &sys : sample.normsys) {
         auto &mod = RooJSONFactoryWSTool::appendNamedChild(modifiers, sys.name);
         mod["type"] << "normsys";
         mod["parameter"] << sys.param->GetName();
         mod["constraint"] << toString(sys.constraint);
         auto &data = mod["data"].set_map();
         data["lo"] << sys.low;
         data["hi"] << sys.high;
      }

      for (const auto &sys : sample.histosys) {
         auto &mod = RooJSONFactoryWSTool::appendNamedChild(modifiers, sys.name);
         mod["type"] << "histosys";
         mod["parameter"] << sys.param->GetName();
         mod["constraint"] << toString(sys.constraint);
         auto &data = mod["data"].set_map();
         if (nBins != sys.low.size() || nBins != sys.high.size()) {
            std::stringstream ss;
            ss << "inconsistent binning: " << nBins << " bins expected, but " << sys.low.size() << "/"
               << sys.high.size() << " found in nominal histogram errors!";
            RooJSONFactoryWSTool::error(ss.str().c_str());
         }
         RooJSONFactoryWSTool::exportArray(nBins, sys.low.data(), data["lo"].set_map()["contents"]);
         RooJSONFactoryWSTool::exportArray(nBins, sys.high.data(), data["hi"].set_map()["contents"]);
      }

      for (const auto &sys : sample.shapesys) {
         auto &mod = RooJSONFactoryWSTool::appendNamedChild(modifiers, sys.name);
         mod["type"] << "shapesys";
         mod["constraint"] << toString(sys.constraint);
         if (sys.constraint) {
            auto &data = mod["data"].set_map();
            auto &vals = data["vals"];
            vals.fill_seq(sys.constraints);
         }
      }

      for (const auto &other : sample.otherElements) {
         auto &mod = RooJSONFactoryWSTool::appendNamedChild(modifiers, other->GetName());
         customModifiers.add(*other);
         mod["type"] << "custom";
      }

      if (sample.useBarlowBeestonLight) {
         auto &mod = RooJSONFactoryWSTool::appendNamedChild(modifiers, ::Literals::staterror);
         mod["type"] << ::Literals::staterror;
         mod["constraint"] << toString(sample.barlowBeestonLightConstraint);
      }

      if (!observablesWritten) {
         auto &output = elem["axes"].set_seq();
         for (auto *obs : static_range_cast<RooRealVar *>(*varSet)) {
            auto &out = output.append_child().set_map();
            out["name"] << obs->GetName();
            writeAxis(out, *obs);
         }
         observablesWritten = true;
      }
      auto &dataNode = s["data"].set_map();
      if (nBins != sample.hist.size()) {
         std::stringstream ss;
         ss << "inconsistent binning: " << nBins << " bins expected, but " << sample.hist.size()
            << " found in nominal histogram!";
         RooJSONFactoryWSTool::error(ss.str().c_str());
      }
      RooJSONFactoryWSTool::exportArray(nBins, sample.hist.data(), dataNode["contents"]);
      if (!sample.histError.empty()) {
         if (nBins != sample.histError.size()) {
            std::stringstream ss;
            ss << "inconsistent binning: " << nBins << " bins expected, but " << sample.histError.size()
               << " found in nominal histogram errors!";
            RooJSONFactoryWSTool::error(ss.str().c_str());
         }
         RooJSONFactoryWSTool::exportArray(nBins, sample.histError.data(), dataNode["errors"]);
      }
   }

   // Export all the custom modifiers
   for (RooAbsArg *modifier : customModifiers) {
      tool->queueExport(*modifier);
   }

   // Export all model parameters
   RooArgSet parameters;
   sumpdf->getParameters(varSet, parameters);
   for (RooAbsArg *param : parameters) {
      // This should exclude the global observables
      if (!startsWith(std::string{param->GetName()}, "nom_")) {
         tool->queueExport(*param);
      }
   }

   return true;
}

class HistFactoryStreamer_ProdPdf : public RooFit::JSONIO::Exporter {
public:
   bool autoExportDependants() const override { return false; }
   bool tryExport(RooJSONFactoryWSTool *tool, const RooProdPdf *prodpdf, JSONNode &elem) const
   {
      RooRealSumPdf *sumpdf = nullptr;
      for (RooAbsArg *v : prodpdf->pdfList()) {
         sumpdf = dynamic_cast<RooRealSumPdf *>(v);
      }
      return tryExportHistFactory(tool, prodpdf->GetName(), sumpdf, elem);
   }
   std::string const &key() const override
   {
      static const std::string keystring = "histfactory_dist";
      return keystring;
   }
   bool exportObject(RooJSONFactoryWSTool *tool, const RooAbsArg *p, JSONNode &elem) const override
   {
      return tryExport(tool, static_cast<const RooProdPdf *>(p), elem);
   }
};

class HistFactoryStreamer_SumPdf : public RooFit::JSONIO::Exporter {
public:
   bool autoExportDependants() const override { return false; }
   bool tryExport(RooJSONFactoryWSTool *tool, const RooRealSumPdf *sumpdf, JSONNode &elem) const
   {
      return tryExportHistFactory(tool, sumpdf->GetName(), sumpdf, elem);
   }
   std::string const &key() const override
   {
      static const std::string keystring = "histfactory_dist";
      return keystring;
   }
   bool exportObject(RooJSONFactoryWSTool *tool, const RooAbsArg *p, JSONNode &elem) const override
   {
      return tryExport(tool, static_cast<const RooRealSumPdf *>(p), elem);
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
