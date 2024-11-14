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
#include <RooProduct.h>
#include <RooWorkspace.h>

#include "static_execute.h"
#include "JSONIOUtils.h"

using RooFit::Detail::JSONNode;

using namespace RooStats::HistFactory;
using namespace RooStats::HistFactory::Detail;
using namespace RooStats::HistFactory::Detail::MagicConstants;

namespace {

inline void writeAxis(JSONNode &axis, RooRealVar const &obs)
{
   auto &binning = obs.getBinning();
   if (binning.isUniform()) {
      axis["nbins"] << obs.numBins();
      axis["min"] << obs.getMin();
      axis["max"] << obs.getMax();
   } else {
      auto &edges = axis["edges"];
      edges.set_seq();
      double val = binning.binLow(0);
      edges.append_child() << val;
      for (int i = 0; i < binning.numBins(); ++i) {
         val = binning.binHigh(i);
         edges.append_child() << val;
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

bool eraseSuffix(std::string &str, std::string_view suffix)
{
   if (endsWith(str, suffix)) {
      str.erase(str.size() - suffix.size());
      return true;
   } else {
      return false;
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

inline std::string defaultGammaName(std::string const &sysname, std::size_t i)
{
   return "gamma_" + sysname + "_bin_" + std::to_string(i);
}

/// Export the names of the gamma parameters to the modifier struct if the
/// names don't match the default gamma parameter names, which is gamma_<sysname>_bin_<i>
void optionallyExportGammaParameters(JSONNode &mod, std::string const &sysname,
                                     std::vector<std::string> const &paramNames)
{
   for (std::size_t i = 0; i < paramNames.size(); ++i) {
      if (paramNames[i] != defaultGammaName(sysname, i)) {
         mod["parameters"].fill_seq(paramNames);
         return;
      }
   }
}

RooRealVar &createNominal(RooWorkspace &ws, std::string const &parname, double val, double min, double max)
{
   RooRealVar &nom = getOrCreate<RooRealVar>(ws, "nom_" + parname, val, min, max);
   nom.setConstant(true);
   return nom;
}

/// Get the conventional name of the constraint pdf for a constrained
/// parameter.
std::string constraintName(std::string const &paramName)
{
   return paramName + "Constraint";
}

RooAbsPdf &getConstraint(RooWorkspace &ws, const std::string &pname)
{
   RooRealVar *constrParam = ws.var(pname);
   constrParam->setError(1.0);
   return getOrCreate<RooGaussian>(ws, constraintName(pname), *constrParam, *ws.var("nom_" + pname), 1.);
}

ParamHistFunc &createPHF(const std::string &phfname, std::string const &sysname,
                         const std::vector<std::string> &parnames, const std::vector<double> &vals,
                         RooJSONFactoryWSTool &tool, RooArgList &constraints, const RooArgSet &observables,
                         const std::string &constraintType, double gammaMin, double gammaMax, double minSigma)
{
   RooWorkspace &ws = *tool.workspace();

   RooArgList gammas;
   for (std::size_t i = 0; i < vals.size(); ++i) {
      const std::string name = parnames.empty() ? defaultGammaName(sysname, i) : parnames[i];
      gammas.add(getOrCreate<RooRealVar>(ws, name, 1., gammaMin, gammaMax));
   }

   auto &phf = tool.wsEmplace<ParamHistFunc>(phfname, observables, gammas);

   if (constraintType != "Const") {
      auto constraintsInfo = createGammaConstraints(
         gammas, vals, minSigma, constraintType == "Poisson" ? Constraint::Poisson : Constraint::Gaussian);
      for (auto const &term : constraintsInfo.constraints) {
         ws.import(*term, RooFit::RecycleConflictNodes());
         constraints.add(*ws.pdf(term->GetName()));
      }
   } else {
      for (auto *gamma : static_range_cast<RooRealVar *>(gammas)) {
         gamma->setConstant(true);
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

const JSONNode &findStaterror(const JSONNode &comp)
{
   if (comp.has_child("modifiers")) {
      for (const auto &mod : comp["modifiers"].children()) {
         if (mod["type"].val() == ::Literals::staterror)
            return mod;
      }
   }
   RooJSONFactoryWSTool::error("sample '" + RooJSONFactoryWSTool::name(comp) + "' does not have a " +
                               ::Literals::staterror + " modifier!");
}

bool importHistSample(RooJSONFactoryWSTool &tool, RooDataHist &dh, RooArgSet const &varlist,
                      RooAbsArg const *mcStatObject, const std::string &fprefix, const JSONNode &p,
                      RooArgList &constraints)
{
   RooWorkspace &ws = *tool.workspace();

   std::string sampleName = RooJSONFactoryWSTool::name(p);
   std::string prefixedName = fprefix + "_" + sampleName;

   std::string channelName = fprefix;
   erasePrefix(channelName, "model_");

   if (!p.has_child("data")) {
      RooJSONFactoryWSTool::error("sample '" + sampleName + "' does not define a 'data' key");
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

      int idx = 0;
      for (const auto &mod : p["modifiers"].children()) {
         std::string const &modtype = mod["type"].val();
         std::string const &sysname =
            mod.has_child("name")
               ? mod["name"].val()
               : (mod.has_child("parameter") ? mod["parameter"].val() : "syst_" + std::to_string(idx));
         ++idx;
         if (modtype == "staterror") {
            // this is dealt with at a different place, ignore it for now
         } else if (modtype == "normfactor") {
            RooRealVar &constrParam = getOrCreate<RooRealVar>(ws, sysname, 1., -3, 5);
            normElems.add(constrParam);
            if (auto constrInfo = mod.find("constraint_name")) {
               auto constraint = tool.request<RooAbsReal>(constrInfo->val(), sampleName);
               if (auto gauss = dynamic_cast<RooGaussian const *>(constraint)) {
                  constrParam.setError(gauss->getSigma().getVal());
               }
               constraints.add(*constraint);
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
            constraints.add(getConstraint(ws, parname));
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
            constraints.add(getConstraint(ws, parname));
         } else if (modtype == "shapesys") {
            std::string funcName = channelName + "_" + sysname + "_ShapeSys";
            // funcName should be "<channel_name>_<sysname>_ShapeSys"
            std::vector<double> vals;
            for (const auto &v : mod["data"]["vals"].children()) {
               vals.push_back(v.val_double());
            }
            std::vector<std::string> parnames;
            for (const auto &v : mod["parameters"].children()) {
               parnames.push_back(v.val());
            }
            if (vals.empty()) {
               RooJSONFactoryWSTool::error("unable to instantiate shapesys '" + sysname + "' with 0 values!");
            }
            std::string constraint(mod["constraint"].val());
            shapeElems.add(createPHF(funcName, sysname, parnames, vals, tool, constraints, varlist, constraint,
                                     defaultGammaMin, defaultShapeSysGammaMax, minShapeUncertainty));
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

      std::string interpName = sampleName + "_" + channelName + "_epsilon";
      if (!overall_nps.empty()) {
         auto &v = tool.wsEmplace<RooStats::HistFactory::FlexibleInterpVar>(interpName, overall_nps, 1., overall_low,
                                                                            overall_high);
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
   bool importArg(RooJSONFactoryWSTool *tool, const JSONNode &p) const override
   {
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
      std::vector<std::string> gammaParnames;
      RooArgSet observables = RooJSONFactoryWSTool::readAxes(p);

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
            if (gammaParnames.empty()) {
               if (auto staterrorParams = findStaterror(comp).find("parameters")) {
                  for (const auto &v : staterrorParams->children()) {
                     gammaParnames.push_back(v.val());
                  }
               }
            }
         }
         data.emplace_back(std::move(dh));
      }

      RooAbsArg *mcStatObject = nullptr;
      RooArgList constraints;
      if (!sumW.empty()) {
         std::string channelName = name;
         erasePrefix(channelName, "model_");

         std::vector<double> errs(sumW.size());
         for (size_t i = 0; i < sumW.size(); ++i) {
            errs[i] = std::sqrt(sumW2[i]) / sumW[i];
            // avoid negative sigma. This NP will be set constant anyway later
            errs[i] = std::max(errs[i], 0.);
         }

         mcStatObject =
            &createPHF("mc_stat_" + channelName, "stat_" + channelName, gammaParnames, errs, *tool, constraints,
                       observables, statErrType, defaultGammaMin, defaultStatErrorGammaMax, statErrThresh);
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
         tool->wsEmplace<RooRealSumPdf>(name, funcs, coefs, true);
      } else {
         std::string sumName = name + "_model";
         erasePrefix(sumName, "model_");
         auto &sum = tool->wsEmplace<RooRealSumPdf>(sumName, funcs, coefs, true);
         sum.SetTitle(name.c_str());
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
      elem["interpolationCodes"].fill_seq(fip->interpolationCodes());
      RooJSONFactoryWSTool::fillSeq(elem["vars"], fip->variables());
      elem["nom"] << fip->nominal();
      elem["high"].fill_seq(fip->high(), fip->variables().size());
      elem["low"].fill_seq(fip->low(), fip->variables().size());
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
      RooJSONFactoryWSTool::fillSeq(elem["high"], pip->highList(), pip->paramList().size());
      RooJSONFactoryWSTool::fillSeq(elem["low"], pip->lowList(), pip->paramList().size());
      return true;
   }
};

class PiecewiseInterpolationFactory : public RooFit::JSONIO::Importer {
public:
   bool importArg(RooJSONFactoryWSTool *tool, const JSONNode &p) const override
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
   bool importArg(RooJSONFactoryWSTool *tool, const JSONNode &p) const override
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
   std::vector<std::string> parameters;
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
   std::vector<std::string> staterrorParameters;
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

namespace {

bool verbose = false;

}

bool tryExportHistFactory(RooJSONFactoryWSTool *tool, const std::string &pdfname, const RooRealSumPdf *sumpdf,
                          JSONNode &elem)
{
   RooWorkspace *ws = tool->workspace();
   RooArgSet customModifiers;

   if (!sumpdf) {
      if (verbose) {
         std::cout << pdfname << " is not a sumpdf" << std::endl;
      }
      return false;
   }

   std::string channelName = pdfname;
   erasePrefix(channelName, "model_");
   eraseSuffix(channelName, "_model");

   for (RooAbsArg *sample : sumpdf->funcList()) {
      if (!dynamic_cast<RooProduct *>(sample) && !dynamic_cast<RooRealSumPdf *>(sample)) {
         if (verbose)
            std::cout << "sample " << sample->GetName() << " is no RooProduct or RooRealSumPdf in " << pdfname
                      << std::endl;
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
      std::vector<RooStats::HistFactory::FlexibleInterpVar *> fips;
      std::vector<ParamHistFunc *> phfs;

      const auto func = sumpdf->funcList().at(sampleidx);
      Sample sample(func->GetName());
      erasePrefix(sample.name, "L_x_");
      eraseSuffix(sample.name, "_shapes");
      eraseSuffix(sample.name, "_" + channelName);
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
         if (TString(e->GetName()).Contains("binWidth")) {
            // The bin width modifiers are handled separately. We can't just
            // check for the RooBinWidthFunction type here, because prior to
            // ROOT 6.26, the multiplication with the inverse bin width was
            // done in a different way (like a normfactor with a RooRealVar,
            // but it was stored in the dataset).
            // Fortunately, the name was similar, so we can match the modifier
            // name.
         } else if (auto constVar = dynamic_cast<RooConstVar *>(e)) {
            if (constVar->getVal() != 1.) {
               sample.normfactors.emplace_back(*e);
            }
         } else if (auto par = dynamic_cast<RooRealVar *>(e)) {
            addNormFactor(par, sample, ws);
         } else if (auto hf = dynamic_cast<const RooHistFunc *>(e)) {
            updateObservables(hf->dataHist());
         } else if (auto phf = dynamic_cast<ParamHistFunc *>(e)) {
            phfs.push_back(phf);
         } else if (auto fip = dynamic_cast<RooStats::HistFactory::FlexibleInterpVar *>(e)) {
            // some (modified) histfactory models have several instances of FlexibleInterpVar
            // we collect and merge them here
            fips.push_back(fip);
         } else if (!pip && (pip = dynamic_cast<PiecewiseInterpolation *>(e))) {
         } else if (auto real = dynamic_cast<RooAbsReal *>(e)) {
            sample.otherElements.push_back(real);
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
      for (auto *fip : fips) {
         for (size_t i = 0; i < fip->variables().size(); ++i) {
            RooAbsArg *var = fip->variables().at(i);
            std::string sysname(var->GetName());
            erasePrefix(sysname, "alpha_");
            const auto *constraint = findConstraint(var);
            if (!constraint && !var->isConstant()) {
               RooJSONFactoryWSTool::error("cannot find constraint for " + std::string(var->GetName()));
            } else {
               sample.normsys.emplace_back(sysname, var, fip->high()[i], fip->low()[i],
                                           constraint ? constraint->IsA() : nullptr);
            }
         }
      }
      sortByName(sample.normsys);

      // sort and configure the histosys
      if (pip) {
         for (size_t i = 0; i < pip->paramList().size(); ++i) {
            RooAbsArg *var = pip->paramList().at(i);
            std::string sysname(var->GetName());
            erasePrefix(sysname, "alpha_");
            if (auto lo = dynamic_cast<RooHistFunc *>(pip->lowList().at(i))) {
               if (auto hi = dynamic_cast<RooHistFunc *>(pip->highList().at(i))) {
                  const auto *constraint = findConstraint(var);
                  if (!constraint && !var->isConstant()) {
                     RooJSONFactoryWSTool::error("cannot find constraint for " + std::string(var->GetName()));
                  } else {
                     sample.histosys.emplace_back(sysname, var, lo, hi, constraint ? constraint->IsA() : nullptr);
                  }
               }
            }
         }
         sortByName(sample.histosys);
      }

      for (ParamHistFunc *phf : phfs) {
         if (startsWith(std::string(phf->GetName()), "mc_stat_")) { // MC stat uncertainty
            int idx = 0;
            for (const auto &g : phf->paramList()) {
               sample.staterrorParameters.push_back(g->GetName());
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
            erasePrefix(sys.name, channelName + "_");
            bool isshapesys = eraseSuffix(sys.name, "_ShapeSys") || eraseSuffix(sys.name, "_shapeSys");
            bool isshapefactor = eraseSuffix(sys.name, "_ShapeFactor") || eraseSuffix(sys.name, "_shapeFactor");

            for (const auto &g : phf->paramList()) {
               sys.parameters.push_back(g->GetName());
               RooAbsPdf *constraint = nullptr;
               if (isshapesys) {
                  constraint = findConstraint(g);
                  if (!constraint)
                     constraint = ws->pdf(constraintName(g->GetName()));
                  if (!constraint && !g->isConstant()) {
                     RooJSONFactoryWSTool::error("cannot find constraint for " + std::string(g->GetName()));
                  }
               } else if (!isshapefactor) {
                  RooJSONFactoryWSTool::error("unknown type of shapesys " + std::string(phf->GetName()));
               }
               if (!constraint) {
                  sys.constraints.push_back(0.0);
               } else if (auto constraint_p = dynamic_cast<RooPoisson *>(constraint)) {
                  sys.constraints.push_back(1. / std::sqrt(constraint_p->getX().getVal()));
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
            // this reconstruction is inherently imprecise, so we truncate it at some decimal places to make sure that
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
      modifiers.set_seq();

      for (const auto &nf : sample.normfactors) {
         auto &mod = modifiers.append_child();
         mod.set_map();
         mod["name"] << nf.name;
         mod["parameter"] << nf.param->GetName();
         mod["type"] << "normfactor";
         if (nf.constraint) {
            mod["constraint_name"] << nf.constraint->GetName();
            tool->queueExport(*nf.constraint);
         }
      }

      for (const auto &sys : sample.normsys) {
         auto &mod = modifiers.append_child();
         mod.set_map();
         mod["name"] << sys.name;
         mod["type"] << "normsys";
         mod["parameter"] << sys.param->GetName();
         mod["constraint"] << toString(sys.constraint);
         auto &data = mod["data"].set_map();
         data["lo"] << sys.low;
         data["hi"] << sys.high;
      }

      for (const auto &sys : sample.histosys) {
         auto &mod = modifiers.append_child();
         mod.set_map();
         mod["name"] << sys.name;
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
         auto &mod = modifiers.append_child();
         mod.set_map();
         mod["name"] << sys.name;
         mod["type"] << "shapesys";
         optionallyExportGammaParameters(mod, sys.name, sys.parameters);
         mod["constraint"] << toString(sys.constraint);
         if (sys.constraint) {
            auto &vals = mod["data"].set_map()["vals"];
            vals.fill_seq(sys.constraints);
         } else {
            auto &vals = mod["data"].set_map()["vals"];
            vals.set_seq();
            for (std::size_t i = 0; i < sys.parameters.size(); ++i) {
               vals.append_child() << 0;
            }
         }
      }

      for (const auto &other : sample.otherElements) {
         auto &mod = modifiers.append_child();
         mod.set_map();
         mod["name"] << other->GetName();
         customModifiers.add(*other);
         mod["type"] << "custom";
      }

      if (sample.useBarlowBeestonLight) {
         auto &mod = modifiers.append_child();
         mod.set_map();
         mod["name"] << ::Literals::staterror;
         mod["type"] << ::Literals::staterror;
         optionallyExportGammaParameters(mod, "stat_" + channelName, sample.staterrorParameters);
         mod["constraint"] << toString(sample.barlowBeestonLightConstraint);
      }

      if (!observablesWritten) {
         auto &output = elem["axes"].set_seq();
         for (auto *obs : static_range_cast<RooRealVar *>(*varSet)) {
            auto &out = output.append_child().set_map();
            std::string name = obs->GetName();
            RooJSONFactoryWSTool::testValidName(name, false);
            out["name"] << name;
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
         auto thispdf = dynamic_cast<RooRealSumPdf *>(v);
         if (thispdf) {
            if (!sumpdf)
               sumpdf = thispdf;
            else
               return false;
         }
      }
      if (!sumpdf)
         return false;

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
