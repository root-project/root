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
#include <RooFormulaVar.h>
#include <RooLognormal.h>
#include <RooGaussian.h>
#include <RooProduct.h>
#include <RooWorkspace.h>

#include <regex>

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
   if (!g)
      return nullptr;
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
void optionallyExportGammaParameters(JSONNode &mod, std::string const &sysname, std::vector<RooAbsReal *> const &params,
                                     bool forceExport = true)
{
   std::vector<std::string> paramNames;
   bool needExport = forceExport;
   for (std::size_t i = 0; i < params.size(); ++i) {
      std::string name(params[i]->GetName());
      paramNames.push_back(name);
      if (name != defaultGammaName(sysname, i)) {
         needExport = true;
      }
   }
   if (needExport) {
      mod["parameters"].fill_seq(paramNames);
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

ParamHistFunc &createPHF(const std::string &phfname, std::string const &sysname,
                         const std::vector<std::string> &parnames, const std::vector<double> &vals,
                         RooJSONFactoryWSTool &tool, RooAbsCollection &constraints, const RooArgSet &observables,
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

RooAbsPdf &
getOrCreateConstraint(RooJSONFactoryWSTool &tool, const JSONNode &mod, RooRealVar &param, const std::string &sample)
{
   if (auto constrName = mod.find("constraint_name")) {
      auto constraint_name = constrName->val();
      auto constraint = tool.workspace()->pdf(constraint_name);
      if (!constraint) {
         constraint = tool.request<RooAbsPdf>(constrName->val(), sample);
      }
      if (!constraint) {
         RooJSONFactoryWSTool::error("unable to find definition of of constraint '" + constraint_name +
                                     "' for modifier '" + RooJSONFactoryWSTool::name(mod) + "'");
      }
      if (auto gauss = dynamic_cast<RooGaussian *const>(constraint)) {
         param.setError(gauss->getSigma().getVal());
      }
      return *constraint;
   } else {
      std::cout << "creating new constraint for " << param << std::endl;
      std::string constraint_type = "Gauss";
      if (auto constrType = mod.find("constraint_type")) {
         constraint_type = constrType->val();
      }
      if (constraint_type == "Gauss") {
         param.setError(1.0);
         return getOrCreate<RooGaussian>(*tool.workspace(), constraintName(param.GetName()), param,
                                         *tool.workspace()->var(std::string("nom_") + param.GetName()), 1.);
      }
      RooJSONFactoryWSTool::error("unknown or invalid constraint for modifier '" + RooJSONFactoryWSTool::name(mod) +
                                  "'");
   }
}

bool importHistSample(RooJSONFactoryWSTool &tool, RooDataHist &dh, RooArgSet const &varlist,
                      RooAbsArg const *mcStatObject, const std::string &fprefix, const JSONNode &p,
                      RooArgSet &constraints)
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
      std::vector<int> overall_interp;

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
            if (mod.has_child("constraint_name") || mod.has_child("constraint_type")) {
               // for norm factors, constraints are optional
               constraints.add(getOrCreateConstraint(tool, mod, constrParam, sampleName));
            }
         } else if (modtype == "normsys") {
            auto *parameter = mod.find("parameter");
            std::string parname(parameter ? parameter->val() : "alpha_" + sysname);
            createNominal(ws, parname, 0.0, -10, 10);
            auto &par = getOrCreate<RooRealVar>(ws, parname, 0., -5, 5);
            overall_nps.add(par);
            auto &data = mod["data"];
            int interp = 4;
            if (mod.has_child("interpolation")) {
               interp = mod["interpolation"].val_int();
            }
            double low = data["lo"].val_double();
            double high = data["hi"].val_double();

            // the below contains a a hack to cut off variations that go below 0
            // this is needed because with interpolation code 4, which is the default, interpolation is done in
            // log-space. hence, values <= 0 result in NaN which propagate throughout the model and cause evaluations to
            // fail if you know a nicer way to solve this, please go ahead and fix the lines below
            if (interp == 4 && low <= 0)
               low = std::numeric_limits<double>::epsilon();
            if (interp == 4 && high <= 0)
               high = std::numeric_limits<double>::epsilon();

            overall_low.push_back(low);
            overall_high.push_back(high);
            overall_interp.push_back(interp);

            constraints.add(getOrCreateConstraint(tool, mod, par, sampleName));
         } else if (modtype == "histosys") {
            auto *parameter = mod.find("parameter");
            std::string parname(parameter ? parameter->val() : "alpha_" + sysname);
            createNominal(ws, parname, 0.0, -10, 10);
            auto &par = getOrCreate<RooRealVar>(ws, parname, 0., -5, 5);
            histNps.add(par);
            auto &data = mod["data"];
            histoLo.add(tool.wsEmplace<RooHistFunc>(
               sysname + "Low_" + prefixedName, varlist,
               RooJSONFactoryWSTool::readBinnedData(data["lo"], sysname + "Low_" + prefixedName, varlist)));
            histoHi.add(tool.wsEmplace<RooHistFunc>(
               sysname + "High_" + prefixedName, varlist,
               RooJSONFactoryWSTool::readBinnedData(data["hi"], sysname + "High_" + prefixedName, varlist)));
            constraints.add(getOrCreateConstraint(tool, mod, par, sampleName));
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
            std::string constraint(mod.has_child("constraint_type") ? mod["constraint_type"].val()
                                   : mod.has_child("constraint")    ? mod["constraint"].val()
                                                                    : "unknown");
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
                                                                            overall_high, overall_interp);
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
         if (staterr.has_child("constraint_type"))
            statErrType = staterr["constraint_type"].val();
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
      RooArgSet constraints;
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

struct NormFactor {
   std::string name;
   RooAbsReal const *param = nullptr;
   RooAbsPdf const *constraint = nullptr;
   TClass *constraintType = RooGaussian::Class();
   NormFactor(RooAbsReal const &par, const RooAbsPdf *constr = nullptr)
      : name{par.GetName()}, param{&par}, constraint{constr}
   {
   }
};

struct NormSys {
   std::string name = "";
   RooAbsReal const *param = nullptr;
   double low = 1.;
   double high = 1.;
   int interpolationCode = 4;
   RooAbsPdf const *constraint = nullptr;
   TClass *constraintType = RooGaussian::Class();
   NormSys() {};
   NormSys(const std::string &n, RooAbsReal *const p, double h, double l, int i, const RooAbsPdf *c)
      : name(n), param(p), low(l), high(h), interpolationCode(i), constraint(c), constraintType(c->IsA())
   {
   }
};

struct HistoSys {
   std::string name;
   RooAbsReal const *param = nullptr;
   std::vector<double> low;
   std::vector<double> high;
   RooAbsPdf const *constraint = nullptr;
   TClass *constraintType = RooGaussian::Class();
   HistoSys(const std::string &n, RooAbsReal *const p, RooHistFunc *l, RooHistFunc *h, const RooAbsPdf *c)
      : name(n), param(p), constraint(c), constraintType(c->IsA())
   {
      low.assign(l->dataHist().weightArray(), l->dataHist().weightArray() + l->dataHist().numEntries());
      high.assign(h->dataHist().weightArray(), h->dataHist().weightArray() + h->dataHist().numEntries());
   }
};
struct ShapeSys {
   std::string name;
   std::vector<double> constraints;
   std::vector<RooAbsReal *> parameters;
   RooAbsPdf const *constraint = nullptr;
   TClass *constraintType = RooGaussian::Class();
   ShapeSys(const std::string &n) : name{n} {}
};

struct GenericElement {
   std::string name;
   RooAbsReal *function = nullptr;
   GenericElement(RooAbsReal *e) : name(e->GetName()), function(e) {};
};

std::string stripOuterParens(const std::string &s)
{
   size_t start = 0;
   size_t end = s.size();

   while (start < end && s[start] == '(' && s[end - 1] == ')') {
      int depth = 0;
      bool balanced = true;
      for (size_t i = start; i < end - 1; ++i) {
         if (s[i] == '(')
            ++depth;
         else if (s[i] == ')')
            --depth;
         if (depth == 0 && i < end - 1) {
            balanced = false;
            break;
         }
      }
      if (balanced) {
         ++start;
         --end;
      } else {
         break;
      }
   }
   return s.substr(start, end - start);
}

std::vector<std::string> splitTopLevelProduct(const std::string &expr)
{
   std::vector<std::string> parts;
   int depth = 0;
   size_t start = 0;
   bool foundTopLevelStar = false;

   for (size_t i = 0; i < expr.size(); ++i) {
      char c = expr[i];
      if (c == '(') {
         ++depth;
      } else if (c == ')') {
         --depth;
      } else if (c == '*' && depth == 0) {
         foundTopLevelStar = true;
         std::string sub = expr.substr(start, i - start);
         parts.push_back(stripOuterParens(sub));
         start = i + 1;
      }
   }

   if (!foundTopLevelStar) {
      return {}; // Not a top-level product
   }

   std::string sub = expr.substr(start);
   parts.push_back(stripOuterParens(sub));
   return parts;
}

#include <regex>
#include <string>
#include <cctype>
#include <cstdlib>
#include <iostream>

NormSys parseOverallModifierFormula(const std::string &s, RooFormulaVar *formula)
{
   static const std::regex pattern(
      R"(^\s*1(?:\.0)?\s*([\+\-])\s*([a-zA-Z_][a-zA-Z0-9_]*|[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)\s*\*\s*([a-zA-Z_][a-zA-Z0-9_]*|[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)\s*$)");

   NormSys sys;
   double sign = 1.0;

   std::smatch match;
   if (std::regex_match(s, match, pattern)) {
      if (match[1].str() == "-") {
         sign = -1.0;
      }

      std::string token2 = match[2].str();
      std::string token3 = match[4].str();

      RooAbsReal *p2 = static_cast<RooAbsReal *>(formula->getParameter(token2.c_str()));
      RooAbsReal *p3 = static_cast<RooAbsReal *>(formula->getParameter(token3.c_str()));
      RooRealVar *v2 = dynamic_cast<RooRealVar *>(p2);
      RooRealVar *v3 = dynamic_cast<RooRealVar *>(p3);

      auto *constr2 = findConstraint(v2);
      auto *constr3 = findConstraint(v3);

      if (constr2 && !p3) {
         sys.name = p2->GetName();
         sys.param = p2;
         sys.high = sign * std::stod(token3);
         sys.low = -sign * std::stod(token3);
      } else if (!p2 && constr3) {
         sys.name = p3->GetName();
         sys.param = p3;
         sys.high = sign * std::stod(token2);
         sys.low = -sign * std::stod(token2);
      } else if (constr2 && p3 && !constr3) {
         sys.name = v2->GetName();
         sys.param = v2;
         sys.high = sign * p3->getVal();
         sys.low = -sign * p3->getVal();
      } else if (p2 && !constr2 && constr3) {
         sys.name = v3->GetName();
         sys.param = v3;
         sys.high = sign * p2->getVal();
         sys.low = -sign * p2->getVal();
      }

      // interpolation code 1 means linear, which is what we have here
      sys.interpolationCode = 1;

      erasePrefix(sys.name, "alpha_");
   }
   return sys;
}

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

struct Sample {
   std::string name;
   std::vector<double> hist;
   std::vector<double> histError;
   std::vector<NormFactor> normfactors;
   std::vector<NormSys> normsys;
   std::vector<HistoSys> histosys;
   std::vector<ShapeSys> shapesys;
   std::vector<GenericElement> tmpElements;
   std::vector<GenericElement> otherElements;
   bool useBarlowBeestonLight = false;
   std::vector<RooAbsReal *> staterrorParameters;
   TClass *barlowBeestonLightConstraintType = RooPoisson::Class();
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

struct Channel {
   std::string name;
   std::vector<Sample> samples;
   std::map<int, double> tot_yield;
   std::map<int, double> tot_yield2;
   std::map<int, double> rel_errors;
   RooArgSet const *varSet = nullptr;
   long unsigned int nBins = 0;
};

Channel readChannel(RooJSONFactoryWSTool *tool, const std::string &pdfname, const RooRealSumPdf *sumpdf)
{
   Channel channel;

   RooWorkspace *ws = tool->workspace();

   channel.name = pdfname;
   erasePrefix(channel.name, "model_");
   eraseSuffix(channel.name, "_model");

   for (size_t sampleidx = 0; sampleidx < sumpdf->funcList().size(); ++sampleidx) {
      PiecewiseInterpolation *pip = nullptr;
      std::vector<ParamHistFunc *> phfs;

      const auto func = sumpdf->funcList().at(sampleidx);
      Sample sample(func->GetName());
      erasePrefix(sample.name, "L_x_");
      eraseSuffix(sample.name, "_shapes");
      eraseSuffix(sample.name, "_" + channel.name);
      erasePrefix(sample.name, pdfname + "_");

      auto updateObservables = [&](RooDataHist const &dataHist) {
         if (channel.varSet == nullptr) {
            channel.varSet = dataHist.get();
            channel.nBins = dataHist.numEntries();
         }
         if (sample.hist.empty()) {
            auto *w = dataHist.weightArray();
            sample.hist.assign(w, w + dataHist.numEntries());
         }
      };
      auto processElements = [&](const auto &elements, auto &&self) -> void {
         for (RooAbsArg *e : elements) {
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
                  sample.normfactors.emplace_back(*constVar);
               }
            } else if (auto par = dynamic_cast<RooRealVar *>(e)) {
               addNormFactor(par, sample, ws);
            } else if (auto hf = dynamic_cast<const RooHistFunc *>(e)) {
               updateObservables(hf->dataHist());
            } else if (auto phf = dynamic_cast<ParamHistFunc *>(e)) {
               phfs.push_back(phf);
            } else if (auto fip = dynamic_cast<RooStats::HistFactory::FlexibleInterpVar *>(e)) {
               // some (modified) histfactory models have several instances of FlexibleInterpVar
               // we collect and merge them
               for (size_t i = 0; i < fip->variables().size(); ++i) {
                  RooAbsReal *var = static_cast<RooAbsReal *>(fip->variables().at(i));
                  std::string sysname(var->GetName());
                  erasePrefix(sysname, "alpha_");
                  const auto *constraint = findConstraint(var);
                  if (!constraint && !var->isConstant()) {
                     RooJSONFactoryWSTool::error("cannot find constraint for " + std::string(var->GetName()));
                  } else {
                     sample.normsys.emplace_back(sysname, var, fip->high()[i], fip->low()[i],
                                                 fip->interpolationCodes()[i], constraint);
                  }
               }
            } else if (!pip && (pip = dynamic_cast<PiecewiseInterpolation *>(e))) {
               // nothing to do here, already assigned
            } else if (RooFormulaVar *formula = dynamic_cast<RooFormulaVar *>(e)) {
               // people do a lot of fancy stuff with RooFormulaVar, like including NormSys via explicit formulae.
               // let's try to decompose it into building blocks
               TString expression(formula->expression());
               for (size_t i = formula->nParameters(); i--;) {
                  const RooAbsArg *p = formula->getParameter(i);
                  expression.ReplaceAll(("x[" + std::to_string(i) + "]").c_str(), p->GetName());
                  expression.ReplaceAll(("@" + std::to_string(i)).c_str(), p->GetName());
               }
               auto components = splitTopLevelProduct(expression.Data());
               if (components.size() == 0) {
                  // it's not a product, let's just treat it as an unknown element
                  sample.otherElements.push_back(formula);
               } else {
                  // it is a prododuct, we can try to handle the elements separately
                  std::vector<RooAbsArg *> realComponents;
                  int idx = 0;
                  for (auto &comp : components) {
                     // check if this is a trivial element of a product, we can treat it as its own modifier
                     auto *part = formula->getParameter(comp.c_str());
                     if (part) {
                        realComponents.push_back(part);
                        continue;
                     }
                     // check if this is an attempt at explicitly encoding an overallSys
                     auto normsys = parseOverallModifierFormula(comp, formula);
                     if (normsys.param) {
                        sample.normsys.emplace_back(std::move(normsys));
                        continue;
                     }

                     // this is something non-trivial, let's deal with it separately
                     std::string name = std::string(formula->GetName()) + "_part" + std::to_string(idx);
                     ++idx;
                     auto *var = new RooFormulaVar(name.c_str(), name.c_str(), comp.c_str(), formula->dependents());
                     sample.tmpElements.push_back({var});
                  }
                  self(realComponents, self);
               }
            } else if (auto real = dynamic_cast<RooAbsReal *>(e)) {
               sample.otherElements.push_back(real);
            }
         }
      };

      RooArgSet elems;
      collectElements(elems, func);
      collectElements(elems, sumpdf->coefList().at(sampleidx));
      processElements(elems, processElements);

      // see if we can get the observables
      if (pip) {
         if (auto nh = dynamic_cast<RooHistFunc const *>(pip->nominalHist())) {
            updateObservables(nh->dataHist());
         }
      }

      // sort and configure norms
      sortByName(sample.normfactors);
      sortByName(sample.normsys);

      // sort and configure the histosys
      if (pip) {
         for (size_t i = 0; i < pip->paramList().size(); ++i) {
            RooAbsReal *var = static_cast<RooAbsReal *>(pip->paramList().at(i));
            std::string sysname(var->GetName());
            erasePrefix(sysname, "alpha_");
            if (auto lo = dynamic_cast<RooHistFunc *>(pip->lowList().at(i))) {
               if (auto hi = dynamic_cast<RooHistFunc *>(pip->highList().at(i))) {
                  const auto *constraint = findConstraint(var);
                  if (!constraint && !var->isConstant()) {
                     RooJSONFactoryWSTool::error("cannot find constraint for " + std::string(var->GetName()));
                  } else {
                     sample.histosys.emplace_back(sysname, var, lo, hi, constraint);
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
               sample.staterrorParameters.push_back(static_cast<RooRealVar *>(g));
               ++idx;
               RooAbsPdf *constraint = findConstraint(g);
               if (channel.tot_yield.find(idx) == channel.tot_yield.end()) {
                  channel.tot_yield[idx] = 0;
                  channel.tot_yield2[idx] = 0;
               }
               channel.tot_yield[idx] += sample.hist[idx - 1];
               channel.tot_yield2[idx] += (sample.hist[idx - 1] * sample.hist[idx - 1]);
               if (constraint) {
                  sample.barlowBeestonLightConstraintType = constraint->IsA();
                  if (RooPoisson *constraint_p = dynamic_cast<RooPoisson *>(constraint)) {
                     double erel = 1. / std::sqrt(constraint_p->getX().getVal());
                     channel.rel_errors[idx] = erel;
                  } else if (RooGaussian *constraint_g = dynamic_cast<RooGaussian *>(constraint)) {
                     double erel = constraint_g->getSigma().getVal() / constraint_g->getMean().getVal();
                     channel.rel_errors[idx] = erel;
                  } else {
                     RooJSONFactoryWSTool::error(
                        "currently, only RooPoisson and RooGaussian are supported as constraint types");
                  }
               }
            }
            sample.useBarlowBeestonLight = true;
         } else { // other ShapeSys
            ShapeSys sys(phf->GetName());
            erasePrefix(sys.name, channel.name + "_");
            bool isshapesys = eraseSuffix(sys.name, "_ShapeSys") || eraseSuffix(sys.name, "_shapeSys");
            bool isshapefactor = eraseSuffix(sys.name, "_ShapeFactor") || eraseSuffix(sys.name, "_shapeFactor");

            for (const auto &g : phf->paramList()) {
               sys.parameters.push_back(static_cast<RooRealVar *>(g));
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
                     sys.constraintType = RooPoisson::Class();
                  }
               } else if (auto constraint_g = dynamic_cast<RooGaussian *>(constraint)) {
                  sys.constraints.push_back(constraint_g->getSigma().getVal() / constraint_g->getMean().getVal());
                  if (!sys.constraint) {
                     sys.constraintType = RooGaussian::Class();
                  }
               }
            }
            sample.shapesys.emplace_back(std::move(sys));
         }
      }
      sortByName(sample.shapesys);

      // add the sample
      channel.samples.emplace_back(std::move(sample));
   }

   sortByName(channel.samples);
   return channel;
}

void configureStatError(Channel &channel)
{
   for (auto &sample : channel.samples) {
      if (sample.useBarlowBeestonLight) {
         sample.histError.resize(sample.hist.size());
         for (auto bin : channel.rel_errors) {
            // reverse engineering the correct partial error
            // the (arbitrary) convention used here is that all samples should have the same relative error
            const int i = bin.first;
            const double relerr_tot = bin.second;
            const double count = sample.hist[i - 1];
            // this reconstruction is inherently imprecise, so we truncate it at some decimal places to make sure that
            // we don't carry around too many useless digits
            sample.histError[i - 1] =
               round_prec(relerr_tot * channel.tot_yield[i] / std::sqrt(channel.tot_yield2[i]) * count, 7);
         }
      }
   }
}

bool exportChannel(RooJSONFactoryWSTool *tool, const Channel &channel, JSONNode &elem)
{
   bool observablesWritten = false;
   for (const auto &sample : channel.samples) {

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
         if (sys.interpolationCode != 4) {
            mod["interpolation"] << sys.interpolationCode;
         }
         if (sys.constraint) {
            mod["constraint_name"] << sys.constraint->GetName();
         } else if (sys.constraintType) {
            mod["constraint_type"] << toString(sys.constraintType);
         }
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
         if (sys.constraint) {
            mod["constraint_name"] << sys.constraint->GetName();
         } else if (sys.constraintType) {
            mod["constraint_type"] << toString(sys.constraintType);
         }
         auto &data = mod["data"].set_map();
         if (channel.nBins != sys.low.size() || channel.nBins != sys.high.size()) {
            std::stringstream ss;
            ss << "inconsistent binning: " << channel.nBins << " bins expected, but " << sys.low.size() << "/"
               << sys.high.size() << " found in nominal histogram errors!";
            RooJSONFactoryWSTool::error(ss.str().c_str());
         }
         RooJSONFactoryWSTool::exportArray(channel.nBins, sys.low.data(), data["lo"].set_map()["contents"]);
         RooJSONFactoryWSTool::exportArray(channel.nBins, sys.high.data(), data["hi"].set_map()["contents"]);
      }

      for (const auto &sys : sample.shapesys) {
         auto &mod = modifiers.append_child();
         mod.set_map();
         mod["name"] << sys.name;
         mod["type"] << "shapesys";
         optionallyExportGammaParameters(mod, sys.name, sys.parameters);
         if (sys.constraint) {
            mod["constraint_name"] << sys.constraint->GetName();
         } else if (sys.constraintType) {
            mod["constraint_type"] << toString(sys.constraintType);
         }
         if (sys.constraint || sys.constraintType) {
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
         mod["name"] << other.name;
         mod["type"] << "custom";
      }
      for (const auto &other : sample.tmpElements) {
         auto &mod = modifiers.append_child();
         mod.set_map();
         mod["name"] << other.name;
         mod["type"] << "custom";
      }

      if (sample.useBarlowBeestonLight) {
         auto &mod = modifiers.append_child();
         mod.set_map();
         mod["name"] << ::Literals::staterror;
         mod["type"] << ::Literals::staterror;
         optionallyExportGammaParameters(mod, "stat_" + channel.name, sample.staterrorParameters);
         mod["constraint_type"] << toString(sample.barlowBeestonLightConstraintType);
      }

      if (!observablesWritten) {
         auto &output = elem["axes"].set_seq();
         for (auto *obs : static_range_cast<RooRealVar *>(*channel.varSet)) {
            auto &out = output.append_child().set_map();
            std::string name = obs->GetName();
            RooJSONFactoryWSTool::testValidName(name, false);
            out["name"] << name;
            writeAxis(out, *obs);
         }
         observablesWritten = true;
      }
      auto &dataNode = s["data"].set_map();
      if (channel.nBins != sample.hist.size()) {
         std::stringstream ss;
         ss << "inconsistent binning: " << channel.nBins << " bins expected, but " << sample.hist.size()
            << " found in nominal histogram!";
         RooJSONFactoryWSTool::error(ss.str().c_str());
      }
      RooJSONFactoryWSTool::exportArray(channel.nBins, sample.hist.data(), dataNode["contents"]);
      if (!sample.histError.empty()) {
         if (channel.nBins != sample.histError.size()) {
            std::stringstream ss;
            ss << "inconsistent binning: " << channel.nBins << " bins expected, but " << sample.histError.size()
               << " found in nominal histogram errors!";
            RooJSONFactoryWSTool::error(ss.str().c_str());
         }
         RooJSONFactoryWSTool::exportArray(channel.nBins, sample.histError.data(), dataNode["errors"]);
      }
   }

   return true;
}

std::vector<RooAbsPdf *> findLostConstraints(const Channel &channel, const std::vector<RooAbsPdf *> &constraints)
{
   // collect all the vars that are used by the model
   std::set<const RooAbsReal *> vars;
   for (const auto &sample : channel.samples) {
      for (const auto &nf : sample.normfactors) {
         vars.insert(nf.param);
      }
      for (const auto &sys : sample.normsys) {
         vars.insert(sys.param);
      }

      for (const auto &sys : sample.histosys) {
         vars.insert(sys.param);
      }
      for (const auto &sys : sample.shapesys) {
         for (const auto &par : sys.parameters) {
            vars.insert(par);
         }
      }
      if (sample.useBarlowBeestonLight) {
         for (const auto &par : sample.staterrorParameters) {
            vars.insert(par);
         }
      }
   }

   // check if there is any constraint present that is unrelated to these vars
   std::vector<RooAbsPdf *> lostConstraints;
   for (auto *pdf : constraints) {
      bool related = false;
      for (const auto *var : vars) {
         if (pdf->dependsOn(*var)) {
            related = true;
         }
      }
      if (!related) {
         lostConstraints.push_back(pdf);
      }
   }
   // return the constraints that would be "lost" when exporting the model
   return lostConstraints;
}

bool tryExportHistFactory(RooJSONFactoryWSTool *tool, const std::string &pdfname, const RooRealSumPdf *sumpdf,
                          std::vector<RooAbsPdf *> constraints, JSONNode &elem)
{
   // some preliminary checks
   if (!sumpdf) {
      if (verbose) {
         std::cout << pdfname << " is not a sumpdf" << std::endl;
      }
      return false;
   }

   for (RooAbsArg *sample : sumpdf->funcList()) {
      if (!dynamic_cast<RooProduct *>(sample) && !dynamic_cast<RooRealSumPdf *>(sample)) {
         if (verbose)
            std::cout << "sample " << sample->GetName() << " is no RooProduct or RooRealSumPdf in " << pdfname
                      << std::endl;
         return false;
      }
   }

   auto channel = readChannel(tool, pdfname, sumpdf);

   // sanity checks
   if (channel.samples.size() == 0)
      return false;
   for (auto &sample : channel.samples) {
      if (sample.hist.empty()) {
         return false;
      }
   }

   // stat error handling
   configureStatError(channel);

   auto lostConstraints = findLostConstraints(channel, constraints);
   // Export all the lost constraints
   for (const auto *constraint : lostConstraints) {
      RooJSONFactoryWSTool::warning(
         "losing constraint term '" + std::string(constraint->GetName()) +
         "', implicit constraints are not supported by HS3 yet! The term will appear in the HS3 file, but will not be "
         "picked up when creating a likelihood from it! You will have to add it manually as an external constraint.");
      tool->queueExport(*constraint);
   }

   // Export all the regular modifiers
   for (const auto &sample : channel.samples) {
      for (auto &modifier : sample.normfactors) {
         if (modifier.constraint) {
            tool->queueExport(*modifier.constraint);
         }
      }
      for (auto &modifier : sample.normsys) {
         if (modifier.constraint) {
            tool->queueExport(*modifier.constraint);
         }
      }
      for (auto &modifier : sample.histosys) {
         if (modifier.constraint) {
            tool->queueExport(*modifier.constraint);
         }
      }
   }

   // Export all the custom modifiers
   for (const auto &sample : channel.samples) {
      for (auto &modifier : sample.otherElements) {
         tool->queueExport(*modifier.function);
      }
      for (auto &modifier : sample.tmpElements) {
         tool->queueExportTemporary(modifier.function);
      }
   }

   // Export all model parameters
   RooArgSet parameters;
   sumpdf->getParameters(channel.varSet, parameters);
   for (RooAbsArg *param : parameters) {
      // This should exclude the global observables
      if (!startsWith(std::string{param->GetName()}, "nom_")) {
         tool->queueExport(*param);
      }
   }

   return exportChannel(tool, channel, elem);
}

class HistFactoryStreamer_ProdPdf : public RooFit::JSONIO::Exporter {
public:
   bool autoExportDependants() const override { return false; }
   bool tryExport(RooJSONFactoryWSTool *tool, const RooProdPdf *prodpdf, JSONNode &elem) const
   {
      std::vector<RooAbsPdf *> constraints;
      RooRealSumPdf *sumpdf = nullptr;
      for (RooAbsArg *v : prodpdf->pdfList()) {
         RooAbsPdf *pdf = static_cast<RooAbsPdf *>(v);
         auto thispdf = dynamic_cast<RooRealSumPdf *>(pdf);
         if (thispdf) {
            if (!sumpdf)
               sumpdf = thispdf;
            else
               return false;
         } else {
            constraints.push_back(pdf);
         }
      }
      if (!sumpdf)
         return false;

      bool ok = tryExportHistFactory(tool, prodpdf->GetName(), sumpdf, constraints, elem);
      return ok;
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
      std::vector<RooAbsPdf *> constraints;
      return tryExportHistFactory(tool, sumpdf->GetName(), sumpdf, constraints, elem);
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
