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
#include <RooFitImplHelpers.h>

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cmath>
#include <iterator>
#include <limits>
#include <map>
#include <optional>
#include <regex>
#include <set>
#include <sstream>
#include <variant>

#include "static_execute.h"
#include "HS3Builder.h"
#include "JSONIOUtils.h"

using RooFit::Detail::JSONNode;
using RooFit::HS3::additivePolynomialLinear;
using RooFit::HS3::Interpolation;
using RooFit::HS3::InterpolationClass;
using RooFit::HS3::multiplicativePiecewiseExponential;
using RooFit::HS3::multiplicativePolynomialExponential;

namespace HS3 = RooFit::HS3;

using namespace RooStats::HistFactory;
using namespace RooStats::HistFactory::Detail;
using namespace RooStats::HistFactory::Detail::MagicConstants;

namespace {

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

std::string interpolationString(const Interpolation &interpolation)
{
   std::stringstream ss;
   ss << R"({"type":")" << toString(interpolation.type) << R"(","in":")" << toString(interpolation.in) << R"(","out":)";
   if (interpolation.out) {
      ss << '"' << toString(*interpolation.out) << '"';
   } else {
      ss << "null";
   }
   ss << '}';
   return ss.str();
}

const char *interpolationClassName(InterpolationClass interpolationClass)
{
   return interpolationClass == InterpolationClass::Piecewise ? "PiecewiseInterpolation" : "FlexibleInterpVar";
}

Interpolation interpolationFromCode(int code, InterpolationClass interpolationClass, const std::string &context)
{
   if (auto interpolation = HS3::interpolationFromCode(code, interpolationClass)) {
      return *interpolation;
   }
   RooJSONFactoryWSTool::error(context + " has unsupported " + interpolationClassName(interpolationClass) + " code " +
                               std::to_string(code));
}

int codeFromInterpolation(const Interpolation &interpolation, InterpolationClass interpolationClass,
                          const std::string &context)
{
   if (auto code = HS3::codeFromInterpolation(interpolation, interpolationClass)) {
      return *code;
   }
   RooJSONFactoryWSTool::error(context + " " + interpolationString(interpolation) + " cannot be represented by " +
                               interpolationClassName(interpolationClass));
}

// The interpolations of the parameters of a PiecewiseInterpolation or FlexibleInterpVar.
std::vector<Interpolation>
toInterpolations(std::vector<int> const &codes, InterpolationClass interpolationClass, std::string const &context)
{
   std::vector<Interpolation> out;
   out.reserve(codes.size());
   for (std::size_t i = 0; i < codes.size(); ++i) {
      out.push_back(
         interpolationFromCode(codes[i], interpolationClass, context + " at parameter index " + std::to_string(i)));
   }
   return out;
}

std::vector<int> toCodes(std::vector<Interpolation> const &interpolations, InterpolationClass interpolationClass,
                         std::string const &context)
{
   std::vector<int> out;
   out.reserve(interpolations.size());
   for (std::size_t i = 0; i < interpolations.size(); ++i) {
      out.push_back(codeFromInterpolation(interpolations[i], interpolationClass,
                                          context + " component 'interpolations' at index " + std::to_string(i)));
   }
   return out;
}

// Reports the problems with an object that doesn't come with a document.
void report(std::string const &name, HS3::Diagnostics &diagnostics)
{
   for (auto &diagnostic : diagnostics) {
      diagnostic.path = name + (diagnostic.path.empty() ? "" : "." + diagnostic.path);
   }
   HS3::report(diagnostics);
}

int interpolationCode(std::optional<Interpolation> const &interpolation,
                      std::optional<Interpolation> const &defaultInterpolation, InterpolationClass interpolationClass,
                      std::string const &context)
{
   if (interpolation) {
      return codeFromInterpolation(*interpolation, interpolationClass, context);
   }
   // HS3 forbids implementation-specific default interpolations.
   if (!defaultInterpolation) {
      RooJSONFactoryWSTool::error(context + " is not given, and the channel has no \"default_interpolation\"");
   }
   return codeFromInterpolation(*defaultInterpolation, interpolationClass, context);
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
   if (auto *constraint = findClient<RooPoisson>(g))
      return constraint;
   if (auto *constraint = findClient<RooGaussian>(g))
      return constraint;
   return findClient<RooLognormal>(g);
}

inline std::string defaultGammaName(std::string const &sysname, std::size_t i)
{
   return "gamma_" + sysname + "_bin_" + std::to_string(i);
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

RooAbsPdf *findNamedConstraint(RooJSONFactoryWSTool &tool, std::string const &constraintName, std::string const &sample)
{
   if (auto *constraint = tool.workspace()->pdf(constraintName)) {
      return constraint;
   }

   try {
      return tool.request<RooAbsPdf>(constraintName, sample);
   } catch (RooJSONFactoryWSTool::DependencyMissingError const &err) {
      if (err.child() != constraintName) {
         throw;
      }
   }

   return nullptr;
}

// The parameter of a modifier, which is created if the document doesn't define it. It's a variable, or a constant if
// the document doesn't give it a domain.
RooAbsReal &getOrCreateParameter(RooJSONFactoryWSTool &tool, std::string const &name, double val, double min,
                                 double max, std::string const &modifier)
{
   RooWorkspace &ws = *tool.workspace();
   TObject *existing = ws.obj(name);
   if (!existing) {
      return getOrCreate<RooRealVar>(ws, name, val, min, max);
   }
   if (dynamic_cast<RooRealVar *>(existing) || dynamic_cast<RooConstVar *>(existing)) {
      return *static_cast<RooAbsReal *>(existing);
   }
   RooJSONFactoryWSTool::error("the parameter '" + name + "' of the modifier '" + modifier +
                               "' is neither a variable nor a constant");
}

void setErrorIfVariable(RooAbsReal &param, double error)
{
   if (auto *var = dynamic_cast<RooRealVar *>(&param)) {
      var->setError(error);
   }
}

// All constraints of parameters that control a whole sample are created here. Following the HS3 standard, the nominal
// value is defined by the type of the modifier, and the standard deviation is one.
void addConstraint(RooJSONFactoryWSTool &tool, HS3::Constraint const &constraint, RooAbsReal &param, double nominal,
                   std::string const &modifier, std::string const &sample, RooArgSet &constraints)
{
   RooWorkspace &ws = *tool.workspace();
   std::visit(HS3::overloaded{
                 [](HS3::Unconstrained) {},
                 [&](HS3::ConstraintType type) {
                    if (type != HS3::ConstraintType::Gauss) {
                       RooJSONFactoryWSTool::error(std::string("legacy constraint value '") + toString(type) +
                                                   "' for modifier '" + modifier +
                                                   "' is a known constraint type, but it cannot be resolved in this "
                                                   "context");
                    }
                    RooRealVar &nom = createNominal(ws, param.GetName(), nominal, -10, 10);
                    setErrorIfVariable(param, 1.0);
                    const std::string name = constraintName(param.GetName());
                    RooAbsPdf *pdf = ws.pdf(name);
                    constraints.add(pdf ? *pdf : tool.wsEmplace<RooGaussian>(name, param, nom, 1.));
                 },
                 [&](HS3::ConstraintPdf const &constraintPdf) {
                    RooAbsPdf *pdf = findNamedConstraint(tool, constraintPdf.pdf.name, sample);
                    if (!pdf) {
                       RooJSONFactoryWSTool::error("unable to find definition of of constraint '" +
                                                   constraintPdf.pdf.name + "' for modifier '" + modifier + "'");
                    }
                    if (auto gauss = dynamic_cast<RooGaussian *>(pdf)) {
                       setErrorIfVariable(param, gauss->getSigma().getVal());
                    }
                    constraints.add(*pdf);
                 },
              },
              constraint);
}

// Create the ParamHistFunc of the gamma parameters, together with their constraints.
ParamHistFunc &createPHF(RooJSONFactoryWSTool &tool, std::string const &phfname, std::string const &sysname,
                         std::vector<HS3::ParameterRef> const &parnames, std::vector<double> const &vals,
                         HS3::GammaConstraint const &constraint, RooArgSet &constraints, RooArgSet const &observables,
                         double gammaMax, double minSigma, std::size_t nBins)
{
   RooWorkspace &ws = *tool.workspace();

   // The validation made sure that there is a value, parameter and constraint for each bin, if they are given.
   RooArgList gammas;
   for (std::size_t i = 0; i < nBins; ++i) {
      const std::string name = parnames.empty() ? defaultGammaName(sysname, i) : parnames[i].name;
      if (auto *existing = dynamic_cast<RooAbsReal *>(ws.obj(name))) {
         gammas.add(*existing);
      } else {
         gammas.add(getOrCreate<RooRealVar>(ws, name, 1., defaultGammaMin, gammaMax));
      }
   }

   // Shape modifiers of the same name in a channel share the function, so they need the same parameters.
   if (RooAbsArg *existing = ws.arg(phfname.c_str())) {
      auto *existingPHF = dynamic_cast<ParamHistFunc *>(existing);
      bool same = existingPHF && existingPHF->paramList().size() == gammas.size();
      for (std::size_t i = 0; same && i < gammas.size(); ++i) {
         same = &existingPHF->paramList()[i] == &gammas[i];
      }
      if (!same) {
         RooJSONFactoryWSTool::error("the modifier '" + sysname + "' has other parameters than another modifier " +
                                     "of this name in the channel");
      }
   }
   auto &phf = tool.wsEmplace<ParamHistFunc>(phfname, observables, gammas);

   auto addGammaConstraints = [&](Constraint::Type type) {
      for (auto const &term : createGammaConstraints(gammas, vals, minSigma, type).constraints) {
         ws.import(*term, RooFit::RecycleConflictNodes(), RooFit::Silence());
         constraints.add(*ws.pdf(term->GetName()));
      }
   };

   if (auto const *pdfs = std::get_if<HS3::GammaConstraintPdfs>(&constraint)) {
      for (auto const &pdfName : pdfs->pdfs) {
         if (!pdfName) {
            continue;
         }
         auto *pdf = findNamedConstraint(tool, pdfName->name, sysname);
         if (!pdf) {
            RooJSONFactoryWSTool::error("unable to find definition of constraint '" + pdfName->name +
                                        "' for modifier '" + sysname + "'");
         }
         constraints.add(*pdf);
      }
   }

   if (vals.empty()) {
      return phf;
   }

   std::visit(HS3::overloaded{
                 [&](HS3::Unconstrained) { configureConstrainedGammas(gammas, vals, minSigma); },
                 [&](HS3::GammaConstraintPdfs const &) { configureConstrainedGammas(gammas, vals, minSigma); },
                 [&](HS3::ConstantGammas) {
                    for (auto *gamma : static_range_cast<RooRealVar *>(gammas)) {
                       gamma->setConstant(true);
                    }
                 },
                 [&](HS3::ConstraintType type) {
                    switch (type) {
                    case HS3::ConstraintType::Gauss: addGammaConstraints(Constraint::Gaussian); break;
                    case HS3::ConstraintType::Poisson: addGammaConstraints(Constraint::Poisson); break;
                    case HS3::ConstraintType::LogNormal:
                       RooJSONFactoryWSTool::error(std::string("unsupported constraint '") + toString(type) +
                                                   "' for the gamma parameters of '" + sysname +
                                                   "', supported are \"Poisson\" and \"Gauss\"");
                    }
                 },
              },
              constraint);

   return phf;
}

// Whether `phf` implements the staterror of a channel, among the ParamHistFuncs `phfs` of a sample. HistFactory marks
// it with an attribute. Workspaces from before the attribute existed have none: then it is recognized by the name that
// HistFactory gave it.
bool isStatError(ParamHistFunc const &phf, std::vector<ParamHistFunc *> const &phfs)
{
   if (phf.getAttribute(statErrorAttribute)) {
      return true;
   }
   for (ParamHistFunc const *other : phfs) {
      if (other->getAttribute(statErrorAttribute)) {
         return false;
      }
   }
   return startsWith(std::string(phf.GetName()), "mc_stat_");
}

double poissonTau(RooPoisson const &constraint, RooAbsArg const &gamma)
{
   auto const *mean = dynamic_cast<RooProduct const *>(&constraint.getMean());
   if (!mean) {
      RooJSONFactoryWSTool::error("Poisson gamma constraint mean is not a RooProduct: " +
                                  std::string(constraint.GetName()));
   }

   // The mean is the product of the gamma and tau. Imported workspaces can also represent tau as a RooRealVar, which
   // is not necessarily constant.
   std::vector<RooAbsReal const *> others;
   for (RooAbsArg *arg : mean->servers()) {
      if (arg != &gamma) {
         if (auto const *real = dynamic_cast<RooAbsReal const *>(arg)) {
            others.push_back(real);
         }
      }
   }
   if (others.size() == 1) {
      return others.front()->getVal();
   }
   for (RooAbsReal const *real : others) {
      if (dynamic_cast<RooConstVar const *>(real) || real->isConstant()) {
         return real->getVal();
      }
   }

   RooJSONFactoryWSTool::error("Could not find tau component in Poisson gamma constraint mean: " +
                               std::string(constraint.GetName()));
   return std::numeric_limits<double>::quiet_NaN();
}

// Returns the relative uncertainty encoded by a gamma constraint pdf. Only RooPoisson (via its tau) and RooGaussian
// (via sigma/mean) are supported; anything else raises an error.
double constraintRelError(RooAbsPdf const &constraint, RooAbsArg const &gamma)
{
   if (auto constraintP = dynamic_cast<RooPoisson const *>(&constraint)) {
      return 1. / std::sqrt(poissonTau(*constraintP, gamma));
   }
   if (auto constraintG = dynamic_cast<RooGaussian const *>(&constraint)) {
      return constraintG->getSigma().getVal() / constraintG->getMean().getVal();
   }
   RooJSONFactoryWSTool::error("currently, only RooPoisson and RooGaussian are supported as constraint types");
   return std::numeric_limits<double>::quiet_NaN();
}

struct ChannelContext {
   RooJSONFactoryWSTool &tool;
   HS3::HistFactoryDist const &dist;
   RooArgSet const &observables;
   std::string fprefix;
   std::string channelName;
   std::size_t nBins;
   RooAbsArg const *mcStatObject;
   RooArgSet &constraints;
};

// Collects what the modifiers of a sample contribute to the RooFit objects.
struct SampleElements {
   RooArgList shapeElems;
   RooArgList normElems;

   RooArgList overallNps;
   std::vector<double> overallLow;
   std::vector<double> overallHigh;
   std::vector<int> overallInterp;

   RooArgList histNps;
   RooArgList histoLo;
   RooArgList histoHi;
   std::vector<int> histoInterp;
};

void buildSample(ChannelContext const &ctx, HS3::Sample const &sample, RooDataHist &dh)
{
   RooJSONFactoryWSTool &tool = ctx.tool;
   RooWorkspace &ws = *tool.workspace();

   std::string const &sampleName = sample.name;
   std::string prefixedName = ctx.fprefix + "_" + sampleName;

   // The objects of a sample are not shared, so an existing object of the same name is a name clash.
   auto newName = [&](std::string name) {
      if (ws.arg(name.c_str())) {
         RooJSONFactoryWSTool::error("the object '" + name + "' for sample '" + sampleName + "' of channel '" +
                                     ctx.channelName + "' has the name of an existing object");
      }
      return name;
   };

   auto &hf = tool.wsEmplace<RooHistFunc>(newName("hist_" + prefixedName), ctx.observables, dh);
   hf.SetTitle(sampleName.c_str());

   SampleElements elems;
   elems.shapeElems.add(tool.wsEmplace<RooBinWidthFunction>(newName(prefixedName + "_binWidth"), hf, true));

   const bool hasStatError = std::any_of(sample.modifiers.begin(), sample.modifiers.end(),
                                         [](auto const &m) { return std::holds_alternative<HS3::StatError>(m); });
   if (hasStatError) {
      elems.shapeElems.add(*ctx.mcStatObject);
   }

   auto context = [&](const char *type, std::string const &name) {
      return std::string("interpolation for ") + type + " modifier '" + name + "' in sample '" + sampleName +
             "' of channel '" + ctx.channelName + "'";
   };

   for (HS3::Modifier const &modifier : sample.modifiers) {
      std::visit(
         HS3::overloaded{
            [&](HS3::StatError const &) {
               // this is dealt with at a different place, ignore it for now
            },
            [&](HS3::NormFactor const &mod) {
               RooAbsReal &param = getOrCreateParameter(tool, mod.parameter.name, 1., -3, 5, mod.name);
               setErrorIfVariable(param, 0.0);
               elems.normElems.add(param);
               // for norm factors, constraints are optional
               addConstraint(tool, mod.constraint, param, 1.0, mod.name, sampleName, ctx.constraints);
            },
            [&](HS3::NormSys const &mod) {
               RooAbsReal &param = getOrCreateParameter(tool, mod.parameter.name, 0., -5, 5, mod.name);
               elems.overallNps.add(param);
               const int interp = interpolationCode(mod.interpolation, ctx.dist.defaultInterpolation,
                                                    InterpolationClass::Flexible, context("normsys", mod.name));
               double low = mod.lo;
               double high = mod.hi;

               // the below contains a a hack to cut off variations that go below 0
               // This is needed because FlexibleInterpVar code 4 interpolates in log-space. Hence, values <= 0 result
               // in NaN, which propagates throughout the model and causes evaluations to fail. If you know a nicer way
               // to solve this, please go ahead and fix the lines below.
               if (interp == 4 && low <= 0)
                  low = std::numeric_limits<double>::epsilon();
               if (interp == 4 && high <= 0)
                  high = std::numeric_limits<double>::epsilon();

               elems.overallLow.push_back(low);
               elems.overallHigh.push_back(high);
               elems.overallInterp.push_back(interp);

               addConstraint(tool, mod.constraint, param, 0.0, mod.name, sampleName, ctx.constraints);
            },
            [&](HS3::HistoSys const &mod) {
               RooAbsReal &param = getOrCreateParameter(tool, mod.parameter.name, 0., -5, 5, mod.name);
               elems.histNps.add(param);
               auto addVariation = [&](RooArgList &list, HS3::HistoSysVariation const &variation, const char *suffix) {
                  const std::string name = newName(mod.name + suffix + prefixedName);
                  list.add(
                     tool.wsEmplace<RooHistFunc>(name, ctx.observables,
                                                 HS3::createDataHist(name, ctx.observables, variation.contents,
                                                                     variation.errors ? &*variation.errors : nullptr)));
               };
               addVariation(elems.histoLo, mod.lo, "Low_");
               addVariation(elems.histoHi, mod.hi, "High_");
               elems.histoInterp.push_back(interpolationCode(mod.interpolation, ctx.dist.defaultInterpolation,
                                                             InterpolationClass::Piecewise,
                                                             context("histosys", mod.name)));
               addConstraint(tool, mod.constraint, param, 0.0, mod.name, sampleName, ctx.constraints);
            },
            [&](HS3::ShapeSys const &mod) {
               const bool constrained = std::visit(
                  HS3::overloaded{[](HS3::Unconstrained) { return false; }, [](HS3::ConstantGammas) { return false; },
                                  [&](HS3::ConstraintType) { return !mod.vals.empty(); },
                                  [](HS3::GammaConstraintPdfs const &constraints) {
                                     return std::any_of(constraints.pdfs.begin(), constraints.pdfs.end(),
                                                        [](auto const &pdf) { return pdf.has_value(); });
                                  }},
                  mod.constraint);
               // Follow the naming of HistFactory: "<channel>_<name>_ShapeSys" for constrained and
               // "<channel>_<name>_shapeFactor" for free parameters.
               const std::string funcName =
                  ctx.channelName + "_" + mod.name + (constrained ? "_ShapeSys" : "_shapeFactor");
               elems.shapeElems.add(createPHF(tool, funcName, mod.name, mod.parameters, mod.vals, mod.constraint,
                                              ctx.constraints, ctx.observables, defaultShapeSysGammaMax,
                                              minShapeUncertainty, ctx.nBins));
            },
            [&](HS3::CustomModifier const &mod) {
               // The function can come later in the document.
               RooAbsReal *obj = tool.request<RooAbsReal>(mod.name, ctx.channelName);
               if (obj->dependsOn(ctx.observables)) {
                  elems.shapeElems.add(*obj);
               } else {
                  elems.normElems.add(*obj);
               }
            },
         },
         modifier);
   }

   if (!elems.overallNps.empty()) {
      std::string interpName = newName(sampleName + "_" + ctx.channelName + "_epsilon");
      elems.normElems.add(tool.wsEmplace<RooStats::HistFactory::FlexibleInterpVar>(
         interpName, elems.overallNps, 1., elems.overallLow, elems.overallHigh, elems.overallInterp));
   }
   if (!elems.histNps.empty()) {
      auto &v = tool.wsEmplace<PiecewiseInterpolation>(newName("histoSys_" + prefixedName), hf, elems.histoLo,
                                                       elems.histoHi, elems.histNps, elems.histoInterp);
      v.setPositiveDefinite();
      elems.shapeElems.add(v);
   } else {
      elems.shapeElems.add(hf);
   }

   tool.wsEmplace<RooProduct>(newName(prefixedName + "_shapes"), elems.shapeElems);
   if (!elems.normElems.empty()) {
      tool.wsEmplace<RooProduct>(newName(prefixedName + "_scaleFactors"), elems.normElems);
   } else {
      ws.factory("RooConstVar::" + newName(prefixedName + "_scaleFactors") + "(1.)");
   }
}

// The constraint of the gamma parameters of the staterror modifiers. Without one, the gamma parameters are free.
HS3::GammaConstraint statErrorConstraint(HS3::HistFactoryDist const &dist)
{
   // The constraint distributions named by a staterror modifier take precedence over a constraint type, and a type
   // given on a staterror modifier over the channel-level setting.
   HS3::GammaConstraint const *modifierConstraint = nullptr;
   for (auto const &sample : dist.samples) {
      for (auto const &modifier : sample.modifiers) {
         auto const *statError = std::get_if<HS3::StatError>(&modifier);
         if (!statError) {
            continue;
         }
         if (std::holds_alternative<HS3::GammaConstraintPdfs>(statError->constraint)) {
            return statError->constraint;
         }
         if (!modifierConstraint && !std::holds_alternative<HS3::Unconstrained>(statError->constraint)) {
            modifierConstraint = &statError->constraint;
         }
      }
   }
   return modifierConstraint ? *modifierConstraint : dist.statError.constraint;
}

// Reject constraints of staterror modifiers that make no sense for gamma parameters.
void checkStatErrorConstraints(HS3::HistFactoryDist const &dist, std::string const &name)
{
   for (auto const &sample : dist.samples) {
      for (auto const &modifier : sample.modifiers) {
         auto const *statError = std::get_if<HS3::StatError>(&modifier);
         auto const *type = statError ? std::get_if<HS3::ConstraintType>(&statError->constraint) : nullptr;
         if (type && *type == HS3::ConstraintType::LogNormal) {
            RooJSONFactoryWSTool::error(std::string("unsupported constraint '") + toString(*type) +
                                        "' of the staterror modifier in channel '" + name +
                                        "', supported are \"Poisson\" and \"Gauss\"");
         }
      }
   }
}

class HistFactoryImporter : public RooFit::JSONIO::Importer {
public:
   // This is only used for HistFactory distributions that don't come with a document.
   bool importArg(RooJSONFactoryWSTool *tool, const JSONNode &p) const override
   {
      RooFit::HS3::importHistFactoryNode(*tool, p, nullptr);
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
      const std::string context = "FlexibleInterpVar '" + std::string{fip->GetName()} + "'";
      const std::size_t nParameters = fip->variables().size();
      if (fip->low().size() != nParameters || fip->high().size() != nParameters ||
          fip->interpolationCodes().size() != nParameters) {
         RooJSONFactoryWSTool::error(context + " has non-matching parameter, variation, and interpolation lengths");
      }
      HS3::InterpolatedNumber out;
      for (RooAbsArg const *var : fip->variables()) {
         out.parameters.push_back(var->GetName());
      }
      out.nominal = fip->nominal();
      out.high = fip->high();
      out.low = fip->low();
      out.interpolations = toInterpolations(fip->interpolationCodes(), InterpolationClass::Flexible, context);
      HS3::writeInterpolatedNumber(elem, out);
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
      const std::string context = "PiecewiseInterpolation '" + std::string{pip->GetName()} + "'";
      const std::size_t nParameters = pip->paramList().size();
      if (pip->lowList().size() != nParameters || pip->highList().size() != nParameters ||
          pip->interpolationCodes().size() != nParameters) {
         RooJSONFactoryWSTool::error(context + " has non-matching parameter, variation, and interpolation lengths");
      }
      HS3::InterpolatedFunction out;
      out.nominal = pip->nominalHist()->GetName();
      for (std::size_t i = 0; i < nParameters; ++i) {
         out.parameters.push_back(HS3::toNameOrNumber(pip->paramList()[i]));
         out.high.push_back(HS3::toNameOrNumber(pip->highList()[i]));
         out.low.push_back(HS3::toNameOrNumber(pip->lowList()[i]));
      }
      out.interpolations = toInterpolations(pip->interpolationCodes(), InterpolationClass::Piecewise, context);
      out.positiveDefinite = pip->positiveDefinite();
      HS3::writeInterpolatedFunction(elem, out);
      return true;
   }
};

class PiecewiseInterpolationFactory : public RooFit::JSONIO::Importer {
public:
   bool importArg(RooJSONFactoryWSTool *tool, const JSONNode &p) const override
   {
      std::string name(RooJSONFactoryWSTool::name(p));
      HS3::Diagnostics diagnostics;
      const HS3::InterpolatedFunction in = HS3::parseInterpolatedFunction(p, diagnostics);
      report(name, diagnostics);

      auto request = [&](std::vector<HS3::NameOrNumber> const &refs) {
         RooArgList out;
         for (auto const &ref : refs) {
            auto const *value = std::get_if<double>(&ref);
            out.add(value ? RooFit::RooConst(*value) : *tool->request<RooAbsReal>(std::get<std::string>(ref), name));
         }
         return out;
      };
      RooArgList vars{request(in.parameters)};
      RooArgList low{request(in.low)};
      RooArgList high{request(in.high)};
      auto &pip = tool->wsEmplace<PiecewiseInterpolation>(
         name, *tool->request<RooAbsReal>(in.nominal, name), low, high, vars,
         toCodes(in.interpolations, InterpolationClass::Piecewise, "PiecewiseInterpolation '" + name + "'"));
      pip.setPositiveDefinite(in.positiveDefinite);
      return true;
   }
};

class FlexibleInterpVarFactory : public RooFit::JSONIO::Importer {
public:
   bool importArg(RooJSONFactoryWSTool *tool, const JSONNode &p) const override
   {
      std::string name(RooJSONFactoryWSTool::name(p));
      HS3::Diagnostics diagnostics;
      const HS3::InterpolatedNumber in = HS3::parseInterpolatedNumber(p, diagnostics);
      report(name, diagnostics);

      RooArgList vars;
      for (std::string const &var : in.parameters) {
         vars.add(*tool->request<RooRealVar>(var, name));
      }
      tool->wsEmplace<RooStats::HistFactory::FlexibleInterpVar>(
         name, vars, in.nominal, in.low, in.high,
         toCodes(in.interpolations, InterpolationClass::Flexible, "FlexibleInterpVar '" + name + "'"));
      return true;
   }
};

std::string stripOuterParens(const std::string &s)
{
   size_t start = 0;
   size_t end = s.size();

   while (start < end && s[start] == '(' && s[end - 1] == ')') {
      int depth = 0;
      bool balanced = true;
      for (size_t i = start; i < end - 1; ++i) {
         if (s[i] == '(') {
            ++depth;
         } else if (s[i] == ')') {
            --depth;
         }
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

/// Whether the `+` or `-` at position `i` of `expr` is a binary operator, i.e. not a sign and not part of the exponent
/// of a number like `1e-5`.
bool isBinaryOperator(const std::string &expr, std::size_t i)
{
   std::size_t j = i;
   while (j > 0 && std::isspace(static_cast<unsigned char>(expr[j - 1]))) {
      --j;
   }
   if (j == 0) {
      return false;
   }
   const char prev = expr[j - 1];
   if (prev == 'e' || prev == 'E') {
      // An exponent if the "e" directly follows the digits of a number, which don't start a name.
      std::size_t k = j - 1;
      while (k > 0 && (std::isdigit(static_cast<unsigned char>(expr[k - 1])) || expr[k - 1] == '.')) {
         --k;
      }
      const bool startsName = k > 0 && (std::isalpha(static_cast<unsigned char>(expr[k - 1])) || expr[k - 1] == '_');
      if (k < j - 1 && !startsName && i == j) {
         return false;
      }
   }
   return std::isalnum(static_cast<unsigned char>(prev)) || prev == '_' || prev == '.' || prev == ')' || prev == ']';
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
      } else if ((c == '+' || c == '-') && depth == 0 && isBinaryOperator(expr, i)) {
         return {}; // A sum or difference at the top level is not a product, e.g. "1 + 0.2 * k"
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

// A normsys that is written as the formula "1 +- a * b", where one of a and b is a constrained parameter.
struct FormulaNormSys {
   RooAbsReal *param = nullptr;
   RooAbsPdf *constraint = nullptr;
   double variation = 0.0;
};

FormulaNormSys parseOverallModifierFormula(const std::string &s, RooFormulaVar *formula)
{
   static const std::regex pattern(
      R"(^\s*1(?:\.0)?\s*([\+\-])\s*([a-zA-Z_][a-zA-Z0-9_]*|[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)\s*\*\s*([a-zA-Z_][a-zA-Z0-9_]*|[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)\s*$)");

   FormulaNormSys sys;
   std::smatch match;
   if (std::regex_match(s, match, pattern)) {
      const double sign = match[1].str() == "-" ? -1.0 : 1.0;

      std::string token2 = match[2].str();
      std::string token3 = match[4].str();

      RooAbsReal *p2 = static_cast<RooAbsReal *>(formula->getParameter(token2.c_str()));
      RooAbsReal *p3 = static_cast<RooAbsReal *>(formula->getParameter(token3.c_str()));
      RooRealVar *v2 = dynamic_cast<RooRealVar *>(p2);
      RooRealVar *v3 = dynamic_cast<RooRealVar *>(p3);

      auto *constr2 = findConstraint(v2);
      auto *constr3 = findConstraint(v3);

      if (constr2 && !p3) {
         sys.param = p2;
         sys.variation = sign * toDouble(token3);
      } else if (!p2 && constr3) {
         sys.param = p3;
         sys.variation = sign * toDouble(token2);
      } else if (constr2 && p3 && !constr3) {
         sys.param = v2;
         sys.variation = sign * p3->getVal();
      } else if (p2 && !constr2 && constr3) {
         sys.param = v3;
         sys.variation = sign * p2->getVal();
      }

      // A recognized formula has exactly one constrained parameter.
      sys.constraint = constr2 ? constr2 : constr3;
   }
   return sys;
}

void collectElements(RooArgList &elems, RooAbsArg *arg)
{
   if (auto prod = dynamic_cast<RooProduct *>(arg)) {
      for (const auto &e : prod->components()) {
         collectElements(elems, e);
      }
   } else {
      elems.add(*arg);
   }
}

bool allRooRealVar(const RooAbsCollection &list)
{
   for (auto *var : list) {
      if (!dynamic_cast<RooRealVar *>(var)) {
         return false;
      }
   }
   return true;
}

// The parameters of normsys and histosys modifiers must be variables.
void checkModifierParameter(RooAbsReal const &param, RooAbsPdf const *constraint)
{
   if (!constraint && !param.isConstant() && !dynamic_cast<RooRealVar const *>(&param)) {
      RooJSONFactoryWSTool::error("cannot find constraint for " + std::string(param.GetName()) +
                                  ": functions are not supported as parameters of this modifier");
   }
}

std::vector<double> contents(RooDataHist const &dataHist)
{
   return {dataHist.weightArray(), dataHist.weightArray() + dataHist.numEntries()};
}

// A channel as it is exported, and what the exporter needs to know about its RooFit objects.
struct ExportedChannel {
   std::string name;
   HS3::HistFactoryDist dist;
   RooArgSet const *observables = nullptr;
   /// The pdfs that the modifiers name as constraints, by name.
   std::map<std::string, RooAbsPdf const *> constraints;
   std::vector<RooAbsReal *> customModifiers;
   /// Parts of formulas that are exported as custom modifiers. They are created for the export.
   std::vector<std::unique_ptr<RooAbsReal>> temporaries;
   // To reconstruct the statistical uncertainties, by bin starting at one.
   std::map<int, double> totYield;
   std::map<int, double> totYield2;
   std::map<int, double> relErrors;
   // Constant inverse bin widths of HistFactory before ROOT 6.26, which histfactory_dist implies.
   std::set<RooAbsArg const *> legacyBinWidthFactors;
};

/// The volume of the first bin and the average bin volume of every histogram of a channel. HistFactory before ROOT
/// 6.26 took the inverse bin width as the number of bins over the range of each observable.
std::vector<double> binVolumes(RooRealSumPdf const &sumpdf)
{
   RooArgSet nodes;
   sumpdf.treeNodeServerList(&nodes);
   std::vector<double> out;
   for (RooAbsArg *node : nodes) {
      auto *hf = dynamic_cast<RooHistFunc const *>(node);
      if (!hf || hf->dataHist().numEntries() == 0) {
         continue;
      }
      out.push_back(hf->dataHist().binVolume(0));
      double average = 1.;
      for (RooAbsArg *obs : *hf->dataHist().get()) {
         if (auto *real = dynamic_cast<RooRealVar const *>(obs)) {
            average *= (real->getMax() - real->getMin()) / real->numBins();
         }
      }
      out.push_back(average);
   }
   return out;
}

/// Whether a constant factor of a sample is the inverse bin width that HistFactory before ROOT 6.26 multiplied the
/// samples with (a constant `RooRealVar` `binWidth_<observables>_<i>` in the coefficients of the `RooRealSumPdf`).
/// `histfactory_dist` implies this factor, so it is not written as a modifier. It is identified by its structure, not
/// by its name: a constant that is not floating in any snapshot, used only by the sum or the products of the channel,
/// with the inverse of the volume of the first bin, or of the average bin volume, of one of the histograms of the
/// channel as value.
bool isLegacyBinWidthFactor(RooAbsArg const &factor, RooRealSumPdf const &sumpdf, RooArgSet const &channelNodes,
                            std::vector<double> const &volumes, RooWorkspace const *ws)
{
   if (!RooJSONFactoryWSTool::config().identifyLegacyBinWidthFactors) {
      return false;
   }
   auto const *var = dynamic_cast<RooRealVar const *>(&factor);
   if (!dynamic_cast<RooConstVar const *>(&factor) && !(var && var->isConstant())) {
      return false;
   }
   if (var && ws) {
      for (auto *snapshot : static_range_cast<RooArgSet const *>(ws->getSnapshots())) {
         auto const *inSnapshot = dynamic_cast<RooRealVar const *>(snapshot->find(*var));
         if (inSnapshot && !inSnapshot->isConstant()) {
            return false;
         }
      }
   }
   for (RooAbsArg const *client : factor.clients()) {
      const bool isChannelProduct = dynamic_cast<RooProduct const *>(client) && channelNodes.find(*client) == client;
      if (client != &sumpdf && !isChannelProduct) {
         return false;
      }
   }
   // The value was stored with 7 significant digits.
   const double value = static_cast<RooAbsReal const &>(factor).getVal();
   return std::any_of(volumes.begin(), volumes.end(),
                      [&](double volume) { return volume > 0. && std::abs(value * volume - 1.) < 1e-6; });
}

ExportedChannel readChannel(RooJSONFactoryWSTool *tool, const std::string &pdfname, const RooRealSumPdf *sumpdf)
{
   ExportedChannel channel;

   RooWorkspace *ws = tool->workspace();

   channel.name = pdfname;
   erasePrefix(channel.name, "model_");
   eraseSuffix(channel.name, "_model");

   RooArgSet channelNodes;
   sumpdf->treeNodeServerList(&channelNodes);
   const std::vector<double> volumes = binVolumes(*sumpdf);

   auto constraint = [&](RooAbsPdf const *pdf) -> HS3::Constraint {
      if (!pdf) {
         return HS3::Unconstrained{};
      }
      channel.constraints.emplace(pdf->GetName(), pdf);
      return HS3::ConstraintPdf{{pdf->GetName()}};
   };
   auto gammaConstraint = [&](RooAbsPdf const *pdf) -> std::optional<HS3::DistributionRef> {
      if (!pdf) {
         return std::nullopt;
      }
      channel.constraints.emplace(pdf->GetName(), pdf);
      return HS3::DistributionRef{pdf->GetName()};
   };

   for (size_t sampleidx = 0; sampleidx < sumpdf->funcList().size(); ++sampleidx) {
      PiecewiseInterpolation *pip = nullptr;
      std::vector<ParamHistFunc *> phfs;

      const auto func = sumpdf->funcList().at(sampleidx);
      HS3::Sample sample;
      sample.name = func->GetName();
      erasePrefix(sample.name, "L_x_");
      eraseSuffix(sample.name, "_shapes");
      eraseSuffix(sample.name, "_" + channel.name);
      erasePrefix(sample.name, pdfname + "_");

      // The modifiers by type, which are written in this order.
      std::vector<HS3::NormFactor> normFactors;
      std::vector<HS3::NormSys> normSys;
      std::vector<HS3::HistoSys> histoSys;
      std::vector<HS3::ShapeSys> shapeSys;
      std::vector<HS3::CustomModifier> custom;
      std::vector<HS3::CustomModifier> temporaries;
      std::optional<HS3::StatError> statError;

      auto addNormSys = [&](RooAbsReal &param, double hi, double lo, HS3::Interpolation interpolation,
                            RooAbsPdf const *pdf) {
         checkModifierParameter(param, pdf);
         std::string sysname(param.GetName());
         erasePrefix(sysname, "alpha_");
         normSys.push_back({sysname, {param.GetName()}, hi, lo, constraint(pdf), interpolation});
      };

      auto updateObservables = [&](RooDataHist const &dataHist) {
         if (channel.observables == nullptr) {
            channel.observables = dataHist.get();
         }
         if (sample.contents.empty()) {
            sample.contents = contents(dataHist);
         }
      };
      auto processElements = [&](const auto &elements, auto &&self) -> void {
         for (RooAbsArg *e : elements) {
            if (auto *binWidth = dynamic_cast<RooBinWidthFunction *>(e); binWidth && binWidth->divideByBinWidth()) {
               // The division by the bin width is implied by histfactory_dist.
            } else if (isLegacyBinWidthFactor(*e, *sumpdf, channelNodes, volumes, ws)) {
               // Before ROOT 6.26, HistFactory multiplied the samples with a constant inverse bin width instead.
               channel.legacyBinWidthFactors.insert(e);
               if (!TString(e->GetName()).Contains("binWidth")) {
                  oocoutW(nullptr, ObjectHandling)
                     << "RooFitHS3: the constant factor '" << e->GetName() << "' of sample '" << sample.name
                     << "' in channel '" << channel.name
                     << "' is used like the inverse bin width of HistFactory before ROOT 6.26 (its value is the "
                        "inverse bin volume), so it is not written: histfactory_dist implies it. Set "
                        "RooJSONFactoryWSTool::config().identifyLegacyBinWidthFactors = false to write such factors "
                        "as normfactors."
                     << std::endl;
               }
            } else if (auto constVar = dynamic_cast<RooConstVar *>(e)) {
               if (constVar->getVal() != 1.) {
                  normFactors.push_back({constVar->GetName(), {constVar->GetName()}, HS3::Unconstrained{}});
               }
            } else if (auto par = dynamic_cast<RooRealVar *>(e)) {
               // Every Gaussian of the parameter is a candidate for its constraint.
               bool isConstrained = false;
               for (RooAbsArg const *pdf : ws->allPdfs()) {
                  if (auto gauss = dynamic_cast<RooGaussian const *>(pdf)) {
                     if (std::string(par->GetName()) == gauss->getX().GetName()) {
                        normFactors.push_back({par->GetName(), {par->GetName()}, constraint(gauss)});
                        isConstrained = true;
                     }
                  }
               }
               if (!isConstrained) {
                  normFactors.push_back({par->GetName(), {par->GetName()}, HS3::Unconstrained{}});
               }
            } else if (auto hf = dynamic_cast<const RooHistFunc *>(e)) {
               updateObservables(hf->dataHist());
            } else if (ParamHistFunc *phf = dynamic_cast<ParamHistFunc *>(e); phf && allRooRealVar(phf->paramList())) {
               phfs.push_back(phf);
            } else if (auto fip = dynamic_cast<RooStats::HistFactory::FlexibleInterpVar *>(e)) {
               // some (modified) histfactory models have several instances of FlexibleInterpVar
               // we collect and merge them
               for (size_t i = 0; i < fip->variables().size(); ++i) {
                  RooAbsReal *var = static_cast<RooAbsReal *>(fip->variables().at(i));
                  std::string sysname(var->GetName());
                  erasePrefix(sysname, "alpha_");
                  const std::string context = "normsys modifier '" + sysname + "' in sample '" + sample.name +
                                              "' of channel '" + channel.name + "'";
                  // A parameter without constraint is unconstrained, see checkUnconstrainedModifiers().
                  addNormSys(*var, fip->high()[i], fip->low()[i],
                             interpolationFromCode(fip->interpolationCodes()[i], InterpolationClass::Flexible, context),
                             findConstraint(var));
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
               if (components.empty()) {
                  // it's not a product, let's just treat it as an unknown element
                  custom.push_back({formula->GetName()});
                  channel.customModifiers.push_back(formula);
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
                     FormulaNormSys sys = parseOverallModifierFormula(comp, formula);
                     if (sys.param) {
                        // Preserve the legacy export behaviour for recognized explicit formulae.
                        addNormSys(*sys.param, 1.0 + sys.variation, 1.0 - sys.variation,
                                   multiplicativePiecewiseExponential, sys.constraint);
                        continue;
                     }

                     // this is something non-trivial, let's deal with it separately
                     std::string name = std::string(formula->GetName()) + "_part" + std::to_string(idx);
                     ++idx;
                     temporaries.push_back({name});
                     channel.temporaries.push_back(std::make_unique<RooFormulaVar>(
                        name.c_str(), name.c_str(), comp.c_str(), formula->dependents()));
                  }
                  self(realComponents, self);
               }
            } else if (auto real = dynamic_cast<RooAbsReal *>(e)) {
               custom.push_back({real->GetName()});
               channel.customModifiers.push_back(real);
            }
         }
      };

      RooArgList elems;
      collectElements(elems, func);
      collectElements(elems, sumpdf->coefList().at(sampleidx));
      processElements(elems, processElements);

      // see if we can get the observables
      if (pip) {
         if (auto nh = dynamic_cast<RooHistFunc const *>(pip->nominalHist())) {
            updateObservables(nh->dataHist());
         }
      }

      if (pip) {
         for (size_t i = 0; i < pip->paramList().size(); ++i) {
            RooAbsReal *var = static_cast<RooAbsReal *>(pip->paramList().at(i));
            std::string sysname(var->GetName());
            erasePrefix(sysname, "alpha_");
            if (auto lo = dynamic_cast<RooHistFunc *>(pip->lowList().at(i))) {
               if (auto hi = dynamic_cast<RooHistFunc *>(pip->highList().at(i))) {
                  // A parameter without constraint is unconstrained, see checkUnconstrainedModifiers().
                  const auto *pdf = findConstraint(var);
                  checkModifierParameter(*var, pdf);
                  const std::string context = "histosys modifier '" + sysname + "' in sample '" + sample.name +
                                              "' of channel '" + channel.name + "'";
                  histoSys.push_back(
                     {sysname,
                      {var->GetName()},
                      {contents(hi->dataHist()), std::nullopt},
                      {contents(lo->dataHist()), std::nullopt},
                      constraint(pdf),
                      interpolationFromCode(pip->interpolationCodes()[i], InterpolationClass::Piecewise, context)});
               }
            }
         }
      }

      for (ParamHistFunc *phf : phfs) {
         if (isStatError(*phf, phfs)) { // MC stat uncertainty
            statError.emplace();
            statError->name = "staterror";
            HS3::GammaConstraintPdfs pdfs;
            int idx = 0;
            for (const auto &g : phf->paramList()) {
               statError->parameters.push_back({g->GetName()});
               ++idx;
               RooAbsPdf *pdf = findConstraint(g);
               pdfs.pdfs.push_back(gammaConstraint(pdf));
               channel.totYield[idx] += sample.contents[idx - 1];
               channel.totYield2[idx] += (sample.contents[idx - 1] * sample.contents[idx - 1]);
               if (pdf) {
                  channel.relErrors[idx] = constraintRelError(*pdf, *g);
               }
            }
            statError->constraint = std::move(pdfs);
         } else { // ShapeSys or ShapeFactor
            HS3::ShapeSys &sys = shapeSys.emplace_back();
            sys.name = phf->GetName();
            erasePrefix(sys.name, channel.name + "_");
            for (std::string_view suffix : {"_ShapeSys", "_shapeSys", "_ShapeFactor", "_shapeFactor"}) {
               if (eraseSuffix(sys.name, suffix))
                  break;
            }

            // Both are written as "shapefactor": the constrained parameters name their constraint distribution in
            // "constraints", the others are free.
            HS3::GammaConstraintPdfs pdfs;
            for (const auto &g : phf->paramList()) {
               sys.parameters.push_back({g->GetName()});
               RooAbsPdf *pdf = findConstraint(g);
               pdfs.pdfs.push_back(gammaConstraint(pdf));
            }
            sys.constraint = std::move(pdfs);
         }
      }

      sortByName(normFactors);
      sortByName(normSys);
      sortByName(histoSys);
      sortByName(shapeSys);
      auto append = [&](auto &modifiers) {
         sample.modifiers.insert(sample.modifiers.end(), std::make_move_iterator(modifiers.begin()),
                                 std::make_move_iterator(modifiers.end()));
      };
      append(normFactors);
      append(normSys);
      append(histoSys);
      append(shapeSys);
      append(custom);
      append(temporaries);
      if (statError) {
         sample.modifiers.push_back(std::move(*statError));
      }

      channel.dist.samples.push_back(std::move(sample));
   }

   sortByName(channel.dist.samples);
   return channel;
}

// The statistical uncertainties of the samples are reconstructed from the constraints of the gamma parameters, with
// the (arbitrary) convention that all samples have the same relative uncertainty.
void configureStatError(ExportedChannel &channel)
{
   for (HS3::Sample &sample : channel.dist.samples) {
      if (std::none_of(sample.modifiers.begin(), sample.modifiers.end(),
                       [](auto const &m) { return std::holds_alternative<HS3::StatError>(m); })) {
         continue;
      }
      std::vector<double> errors(sample.contents.size());
      for (auto const &bin : channel.relErrors) {
         const int i = bin.first;
         // this reconstruction is inherently imprecise, so we truncate it at some decimal places to make sure that
         // we don't carry around too many useless digits
         errors[i - 1] =
            round_prec(bin.second * channel.totYield[i] / std::sqrt(channel.totYield2[i]) * sample.contents[i - 1], 7);
      }
      sample.errors = std::move(errors);
   }
}

// The modifiers name only the constraint terms of the channel. A pdf of a parameter that is not, like an external
// constraint or a prior, stays where it is. Returns the named constraint terms.
std::set<RooAbsPdf const *>
nameChannelConstraints(ExportedChannel &channel, std::vector<RooAbsPdf *> const &constraints)
{
   std::set<RooAbsPdf const *> named;
   auto isNamed = [&](std::string const &name) {
      RooAbsPdf const *pdf = channel.constraints.at(name);
      if (std::find(constraints.begin(), constraints.end(), pdf) == constraints.end()) {
         return false;
      }
      named.insert(pdf);
      return true;
   };
   auto filter = [&](HS3::Constraint &constraint) {
      auto const *pdf = std::get_if<HS3::ConstraintPdf>(&constraint);
      if (pdf && !isNamed(pdf->pdf.name)) {
         constraint = HS3::Unconstrained{};
      }
   };
   auto filterGammas = [&](HS3::GammaConstraint &constraint) {
      auto *pdfs = std::get_if<HS3::GammaConstraintPdfs>(&constraint);
      if (!pdfs) {
         return;
      }
      bool any = false;
      for (auto &pdf : pdfs->pdfs) {
         if (pdf && !isNamed(pdf->name)) {
            pdf.reset();
         }
         any = any || pdf;
      }
      // Nothing is written if all parameters are unconstrained.
      if (!any) {
         constraint = HS3::Unconstrained{};
      }
   };
   for (HS3::Sample &sample : channel.dist.samples) {
      for (HS3::Modifier &modifier : sample.modifiers) {
         std::visit(HS3::overloaded{
                       [&](HS3::NormFactor &m) { filter(m.constraint); },
                       [&](HS3::NormSys &m) { filter(m.constraint); }, [&](HS3::HistoSys &m) { filter(m.constraint); },
                       [&](HS3::ShapeSys &m) { filterGammas(m.constraint); },
                       [&](HS3::StatError &m) { filterGammas(m.constraint); }, [](HS3::CustomModifier &) {}},
                    modifier);
      }
   }
   return named;
}

bool tryExportHistFactory(RooJSONFactoryWSTool *tool, const std::string &pdfname, const RooRealSumPdf *sumpdf,
                          std::vector<RooAbsPdf *> constraints)
{
   // some preliminary checks
   if (!sumpdf) {
      return false;
   }

   for (RooAbsArg *sample : sumpdf->funcList()) {
      if (!dynamic_cast<RooProduct *>(sample) && !dynamic_cast<RooRealSumPdf *>(sample)) {
         return false;
      }
   }

   ExportedChannel channel = readChannel(tool, pdfname, sumpdf);
   HS3::HistFactoryDist &dist = channel.dist;

   // sanity checks
   if (dist.samples.empty())
      return false;
   for (auto &sample : dist.samples) {
      if (sample.contents.empty()) {
         return false;
      }
   }

   HS3::Diagnostics diagnostics;
   HS3::mergeDuplicateModifiers(dist, channel.name, diagnostics);
   HS3::report(diagnostics);
   configureStatError(channel);
   HS3::factorOutDefaultInterpolation(dist);

   for (auto *obs : static_range_cast<RooRealVar *>(*channel.observables)) {
      dist.axes.push_back(HS3::toBinnedAxis(*obs));
   }

   std::set<RooAbsPdf const *> namedConstraints = nameChannelConstraints(channel, constraints);
   for (RooAbsPdf const *pdf : namedConstraints) {
      tool->queueExport(*pdf);
   }

   for (RooAbsReal *modifier : channel.customModifiers) {
      tool->queueExport(*modifier);
   }
   for (auto &modifier : channel.temporaries) {
      tool->queueExportTemporary(modifier.release());
   }

   // Export all model parameters
   RooArgSet parameters;
   sumpdf->getParameters(channel.observables, parameters);
   for (RooAbsArg *param : parameters) {
      // The global observables are not among them, since they only enter the constraint terms. The legacy inverse bin
      // widths are not written at all.
      if (!channel.legacyBinWidthFactors.count(param)) {
         tool->queueExport(*param);
      }
   }

   // All other constraint terms of the channel are written separately.
   std::vector<HS3::DistributionRef> channelConstraintNames;
   for (RooAbsPdf const *pdf : constraints) {
      if (!namedConstraints.count(pdf)) {
         tool->queueExport(*pdf);
         channelConstraintNames.push_back({pdf->GetName()});
      }
   }

   HS3::setExportedObject(*tool, std::move(dist), std::move(channelConstraintNames));
   return true;
}

class HistFactoryStreamer_ProdPdf : public RooFit::JSONIO::Exporter {
public:
   bool autoExportDependants() const override { return false; }
   bool tryExport(RooJSONFactoryWSTool *tool, const RooProdPdf *prodpdf) const
   {
      std::vector<RooAbsPdf *> constraints;
      RooRealSumPdf *sumpdf = nullptr;
      for (auto *pdf : static_range_cast<RooAbsPdf *>(prodpdf->pdfList())) {
         auto thispdf = dynamic_cast<RooRealSumPdf *>(pdf);
         if (thispdf) {
            if (!sumpdf) {
               sumpdf = thispdf;
            } else {
               return false;
            }
         } else {
            constraints.push_back(pdf);
         }
      }
      if (!sumpdf)
         return false;

      return tryExportHistFactory(tool, prodpdf->GetName(), sumpdf, constraints);
   }
   std::string const &key() const override
   {
      static const std::string keystring = "histfactory_dist";
      return keystring;
   }
   bool exportObject(RooJSONFactoryWSTool *tool, const RooAbsArg *p, JSONNode &) const override
   {
      return tryExport(tool, static_cast<const RooProdPdf *>(p));
   }
};

class HistFactoryStreamer_SumPdf : public RooFit::JSONIO::Exporter {
public:
   bool autoExportDependants() const override { return false; }
   bool tryExport(RooJSONFactoryWSTool *tool, const RooRealSumPdf *sumpdf) const
   {
      std::vector<RooAbsPdf *> constraints;
      return tryExportHistFactory(tool, sumpdf->GetName(), sumpdf, constraints);
   }
   std::string const &key() const override
   {
      static const std::string keystring = "histfactory_dist";
      return keystring;
   }
   bool exportObject(RooJSONFactoryWSTool *tool, const RooAbsArg *p, JSONNode &) const override
   {
      return tryExport(tool, static_cast<const RooRealSumPdf *>(p));
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

void RooFit::HS3::buildHistFactoryDist(RooJSONFactoryWSTool &tool, std::string const &name, HistFactoryDist const &dist,
                                       std::vector<DistributionRef> const *channelConstraints)
{
   checkStatErrorConstraints(dist, name);

   RooArgSet observables = createObservables(dist.axes);

   std::vector<std::unique_ptr<RooDataHist>> data;
   data.reserve(dist.samples.size());
   std::vector<double> sumW;
   std::vector<double> sumW2;
   std::vector<ParameterRef> gammaParnames;
   for (auto const &sample : dist.samples) {
      auto dh = createDataHist(name + "_" + sample.name + "_dataHist", observables, sample.contents,
                               sample.errors ? &*sample.errors : nullptr);
      const std::size_t nbins = dh->numEntries();

      auto found = std::find_if(sample.modifiers.begin(), sample.modifiers.end(),
                                [](auto const &m) { return std::holds_alternative<StatError>(m); });
      if (found != sample.modifiers.end()) {
         if (sumW.empty()) {
            sumW.resize(nbins);
            sumW2.resize(nbins);
         }
         // A sample without "errors" contributes no statistical uncertainty, unless Poisson uncertainties are implied.
         const bool hasErrors = sample.errors || dist.statError.impliedPoissonErrors;
         for (std::size_t i = 0; i < nbins; ++i) {
            sumW[i] += dh->weight(i);
            sumW2[i] += hasErrors ? dh->weightSquared(i) : 0.;
         }
         if (gammaParnames.empty()) {
            gammaParnames = std::get<StatError>(*found).parameters;
         }
      }
      data.emplace_back(std::move(dh));
   }

   std::string channelName = name;
   erasePrefix(channelName, "model_");

   std::size_t nBins = 1;
   for (auto const &axis : dist.axes) {
      nBins *= std::visit(overloaded{[](RegularBinning const &b) { return static_cast<std::size_t>(b.nbins); },
                                     [](EdgesBinning const &b) { return b.edges.size() - 1; }},
                          axis.binning);
   }

   RooAbsArg *mcStatObject = nullptr;
   RooArgSet constraints;
   if (!sumW.empty()) {
      std::vector<double> errs(sumW.size());
      for (std::size_t i = 0; i < sumW.size(); ++i) {
         if (sumW[i] == 0.) {
            errs[i] = 0.;
            continue;
         }
         errs[i] = std::sqrt(sumW2[i]) / sumW[i];
         // avoid negative sigma. This NP will be set constant anyway later
         errs[i] = std::max(errs[i], 0.);
      }

      mcStatObject = &createPHF(tool, "mc_stat_" + channelName, "stat_" + channelName, gammaParnames, errs,
                                statErrorConstraint(dist), constraints, observables, defaultStatErrorGammaMax,
                                dist.statError.relThreshold, nBins);
      mcStatObject->setAttribute(statErrorAttribute);
   }

   ChannelContext ctx{tool, dist, observables, name, channelName, nBins, mcStatObject, constraints};

   RooArgList funcs;
   RooArgList coefs;
   for (std::size_t i = 0; i < dist.samples.size(); ++i) {
      HS3::Sample const &sample = dist.samples[i];
      buildSample(ctx, sample, *data[i]);
      funcs.add(*tool.request<RooAbsReal>(name + "_" + sample.name + "_shapes", name));
      coefs.add(*tool.request<RooAbsReal>(name + "_" + sample.name + "_scaleFactors", name));
   }

   if (channelConstraints) {
      for (auto const &ref : *channelConstraints) {
         RooAbsPdf *pdf = findNamedConstraint(tool, ref.name, name);
         if (!pdf) {
            RooJSONFactoryWSTool::error("unable to find the constraint '" + ref.name + "' of '" + name + "'");
         }
         constraints.add(*pdf);
      }
   }

   // A histfactory_dist is a product of Poisson distributions of the bin counts. RooFit's binned likelihood
   // computes exactly that (it differs from the event-wise likelihood only by a constant), and binned generation
   // generates bin counts, so both are enabled, as HistFactory does by default.
   auto enableBinned = [](RooRealSumPdf &sum) {
      sum.setAttribute("BinnedLikelihood");
      sum.setAttribute("GenerateBinned");
   };
   if (constraints.empty()) {
      enableBinned(tool.wsEmplace<RooRealSumPdf>(name, funcs, coefs, true));
   } else {
      std::string sumName = name + "_model";
      erasePrefix(sumName, "model_");
      auto &sum = tool.wsEmplace<RooRealSumPdf>(sumName, funcs, coefs, true);
      sum.SetTitle(name.c_str());
      enableBinned(sum);
      tool.wsEmplace<RooProdPdf>(name, constraints, RooFit::Conditional(sum, observables));
   }
}

bool RooFit::HS3::isBuiltinHistFactoryImporter(RooFit::JSONIO::Importer const &importer)
{
   return dynamic_cast<HistFactoryImporter const *>(&importer) != nullptr;
}

// The upgrade rewrites the JSON, so it has to work on a copy.
void RooFit::HS3::importHistFactoryNode(RooJSONFactoryWSTool &tool, JSONNode const &p,
                                        std::vector<DistributionRef> const *channelConstraints)
{
   std::string name = RooJSONFactoryWSTool::name(p);
   if (!p.has_child("samples")) {
      RooJSONFactoryWSTool::error("no samples in '" + name + "', skipping.");
   }
   std::stringstream ss;
   p.writeJSON(ss);
   std::unique_ptr<RooFit::Detail::JSONTree> tree = RooFit::Detail::JSONTree::create(ss.str());

   HS3::Version version;
   HS3::parseVersion(tool.hs3VersionInput(), version);

   HS3::Diagnostics diagnostics;
   HS3::upgradeHistFactoryDist(tree->rootnode(), version, diagnostics);
   HS3::HistFactoryDist dist = HS3::parseHistFactoryDist(tree->rootnode(), diagnostics);
   if (!HS3::hasErrors(diagnostics)) {
      HS3::validateHistFactoryDist(dist, name, diagnostics);
   }
   HS3::report(diagnostics);
   HS3::buildHistFactoryDist(tool, name, dist, channelConstraints);
}
