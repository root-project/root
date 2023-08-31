/*
 * Project: RooFit
 * Authors:
 *   Jonas Rembser, CERN 2023
 *
 * Copyright (c) 2023, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#ifndef HistFactoryImplHelpers_h
#define HistFactoryImplHelpers_h

#include <RooStats/HistFactory/Systematics.h>

#include <RooGlobalFunc.h>
#include <RooWorkspace.h>

#include <ROOT/RSpan.hxx>

namespace RooStats {
namespace HistFactory {
namespace Detail {

namespace MagicConstants {

constexpr double defaultGammaMin = 0;
constexpr double defaultShapeFactorGammaMax = 1000;
constexpr double defaultShapeSysGammaMax = 10;
constexpr double defaultStatErrorGammaMax = 10;
constexpr double minShapeUncertainty = 0.0;

} // namespace MagicConstants

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

void configureConstrainedGammas(RooArgList const &gammas, std::span<const double> relSigmas, double minSigma);

struct CreateGammaConstraintsOutput {
   std::vector<std::unique_ptr<RooAbsPdf>> constraints;
   std::vector<RooRealVar*> globalObservables;
};

CreateGammaConstraintsOutput createGammaConstraints(RooArgList const &paramList,
                                                    std::span<const double> relSigmas, double minSigma,
                                                    Constraint::Type type);

} // namespace Detail
} // namespace HistFactory
} // namespace RooStats

#endif
