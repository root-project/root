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

#include <RooGlobalFunc.h>
#include <RooWorkspace.h>

namespace RooStats {
namespace HistFactory {
namespace Detail {

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

} // namespace Detail
} // namespace HistFactory
} // namespace RooStats

#endif
