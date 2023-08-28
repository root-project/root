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

#include <RooFit/Detail/NormalizationHelpers.h>

#include <RooAbsArg.h>
#include <RooArgList.h>
#include <RooArgSet.h>

#include <TNamed.h>

RooFit::Detail::CompileContext::CompileContext(RooArgSet const &topLevelNormSet) : _topLevelNormSet{topLevelNormSet} {}

RooFit::Detail::CompileContext::~CompileContext() {}

void RooFit::Detail::CompileContext::add(RooAbsArg &arg)
{
   _clonedArgsSet.emplace(arg.namePtr(), &arg);
}

RooAbsArg *RooFit::Detail::CompileContext::find(RooAbsArg &arg) const
{
   auto existingServerClone = _clonedArgsSet.find(arg.namePtr());
   if (existingServerClone != _clonedArgsSet.end()) {
      return existingServerClone->second;
   }
   return nullptr;
}

void RooFit::Detail::CompileContext::compileServers(RooAbsArg &arg, RooArgSet const &normSet)
{
   for (RooAbsArg *server : arg.servers()) {
      this->compile(*server, arg, normSet);
   }
   arg.redirectServers(_replacements);
}

void RooFit::Detail::CompileContext::compileServer(RooAbsArg &server, RooAbsArg &arg, RooArgSet const &normSet)
{
   this->compile(server, arg, normSet);
   arg.redirectServers(_replacements);
}

RooAbsArg *RooFit::Detail::CompileContext::compileImpl(RooAbsArg &arg, RooAbsArg &owner, RooArgSet const &normSet)
{
   if (auto existingServerClone = this->find(arg)) {
      return existingServerClone;
   }
   if (arg.isFundamental() && !_topLevelNormSet.find(arg)) {
      return nullptr;
   }
   if (isMarkedAsCompiled(arg)) {
      return nullptr;
   }

   std::unique_ptr<RooAbsArg> newArg = arg.compileForNormSet(normSet, *this);
   markAsCompiled(*newArg);
   _replacements[&arg] = newArg.get();
   this->add(*newArg);
   RooAbsArg *out = newArg.get();
   owner.addOwnedComponents(std::move(newArg));
   return out;
}

void RooFit::Detail::CompileContext::markAsCompiled(RooAbsArg &arg) const
{
   arg.setAttribute("_COMPILED");
}

bool RooFit::Detail::CompileContext::isMarkedAsCompiled(RooAbsArg const &arg) const
{
   return arg.getAttribute("_COMPILED");
}
