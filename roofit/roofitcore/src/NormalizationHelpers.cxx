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

/// Mark `arg` and every branch node reachable through its server tree as
/// already compiled. Use this after assembling or cloning a sub-graph
/// yourself inside `compileForNormSet`: it prevents a follow-up
/// `compileServers` call from re-cloning any of those internal nodes, while
/// still letting the recursive descent reach the genuine leaves
/// (fundamental observables and parameters) at the bottom of the tree.
void RooFit::Detail::CompileContext::markSubtreeAsCompiled(RooAbsArg &arg) const
{
   RooArgSet branches;
   arg.branchNodeServerList(&branches);
   for (RooAbsArg *b : branches) {
      markAsCompiled(*b);
   }
}

bool RooFit::Detail::CompileContext::isMarkedAsCompiled(RooAbsArg const &arg) const
{
   return arg.getAttribute("_COMPILED");
}

/// Replace any reference to the original computation graph that is left in the
/// compiled one. Objects that are created while compiling, like the
/// normalization integrals, are built from the original normalization set, so
/// they can still reference the original observables. Such a leftover is a
/// second node with the same name in the compiled graph, which breaks the
/// lookup of nodes by name later on: the dataset columns would potentially be
/// attached to the leftover instead of the compiled observable, in which case
/// the compiled function doesn't depend on the data at all anymore.
void RooFit::Detail::CompileContext::redirectToCompiledServers(RooAbsArg &topNode) const
{
   RooArgList nodes;
   topNode.treeNodeServerList(&nodes, nullptr, /*doBranch=*/true, /*doLeaf=*/true, /*valueOnly=*/false,
                              /*recurseNonDerived=*/false);
   for (RooAbsArg *node : nodes) {
      node->redirectServers(_replacements);
   }
}
