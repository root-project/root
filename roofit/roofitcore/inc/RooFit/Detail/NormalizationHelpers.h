/*
 * Project: RooFit
 * Authors:
 *   Jonas Rembser, CERN 2022
 *
 * Copyright (c) 2022, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#ifndef RooFit_Detail_NormalizationHelpers_h
#define RooFit_Detail_NormalizationHelpers_h

#include <memory>
#include <string>
#include <unordered_map>

class RooAbsArg;
class RooArgSet;

class TNamed;

namespace RooFit {

namespace Detail {

class CompileContext {
public:
   CompileContext(RooArgSet const &topLevelNormSet);

   ~CompileContext();

   template <class T>
   T *compile(T &arg, RooAbsArg &owner, RooArgSet const &normSet)
   {
      return static_cast<T *>(compileImpl(arg, owner, normSet));
   }

   void compileServers(RooAbsArg &arg, RooArgSet const &normSet);
   void compileServer(RooAbsArg &server, RooAbsArg &arg, RooArgSet const &normSet);

   void markAsCompiled(RooAbsArg &arg) const;

   // This information is used for the binned likelihood optimization.
   void setLikelihoodMode(bool flag) { _likelihoodMode = flag; }
   bool likelihoodMode() const { return _likelihoodMode; }
   void setBinnedLikelihoodMode(bool flag) { _binnedLikelihoodMode = flag; }
   bool binnedLikelihoodMode() const { return _binnedLikelihoodMode; }
   void setBinWidthFuncFlag(bool flag) { _binWidthFuncFlag = flag; }
   bool binWidthFuncFlag() const { return _binWidthFuncFlag; }

private:
   RooAbsArg *compileImpl(RooAbsArg &arg, RooAbsArg &owner, RooArgSet const &normSet);
   void add(RooAbsArg &arg);
   RooAbsArg *find(RooAbsArg &arg) const;
   bool isMarkedAsCompiled(RooAbsArg const &arg) const;

   RooArgSet const &_topLevelNormSet;
   std::unordered_map<TNamed const *, RooAbsArg *> _clonedArgsSet;
   std::unordered_map<RooAbsArg *, RooAbsArg *> _replacements;

   bool _likelihoodMode = false;
   bool _binnedLikelihoodMode = false;
   bool _binWidthFuncFlag = false;
};

template <class T>
std::unique_ptr<T> compileForNormSet(T const &arg, RooArgSet const &normSet)
{
   RooFit::Detail::CompileContext ctx{normSet};
   return std::unique_ptr<T>{static_cast<T *>(arg.compileForNormSet(normSet, ctx).release())};
}

} // namespace Detail

} // namespace RooFit

#endif
