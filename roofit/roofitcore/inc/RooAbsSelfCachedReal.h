/*
 * Project: RooFit
 *
 * Copyright (c) 2023, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#ifndef RooFit_RooAbsSelfCachedReal_h
#define RooFit_RooAbsSelfCachedReal_h

/**
\file RooAbsSelfCachedReal.h
\class RooAbsSelfCached
\ingroup Roofitcore

Abstract base class for functions whose
output is cached in terms of a histogram in all observables between
getVal() and evaluate(). For certain p.d.f.s that are very
expensive to calculate it may be beneficial to implement them as a
RooAbsSelfCached rather than a RooAbsReal/Pdf. Class
RooAbsSelfCached is designed to have its interface identical to
that of RooAbsReal/Pdf, so any p.d.f can make use of its caching
functionality by merely switching its base class.  Existing
RooAbsReal/Pdf objects can also be cached a posteriori with the
RooCachedReal/Pdf wrapper function that takes any RooAbsReal/Pdf object as
input.
**/

#include <RooAbsCachedPdf.h>
#include <RooAbsCachedReal.h>
#include <RooDataHist.h>
#include <RooHistPdf.h>
#include <RooMsgService.h>
#include <RooRealProxy.h>

#include <Riostream.h>

template <class Base_t>
class RooAbsSelfCached : public Base_t {
public:
   RooAbsSelfCached() {}
   /// Constructor
   RooAbsSelfCached(const char *name, const char *title, int ipOrder = 0) : Base_t(name, title, ipOrder) {}

   /// Copy constructor
   RooAbsSelfCached(const RooAbsSelfCached &other, const char *name = nullptr) : Base_t(other, name) {}

protected:
   const char *inputBaseName() const override
   {
      // Use own name as base name for caches
      return Base_t::GetName();
   }
   RooFit::OwningPtr<RooArgSet> actualObservables(const RooArgSet &nset) const override;
   RooFit::OwningPtr<RooArgSet> actualParameters(const RooArgSet &nset) const override;
   void fillCacheObject(typename Base_t::CacheElem &cache) const override;

private:
   ClassDefOverride(RooAbsSelfCached, 0); // Abstract base class for self-caching functions
};

using RooAbsSelfCachedReal = RooAbsSelfCached<RooAbsCachedReal>;
using RooAbsSelfCachedPdf = RooAbsSelfCached<RooAbsCachedPdf>;

////////////////////////////////////////////////////////////////////////////////
/// Fill cache with sampling of function as defined by the evaluate() implementation

template <class Base_t>
void RooAbsSelfCached<Base_t>::fillCacheObject(typename Base_t::CacheElem &cache) const
{
   RooDataHist &cacheHist = *cache.hist();

   // Make deep clone of self in non-caching mde and attach to dataset observables
   RooArgSet cloneSet;
   RooArgSet(*this).snapshot(cloneSet, true);
   RooAbsSelfCached *clone2 = (RooAbsSelfCached *)cloneSet.find(Base_t::GetName());
   clone2->disableCache(true);
   clone2->attachDataSet(cacheHist);

   // Iterator over all bins of RooDataHist and fill weights
   for (int i = 0; i < cacheHist.numEntries(); i++) {
      const RooArgSet *obs = cacheHist.get(i);
      double wgt = clone2->getVal(obs);
      cacheHist.set(i, wgt, 0.);
   }

   cache.setUnitNorm();
}

////////////////////////////////////////////////////////////////////////////////
/// Defines observables to be cached, given a set of user defined observables
/// Returns the subset of nset that are observables this p.d.f

template <class Base_t>
RooFit::OwningPtr<RooArgSet> RooAbsSelfCached<Base_t>::actualObservables(const RooArgSet &nset) const
{
   // Make list of servers
   RooArgSet serverSet;

   for (auto server : Base_t::servers()) {
      serverSet.add(*server);
   }

   // Return servers that are in common with given normalization set
   return RooFit::OwningPtr<RooArgSet>{serverSet.selectCommon(nset)};
}

template <>
inline RooFit::OwningPtr<RooArgSet> RooAbsSelfCached<RooAbsCachedPdf>::actualObservables(const RooArgSet & /*nset*/) const
{
   // Make list of servers
   auto serverSet = new RooArgSet;

   for (auto server : servers()) {
      serverSet->add(*server);
   }

   // Return servers that are in common with given normalization set.
   // For unknown reasons, the original implementation in RooAbsSelfCachedPdf
   // skipped the "selectCommon" strep, which is why this is the only method
   // that is implemented separately for RooAbsSelfCachedPdf.
   return RooFit::OwningPtr<RooArgSet>{serverSet};
}

////////////////////////////////////////////////////////////////////////////////
/// Defines parameters on which cache contents depends. Returns
/// subset of variables of self that is not contained in the
/// supplied nset

template <class Base_t>
RooFit::OwningPtr<RooArgSet> RooAbsSelfCached<Base_t>::actualParameters(const RooArgSet &nset) const
{
   // Make list of servers
   RooArgSet *serverSet = new RooArgSet;

   for (auto server : Base_t::servers()) {
      serverSet->add(*server);
   }

   // Remove all given observables from server list
   serverSet->remove(nset, true, true);

   return RooFit::OwningPtr<RooArgSet>{serverSet};
}

#endif
