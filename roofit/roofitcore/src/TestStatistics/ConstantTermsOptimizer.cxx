/*
 * Project: RooFit
 * Authors:
 *   PB, Patrick Bos, Netherlands eScience Center, p.bos@esciencecenter.nl
 *
 * Copyright (c) 2021, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#include "ConstantTermsOptimizer.h"

#include <RooMsgService.h>
#include <RooVectorDataStore.h> // complete type for dynamic cast
#include <RooAbsReal.h>
#include <RooArgSet.h>
#include <RooAbsData.h>

namespace RooFit {
namespace TestStatistics {

/** \class ConstantTermsOptimizer
 *
 * \brief Analyzes a function given a dataset/observables for constant terms and caches those in the dataset
 *
 * This optimizer should be used on a consistent combination of function (usually a pdf) and a dataset with observables.
 * It then analyzes the function to find parts that can be precalculated because they are constant given the set of
 * observables. These are cached inside the dataset and used in subsequent evaluations of the function on that dataset.
 * The typical use case for this is inside likelihood minimization where many calls of the same pdf/dataset combination
 * are made. \p norm_set must provide the normalization set of the function, which would typically be the set of
 * observables in the dataset; this is used to make sure all object caches are created before analysis by evaluating the
 * function on this set at the beginning of enableConstantTermsOptimization.
 */

RooArgSet ConstantTermsOptimizer::requiredExtraObservables()
{
   // TODO: the RooAbsOptTestStatistics::requiredExtraObservables() call this code was copied
   //       from was overloaded for RooXYChi2Var only; implement different options when necessary
   return RooArgSet();
}

void ConstantTermsOptimizer::enableConstantTermsOptimization(RooAbsReal *function, RooArgSet *norm_set, RooAbsData *dataset,
                                        bool applyTrackingOpt)
{
   // Trigger create of all object caches now in nodes that have deferred object creation
   // so that cache contents can be processed immediately
   function->getVal(norm_set);

   // Apply tracking optimization here. Default strategy is to track components
   // of RooAddPdfs and RooRealSumPdfs. If these components are a RooProdPdf
   // or a RooProduct respectively, track the components of these products instead
   // of the product term
   RooArgSet trackNodes;

   // Add safety check here - applyTrackingOpt will only be applied if present
   // dataset is constructed in terms of a RooVectorDataStore
   if (applyTrackingOpt) {
      if (!dynamic_cast<RooVectorDataStore *>(dataset->store())) {
         oocoutW(nullptr, Optimization)
            << "enableConstantTermsOptimization(function: " << function->GetName()
            << ", dataset: " << dataset->GetName()
            << ") WARNING Cache-and-track optimization (Optimize level 2) is only available for datasets"
            << " implemented in terms of RooVectorDataStore - ignoring this option for current dataset" << std::endl;
         applyTrackingOpt = false;
      }
   }

   if (applyTrackingOpt) {
      RooArgSet branches;
      function->branchNodeServerList(&branches);
      for (const auto arg : branches) {
         arg->setCacheAndTrackHints(trackNodes);
      }
      // Do not set CacheAndTrack on constant expressions
      auto constNodes = (RooArgSet *)trackNodes.selectByAttrib("Constant", true);
      trackNodes.remove(*constNodes);
      delete constNodes;

      // Set CacheAndTrack flag on all remaining nodes
      trackNodes.setAttribAll("CacheAndTrack", true);
   }

   // Find all nodes that depend exclusively on constant parameters
   RooArgSet cached_nodes;

   function->findConstantNodes(*dataset->get(), cached_nodes);

   // Cache constant nodes with dataset - also cache entries corresponding to zero-weights in data when using
   // BinnedLikelihood
   // NOTE: we pass nullptr as cache-owner here, because we don't use the cacheOwner() anywhere in TestStatistics
   // TODO: make sure this (nullptr) is always correct
   dataset->cacheArgs(nullptr, cached_nodes, norm_set, !function->getAttribute("BinnedLikelihood"));

   // Put all cached nodes in AClean value caching mode so that their evaluate() is never called
   for (const auto cacheArg : cached_nodes) {
      cacheArg->setOperMode(RooAbsArg::AClean);
   }

   std::unique_ptr<RooArgSet> constNodes {(RooArgSet *)cached_nodes.selectByAttrib("ConstantExpressionCached", true)};
   RooArgSet actualTrackNodes(cached_nodes);
   actualTrackNodes.remove(*constNodes);
   if (constNodes->getSize() > 0) {
      if (constNodes->getSize() < 20) {
         oocoutI(nullptr, Minimization)
            << " The following expressions have been identified as constant and will be precalculated and cached: "
            << *constNodes << std::endl;
      } else {
         oocoutI(nullptr, Minimization) << " A total of " << constNodes->getSize()
                             << " expressions have been identified as constant and will be precalculated and cached."
                             << std::endl;
      }
   }
   if (actualTrackNodes.getSize() > 0) {
      if (actualTrackNodes.getSize() < 20) {
         oocoutI(nullptr, Minimization) << " The following expressions will be evaluated in cache-and-track mode: "
                             << actualTrackNodes << std::endl;
      } else {
         oocoutI(nullptr, Minimization) << " A total of " << constNodes->getSize()
                             << " expressions will be evaluated in cache-and-track-mode." << std::endl;
      }
   }

   // Disable reading of observables that are no longer used
   dataset->optimizeReadingWithCaching(*function, cached_nodes, requiredExtraObservables());
}

void ConstantTermsOptimizer::disableConstantTermsOptimization(RooAbsReal *function, RooArgSet *norm_set, RooArgSet *observables,
                                                                 RooAbsData *dataset)
{
   // Delete the cache
   dataset->resetCache();

   // Reactivate all tree branches
   dataset->setArgStatus(*dataset->get(), true);

   // Reset all nodes to ADirty
   optimizeCaching(function, norm_set, observables, dataset);

   // Disable propagation of dirty state flags for observables
   dataset->setDirtyProp(false);

   //   _cachedNodes.removeAll();

   //   _optimized = false;
}

void ConstantTermsOptimizer::optimizeCaching(RooAbsReal *function, RooArgSet *norm_set, RooArgSet *observables, RooAbsData *dataset)
{
   // Trigger create of all object caches now in nodes that have deferred object creation
   // so that cache contents can be processed immediately
   function->getVal(norm_set);

   // Set value caching mode for all nodes that depend on any of the observables to ADirty
   function->optimizeCacheMode(*observables);

   // Disable propagation of dirty state flags for observables
   dataset->setDirtyProp(false);

   // Disable reading of observables that are not used
   dataset->optimizeReadingWithCaching(*function, RooArgSet(), requiredExtraObservables()) ;
}

} // namespace TestStatistics
} // namespace RooFit
