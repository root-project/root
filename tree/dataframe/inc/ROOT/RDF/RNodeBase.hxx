// Author: Enrico Guiraud, Danilo Piparo CERN  09/2018

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RDFNODEBASE
#define ROOT_RDFNODEBASE

#include "RtypesCore.h"
#include "TError.h" // R__ASSERT

#include <memory>
#include <string>
#include <vector>

namespace ROOT {
namespace RDF {
class RCutFlowReport;
}

namespace Internal {
namespace RDF {
namespace GraphDrawing {
class GraphNode;
}
}
}

namespace Detail {
namespace RDF {

class RLoopManager;

/// Base class for non-leaf nodes of the computational graph.
/// It only exposes the bare minimum interface required to work as a generic part of the computation graph.
/// RDataFrames and results of transformations can be cast to this type via ROOT::RDF::RNode (or ROOT.RDF.AsRNode in PyROOT).
class RNodeBase {
protected:
   RLoopManager *fLoopManager;
   unsigned int fNChildren{0};      ///< Number of nodes of the functional graph hanging from this object
   unsigned int fNStopsReceived{0}; ///< Number of times that a children node signaled to stop processing entries.
   std::vector<std::string> fVariations; ///< List of systematic variations that affect this node.

public:
   RNodeBase(const std::vector<std::string> &variations = {}, RLoopManager *lm = nullptr)
      : fLoopManager(lm), fVariations(variations)
   {
   }
   virtual ~RNodeBase() {}
   virtual bool CheckFilters(unsigned int, Long64_t) = 0;
   virtual void Report(ROOT::RDF::RCutFlowReport &) const = 0;
   virtual void PartialReport(ROOT::RDF::RCutFlowReport &) const = 0;
   virtual void IncrChildrenCount() = 0;
   virtual void StopProcessing() = 0;
   virtual void AddFilterName(std::vector<std::string> &filters) = 0;
   // Helper function for SaveGraph
   virtual std::shared_ptr<ROOT::Internal::RDF::GraphDrawing::GraphNode> GetGraph() = 0;

   virtual void ResetChildrenCount()
   {
      fNChildren = 0;
      fNStopsReceived = 0;
   }

   virtual RLoopManager *GetLoopManagerUnchecked() { return fLoopManager; }

   const std::vector<std::string> &GetVariations() const { return fVariations; }

   /// Return a clone of this node that acts as a Filter working with values in the variationName "universe".
   virtual std::shared_ptr<RNodeBase> GetVariedFilter(const std::string &variationName)
   {
      R__ASSERT(false &&
                "GetVariedFilter was called on a node type that does not implement it. This should never happen.");
      (void)variationName;
      return nullptr;
   }
};
} // ns RDF
} // ns Detail
} // ns ROOT

#endif
