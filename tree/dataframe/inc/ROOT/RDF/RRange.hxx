// Author: Enrico Guiraud, Danilo Piparo CERN  09/2018

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RDFRANGE
#define ROOT_RDFRANGE

#include "ROOT/RDF/RLoopManager.hxx"
#include "ROOT/RDF/RRangeBase.hxx"
#include "ROOT/RDF/Utils.hxx"
#include "RtypesCore.h"

#include <cassert>
#include <memory>

namespace ROOT {

// fwd decl
namespace Internal {
namespace RDF {
namespace GraphDrawing {
std::shared_ptr<GraphNode> CreateRangeNode(const ROOT::Detail::RDF::RRangeBase *rangePtr,
                                           std::unordered_map<void *, std::shared_ptr<GraphNode>> &visitedMap);
} // ns GraphDrawing
} // ns RDF
} // ns Internal

namespace Detail {
namespace RDF {
namespace RDFGraphDrawing = ROOT::Internal::RDF::GraphDrawing;

template <typename PrevNode>
class RRange final : public RRangeBase {
   const std::shared_ptr<PrevNode> fPrevNodePtr;
   PrevNode &fPrevNode;

public:
   RRange(unsigned int start, unsigned int stop, unsigned int stride, std::shared_ptr<PrevNode> pd)
      : RRangeBase(pd->GetLoopManagerUnchecked(), start, stop, stride, pd->GetLoopManagerUnchecked()->GetNSlots(),
                   pd->GetVariations()),
        fPrevNodePtr(std::move(pd)), fPrevNode(*fPrevNodePtr)
   {
   }

   RRange(const RRange &) = delete;
   RRange &operator=(const RRange &) = delete;
   // must call Deregister here, before fPrevNode is destroyed,
   // otherwise if fPrevNode is fLoopManager we get a use after delete
   ~RRange() { fLoopManager->Deregister(this); }

   /// Ranges act as filters when it comes to selecting entries that downstream nodes should process
   bool CheckFilters(unsigned int slot, Long64_t entry) final
   {
      if (entry != fLastCheckedEntry) {
         if (fHasStopped)
            return false;
         if (!fPrevNode.CheckFilters(slot, entry)) {
            // a filter upstream returned false, cache the result
            fLastResult = false;
         } else {
            // apply range filter logic, cache the result
            ++fNProcessedEntries;
            if (fNProcessedEntries <= fStart || (fStop > 0 && fNProcessedEntries > fStop) ||
                (fStride != 1 && fNProcessedEntries % fStride != 0))
               fLastResult = false;
            else
               fLastResult = true;
            if (fNProcessedEntries == fStop) {
               fHasStopped = true;
               fPrevNode.StopProcessing();
            }
         }
         fLastCheckedEntry = entry;
      }
      return fLastResult;
   }

   // recursive chain of `Report`s
   // RRange simply forwards these calls to the previous node
   void Report(ROOT::RDF::RCutFlowReport &rep) const final { fPrevNode.PartialReport(rep); }

   void PartialReport(ROOT::RDF::RCutFlowReport &rep) const final { fPrevNode.PartialReport(rep); }

   void StopProcessing() final
   {
      ++fNStopsReceived;
      if (fNStopsReceived == fNChildren && !fHasStopped)
         fPrevNode.StopProcessing();
   }

   void IncrChildrenCount() final
   {
      ++fNChildren;
      // propagate "children activation" upstream
      if (fNChildren == 1)
         fPrevNode.IncrChildrenCount();
   }

   /// This function must be defined by all nodes, but only the filters will add their name
   void AddFilterName(std::vector<std::string> &filters) { fPrevNode.AddFilterName(filters); }
   std::shared_ptr<RDFGraphDrawing::GraphNode>
   GetGraph(std::unordered_map<void *, std::shared_ptr<RDFGraphDrawing::GraphNode>> &visitedMap)
   {
      // TODO: Ranges node have no information about custom columns, hence it is not possible now
      // if defines have been used before.
      auto prevNode = fPrevNode.GetGraph(visitedMap);
      const auto &prevColumns = prevNode->GetDefinedColumns();

      auto thisNode = RDFGraphDrawing::CreateRangeNode(this, visitedMap);

      /* If the returned node is not new, there is no need to perform any other operation.
       * This is a likely scenario when building the entire graph in which branches share
       * some nodes. */
      if (!thisNode->IsNew()) {
         return thisNode;
      }
      thisNode->SetPrevNode(prevNode);

      // If there have been some defines between the last Filter and this Range node we won't detect them:
      // Ranges don't keep track of Defines (they have no RColumnRegister data member).
      // Let's pretend that the Defines of this node are the same as the node above, so that in the graph
      // the Defines will just appear below the Range instead (no functional change).
      thisNode->AddDefinedColumns(prevColumns);

      return thisNode;
   }

   std::shared_ptr<RNodeBase> GetVariedFilter(const std::string &variationName) final
   {
      // nobody should ask for a varied filter for the nominal variation: they can just
      // use the nominal filter!
      assert(variationName != "nominal");
      // nobody should ask for a varied filter for a variation on which this filter does not depend:
      // they can just use the nominal filter.
      assert(RDFInternal::IsStrInVec(variationName, fVariations));

      auto it = fVariedRanges.find(variationName);
      if (it != fVariedRanges.end())
         return it->second;

      auto prevNode = fPrevNodePtr;
      if (static_cast<RNodeBase *>(fPrevNodePtr.get()) != static_cast<RNodeBase *>(fLoopManager) &&
          RDFInternal::IsStrInVec(variationName, prevNode->GetVariations()))
         prevNode = std::static_pointer_cast<PrevNode>(prevNode->GetVariedFilter(variationName));

      auto variedRange = std::unique_ptr<RRangeBase>(new RRange(fStart, fStop, fStride, std::move(prevNode)));
      fLoopManager->Book(variedRange.get());
      auto e = fVariedRanges.insert({variationName, std::move(variedRange)});
      return e.first->second;
   }
};

} // namespace RDF
} // namespace Detail
} // namespace ROOT

#endif // ROOT_RDFRANGE
