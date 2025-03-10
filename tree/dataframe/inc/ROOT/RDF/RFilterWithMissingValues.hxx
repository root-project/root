// Author: Vincenzo Eduardo Padulano CERN 09/2024

/*************************************************************************
 * Copyright (C) 1995-2024, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RDF_RFilterWithMissingValues
#define ROOT_RDF_RFilterWithMissingValues

#include "ROOT/RDF/ColumnReaderUtils.hxx"
#include "ROOT/RDF/RColumnReaderBase.hxx"
#include "ROOT/RDF/RCutFlowReport.hxx"
#include "ROOT/RDF/Utils.hxx"
#include "ROOT/RDF/RFilterBase.hxx"
#include "ROOT/RDF/RLoopManager.hxx"
#include "ROOT/RDF/RTreeColumnReader.hxx"
#include "ROOT/TypeTraits.hxx"
#include "RtypesCore.h"

#include <algorithm>
#include <cassert>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility> // std::index_sequence
#include <vector>

// fwd decls for RFilterWithMissingValues
namespace ROOT::Internal::RDF::GraphDrawing {
std::shared_ptr<GraphNode> CreateFilterNode(const ROOT::Detail::RDF::RFilterBase *filterPtr,
                                            std::unordered_map<void *, std::shared_ptr<GraphNode>> &visitedMap);

std::shared_ptr<GraphNode> AddDefinesToGraph(std::shared_ptr<GraphNode> node,
                                             const ROOT::Internal::RDF::RColumnRegister &colRegister,
                                             const std::vector<std::string> &prevNodeDefines,
                                             std::unordered_map<void *, std::shared_ptr<GraphNode>> &visitedMap);
} // namespace ROOT::Internal::RDF::GraphDrawing

namespace ROOT::Detail::RDF {

namespace RDFGraphDrawing = ROOT::Internal::RDF::GraphDrawing;
class RJittedFilter;

/**
 * \brief implementation of FilterAvailable and FilterMissing operations
 *
 * The filter evaluates if the entry is missing a value for the input column.
 * Depending on which function was called by the user, the entry with the
 * missing value:
 * - will be discarded in case the user called FilterAvailable
 * - will be kept in case the user called FilterMissing
 */
template <typename PrevNodeRaw>
class R__CLING_PTRCHECK(off) RFilterWithMissingValues final : public RFilterBase {

   // If the PrevNode is a RJittedFilter, treat it as a more generic RFilterBase: when dealing with systematic
   // variations we'll have a RJittedFilter node for the nominal case but other "universes" will use concrete filters,
   // so we normalize the "previous node type" to the base type RFilterBase.
   using PrevNode_t = std::conditional_t<std::is_same<PrevNodeRaw, RJittedFilter>::value, RFilterBase, PrevNodeRaw>;
   const std::shared_ptr<PrevNode_t> fPrevNodePtr;

   // One column reader per slot
   std::vector<RColumnReaderBase *> fValues;

   // Whether the entry should be kept in case of missing value for the input column
   bool fDiscardEntryWithMissingValue;

public:
   RFilterWithMissingValues(bool discardEntry, std::shared_ptr<PrevNode_t> pd,
                            const RDFInternal::RColumnRegister &colRegister, const ColumnNames_t &columns,
                            std::string_view filterName = "", const std::string &variationName = "nominal")
      : RFilterBase(pd->GetLoopManagerUnchecked(), filterName, pd->GetLoopManagerUnchecked()->GetNSlots(), colRegister,
                    columns, pd->GetVariations(), variationName),
        fPrevNodePtr(std::move(pd)),
        fValues(fPrevNodePtr->GetLoopManagerUnchecked()->GetNSlots()),
        fDiscardEntryWithMissingValue(discardEntry)
   {
      fLoopManager->Register(this);
      // We suppress errors that TTreeReader prints regarding the missing branch
      fLoopManager->InsertSuppressErrorsForMissingBranch(fColumnNames[0]);
   }

   RFilterWithMissingValues(const RFilterWithMissingValues &) = delete;
   RFilterWithMissingValues &operator=(const RFilterWithMissingValues &) = delete;
   RFilterWithMissingValues(RFilterWithMissingValues &&) = delete;
   RFilterWithMissingValues &operator=(RFilterWithMissingValues &&) = delete;
   ~RFilterWithMissingValues() final
   {
      // must Deregister objects from the RLoopManager here, before the fPrevNodePtr data member is destroyed:
      // otherwise if fPrevNodePtr is the RLoopManager, it will be destroyed before the calls to Deregister happen.
      fLoopManager->Deregister(this);
      fLoopManager->EraseSuppressErrorsForMissingBranch(fColumnNames[0]);
   }

   bool CheckFilters(unsigned int slot, Long64_t entry) final
   {
      constexpr static auto cacheLineStepLong64_t = RDFInternal::CacheLineStep<Long64_t>();
      constexpr static auto cacheLineStepint = RDFInternal::CacheLineStep<int>();
      constexpr static auto cacheLineStepULong64_t = RDFInternal::CacheLineStep<ULong64_t>();

      if (entry != fLastCheckedEntry[slot * cacheLineStepLong64_t]) {
         if (!fPrevNodePtr->CheckFilters(slot, entry)) {
            // a filter upstream returned false, cache the result
            fLastResult[slot * cacheLineStepint] = false;
         } else {
            // evaluate this filter, cache the result
            const bool valueIsMissing = fValues[slot]->template TryGet<void>(entry) == nullptr;
            if (fDiscardEntryWithMissingValue) {
               valueIsMissing ? ++fRejected[slot * cacheLineStepULong64_t] : ++fAccepted[slot * cacheLineStepULong64_t];
               fLastResult[slot * cacheLineStepint] = !valueIsMissing;
            } else {
               valueIsMissing ? ++fAccepted[slot * cacheLineStepULong64_t] : ++fRejected[slot * cacheLineStepULong64_t];
               fLastResult[slot * cacheLineStepint] = valueIsMissing;
            }
         }
         fLastCheckedEntry[slot * cacheLineStepLong64_t] = entry;
      }
      return fLastResult[slot * cacheLineStepint];
   }

   void InitSlot(TTreeReader *r, unsigned int slot) final
   {
      fValues[slot] =
         RDFInternal::GetColumnReader(slot, fColRegister.GetReaderUnchecked(slot, fColumnNames[0], fVariation),
                                      *fLoopManager, r, fColumnNames[0], typeid(void));
      fLastCheckedEntry[slot * RDFInternal::CacheLineStep<Long64_t>()] = -1;
   }

   // recursive chain of `Report`s
   void Report(ROOT::RDF::RCutFlowReport &rep) const final { PartialReport(rep); }

   void PartialReport(ROOT::RDF::RCutFlowReport &rep) const final
   {
      fPrevNodePtr->PartialReport(rep);
      FillReport(rep);
   }

   void StopProcessing() final
   {
      ++fNStopsReceived;
      if (fNStopsReceived == fNChildren)
         fPrevNodePtr->StopProcessing();
   }

   void IncrChildrenCount() final
   {
      ++fNChildren;
      // propagate "children activation" upstream. named filters do the propagation via `TriggerChildrenCount`.
      if (fNChildren == 1 && fName.empty())
         fPrevNodePtr->IncrChildrenCount();
   }

   void TriggerChildrenCount() final
   {
      assert(!fName.empty()); // this method is to only be called on named filters
      fPrevNodePtr->IncrChildrenCount();
   }

   void AddFilterName(std::vector<std::string> &filters) final
   {
      fPrevNodePtr->AddFilterName(filters);
      auto name = (HasName() ? fName : fDiscardEntryWithMissingValue ? "FilterAvailable" : "FilterMissing");
      filters.push_back(name);
   }

   /// Clean-up operations to be performed at the end of a task.
   void FinalizeSlot(unsigned int slot) final { fValues[slot] = nullptr; }

   std::shared_ptr<RDFGraphDrawing::GraphNode>
   GetGraph(std::unordered_map<void *, std::shared_ptr<RDFGraphDrawing::GraphNode>> &visitedMap) final
   {
      // Recursively call for the previous node.
      auto prevNode = fPrevNodePtr->GetGraph(visitedMap);
      const auto &prevColumns = prevNode->GetDefinedColumns();

      auto thisNode = RDFGraphDrawing::CreateFilterNode(this, visitedMap);

      /* If the returned node is not new, there is no need to perform any other operation.
       * This is a likely scenario when building the entire graph in which branches share
       * some nodes. */
      if (!thisNode->IsNew()) {
         return thisNode;
      }

      auto upmostNode = AddDefinesToGraph(thisNode, fColRegister, prevColumns, visitedMap);

      // Keep track of the columns defined up to this point.
      thisNode->AddDefinedColumns(fColRegister.GenerateColumnNames());

      upmostNode->SetPrevNode(prevNode);
      return thisNode;
   }

   /// Return a clone of this Filter that works with values in the variationName "universe".
   std::shared_ptr<RNodeBase> GetVariedFilter(const std::string &variationName) final
   {
      // Only the nominal filter should be asked to produce varied filters
      assert(fVariation == "nominal");
      // nobody should ask for a varied filter for the nominal variation: they can just
      // use the nominal filter!
      assert(variationName != "nominal");
      // nobody should ask for a varied filter for a variation on which this filter does not depend:
      // they can just use the nominal filter.
      assert(RDFInternal::IsStrInVec(variationName, fVariations));

      auto it = fVariedFilters.find(variationName);
      if (it != fVariedFilters.end())
         return it->second;

      auto prevNode = fPrevNodePtr;
      if (static_cast<RNodeBase *>(fPrevNodePtr.get()) != static_cast<RNodeBase *>(fLoopManager) &&
          RDFInternal::IsStrInVec(variationName, prevNode->GetVariations()))
         prevNode = std::static_pointer_cast<PrevNode_t>(prevNode->GetVariedFilter(variationName));

      // the varied filters get a copy of the callable object.
      // TODO document this
      auto variedFilter = std::unique_ptr<RFilterBase>(new RFilterWithMissingValues<PrevNode_t>(
         fDiscardEntryWithMissingValue, std::move(prevNode), fColRegister, fColumnNames, fName, variationName));
      auto e = fVariedFilters.insert({variationName, std::move(variedFilter)});
      return e.first->second;
   }
};

} // namespace ROOT::Detail::RDF

#endif // ROOT_RDF_RFilterWithMissingValues
