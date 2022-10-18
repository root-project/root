// Author: Enrico Guiraud, Danilo Piparo CERN  09/2018

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RFILTER
#define ROOT_RFILTER

#include "ROOT/RDF/ColumnReaderUtils.hxx"
#include "ROOT/RDF/RColumnReaderBase.hxx"
#include "ROOT/RDF/RCutFlowReport.hxx"
#include "ROOT/RDF/Utils.hxx"
#include "ROOT/RDF/RFilterBase.hxx"
#include "ROOT/RDF/RLoopManager.hxx"
#include "ROOT/TypeTraits.hxx"
#include "RtypesCore.h"

#include <algorithm>
#include <cassert>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility> // std::index_sequence
#include <vector>

namespace ROOT {

namespace Internal {
namespace RDF {
using namespace ROOT::Detail::RDF;

// fwd decl for RFilter
namespace GraphDrawing {
std::shared_ptr<GraphNode> CreateFilterNode(const RFilterBase *filterPtr);

std::shared_ptr<GraphNode> AddDefinesToGraph(std::shared_ptr<GraphNode> node,
                                             const RDFInternal::RColumnRegister &colRegister,
                                             const std::vector<std::string> &prevNodeDefines);
} // ns GraphDrawing

} // ns RDF
} // ns Internal

namespace Detail {
namespace RDF {
using namespace ROOT::TypeTraits;
namespace RDFGraphDrawing = ROOT::Internal::RDF::GraphDrawing;
class RJittedFilter;

template <typename FilterF, typename PrevNodeRaw>
class R__CLING_PTRCHECK(off) RFilter final : public RFilterBase {
   using ColumnTypes_t = typename CallableTraits<FilterF>::arg_types;
   using TypeInd_t = std::make_index_sequence<ColumnTypes_t::list_size>;
   // If the PrevNode is a RJittedFilter, treat it as a more generic RFilterBase: when dealing with systematic
   // variations we'll have a RJittedFilter node for the nominal case but other "universes" will use concrete filters,
   // so we normalize the "previous node type" to the base type RFilterBase.
   using PrevNode_t = std::conditional_t<std::is_same<PrevNodeRaw, RJittedFilter>::value, RFilterBase, PrevNodeRaw>;

   FilterF fFilter;
   /// Column readers per slot and per input column
   std::vector<std::array<std::unique_ptr<RColumnReaderBase>, ColumnTypes_t::list_size>> fValues;
   const std::shared_ptr<PrevNode_t> fPrevNodePtr;
   PrevNode_t &fPrevNode;

public:
   RFilter(FilterF f, const ROOT::RDF::ColumnNames_t &columns, std::shared_ptr<PrevNode_t> pd,
           const RDFInternal::RColumnRegister &colRegister, std::string_view name = "",
           const std::string &variationName = "nominal")
      : RFilterBase(pd->GetLoopManagerUnchecked(), name, pd->GetLoopManagerUnchecked()->GetNSlots(), colRegister,
                    columns, pd->GetVariations(), variationName),
        fFilter(std::move(f)), fValues(pd->GetLoopManagerUnchecked()->GetNSlots()), fPrevNodePtr(std::move(pd)),
        fPrevNode(*fPrevNodePtr)
   {
      fLoopManager->Book(this);
   }

   RFilter(const RFilter &) = delete;
   RFilter &operator=(const RFilter &) = delete;
   ~RFilter() {
      // must Deregister objects from the RLoopManager here, before the fPrevNode data member is destroyed:
      // otherwise if fPrevNode is the RLoopManager, it will be destroyed before the calls to Deregister happen.
      fColRegister.Clear(); // triggers RDefine deregistration
      fLoopManager->Deregister(this);
   }

   bool CheckFilters(unsigned int slot, Long64_t entry) final
   {
      if (entry != fLastCheckedEntry[slot * RDFInternal::CacheLineStep<Long64_t>()]) {
         if (!fPrevNode.CheckFilters(slot, entry)) {
            // a filter upstream returned false, cache the result
            fLastResult[slot * RDFInternal::CacheLineStep<int>()] = false;
         } else {
            // evaluate this filter, cache the result
            auto passed = CheckFilterHelper(slot, entry, ColumnTypes_t{}, TypeInd_t{});
            passed ? ++fAccepted[slot * RDFInternal::CacheLineStep<ULong64_t>()]
                   : ++fRejected[slot * RDFInternal::CacheLineStep<ULong64_t>()];
            fLastResult[slot * RDFInternal::CacheLineStep<int>()] = passed;
         }
         fLastCheckedEntry[slot * RDFInternal::CacheLineStep<Long64_t>()] = entry;
      }
      return fLastResult[slot * RDFInternal::CacheLineStep<int>()];
   }

   template <typename... ColTypes, std::size_t... S>
   bool CheckFilterHelper(unsigned int slot, Long64_t entry, TypeList<ColTypes...>, std::index_sequence<S...>)
   {
      // silence "unused parameter" warnings in gcc
      (void)slot;
      (void)entry;
      return fFilter(fValues[slot][S]->template Get<ColTypes>(entry)...);
   }

   void InitSlot(TTreeReader *r, unsigned int slot) final
   {
      RDFInternal::RColumnReadersInfo info{fColumnNames, fColRegister, fIsDefine.data(), fLoopManager->GetDSValuePtrs(),
                                           fLoopManager->GetDataSource()};
      fValues[slot] = RDFInternal::MakeColumnReaders(slot, r, ColumnTypes_t{}, info, fVariation);
      fLastCheckedEntry[slot * RDFInternal::CacheLineStep<Long64_t>()] = -1;
   }

   // recursive chain of `Report`s
   void Report(ROOT::RDF::RCutFlowReport &rep) const final { PartialReport(rep); }

   void PartialReport(ROOT::RDF::RCutFlowReport &rep) const final
   {
      fPrevNode.PartialReport(rep);
      FillReport(rep);
   }

   void StopProcessing() final
   {
      ++fNStopsReceived;
      if (fNStopsReceived == fNChildren)
         fPrevNode.StopProcessing();
   }

   void IncrChildrenCount() final
   {
      ++fNChildren;
      // propagate "children activation" upstream. named filters do the propagation via `TriggerChildrenCount`.
      if (fNChildren == 1 && fName.empty())
         fPrevNode.IncrChildrenCount();
   }

   void TriggerChildrenCount() final
   {
      assert(!fName.empty()); // this method is to only be called on named filters
      fPrevNode.IncrChildrenCount();
   }

   void AddFilterName(std::vector<std::string> &filters)
   {
      fPrevNode.AddFilterName(filters);
      auto name = (HasName() ? fName : "Unnamed Filter");
      filters.push_back(name);
   }

   /// Clean-up operations to be performed at the end of a task.
   virtual void FinaliseSlot(unsigned int slot) final
   {
      for (auto &v : fValues[slot])
         v.reset();
   }

   std::shared_ptr<RDFGraphDrawing::GraphNode> GetGraph()
   {
      // Recursively call for the previous node.
      auto prevNode = fPrevNode.GetGraph();
      auto prevColumns = prevNode->GetDefinedColumns();

      auto thisNode = RDFGraphDrawing::CreateFilterNode(this);

      /* If the returned node is not new, there is no need to perform any other operation.
       * This is a likely scenario when building the entire graph in which branches share
       * some nodes. */
      if (!thisNode->GetIsNew()) {
         return thisNode;
      }

      auto upmostNode = AddDefinesToGraph(thisNode, fColRegister, prevColumns);

      // Keep track of the columns defined up to this point.
      thisNode->AddDefinedColumns(fColRegister.GetNames());

      upmostNode->SetPrevNode(prevNode);
      return thisNode;
   }

   /// Return a clone of this Filter that works with values in the variationName "universe".
   std::shared_ptr<RNodeBase> GetVariedFilter(const std::string &variationName) final
   {
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
      auto variedFilter = std::unique_ptr<RFilterBase>(
         new RFilter(fFilter, fColumnNames, std::move(prevNode), fColRegister, fName, variationName));
      auto e = fVariedFilters.insert({variationName, std::move(variedFilter)});
      return e.first->second;
   }
};

} // ns RDF
} // ns Detail
} // ns ROOT

#endif // ROOT_RFILTER
