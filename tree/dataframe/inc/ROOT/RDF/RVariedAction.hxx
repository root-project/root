// Author: Enrico Guiraud, CERN 11/2021

/*************************************************************************
 * Copyright (C) 1995-2022, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RVARIEDACTION
#define ROOT_RVARIEDACTION

#include "ColumnReaderUtils.hxx"
#include "GraphNode.hxx"
#include "RActionBase.hxx"
#include "RColumnReaderBase.hxx"
#include "RLoopManager.hxx"
#include "RJittedFilter.hxx"
#include "ROOT/RDF/RMergeableValue.hxx"

#include <Rtypes.h> // R__CLING_PTRCHECK
#include <ROOT/TypeTraits.hxx>

#include <algorithm>
#include <array>
#include <memory>
#include <utility> // make_index_sequence
#include <vector>

namespace ROOT {
namespace Internal {
namespace RDF {

namespace RDFGraphDrawing = ROOT::Internal::RDF::GraphDrawing;

/// Just like an RAction, but it has N action helpers (one per variation + nominal) and N previous nodes.
template <typename Helper, typename PrevNode, typename ColumnTypes_t>
class R__CLING_PTRCHECK(off) RVariedAction final : public RActionBase {
   using TypeInd_t = std::make_index_sequence<ColumnTypes_t::list_size>;
   // If the PrevNode is a RJittedFilter, our collection of previous nodes will have to use the RNodeBase type:
   // we'll have a RJittedFilter for the nominal case, but the others will be concrete filters.
   using PrevNodeType = std::conditional_t<std::is_same<PrevNode, RJittedFilter>::value, RFilterBase, PrevNode>;

   std::vector<Helper> fHelpers; ///< Action helpers per variation.
   /// Owning pointers to upstream nodes for each systematic variation (with the "nominal" at index 0).
   std::vector<std::shared_ptr<PrevNodeType>> fPrevNodes;

   /// Column readers per slot (outer dimension), per variation and per input column (inner dimension, std::array).
   std::vector<std::vector<std::array<std::unique_ptr<RColumnReaderBase>, ColumnTypes_t::list_size>>> fInputValues;

   /// The nth flag signals whether the nth input column is a custom column or not.
   std::array<bool, ColumnTypes_t::list_size> fIsDefine;

   std::vector<std::shared_ptr<PrevNodeType>> MakePrevFilters(std::shared_ptr<PrevNode> nominal) const
   {
      const auto &variations = GetVariations();
      std::vector<std::shared_ptr<PrevNodeType>> prevFilters;
      prevFilters.reserve(variations.size());
      if (static_cast<RNodeBase *>(nominal.get()) == fLoopManager) {
         // just fill this with the RLoopManager N times
         prevFilters.resize(variations.size(), nominal);
      } else {
         // create varied versions of the previous filter node
         const auto &prevVariations = nominal->GetVariations();
         for (const auto &variation : variations) {
            if (IsStrInVec(variation, prevVariations)) {
               prevFilters.emplace_back(std::static_pointer_cast<PrevNodeType>(nominal->GetVariedFilter(variation)));
            } else {
               prevFilters.emplace_back(nominal);
            }
         }
      }

      return prevFilters;
   }

public:
   RVariedAction(std::vector<Helper> &&helpers, const ColumnNames_t &columns, std::shared_ptr<PrevNode> prevNode,
                 const RColumnRegister &colRegister)
      : RActionBase(prevNode->GetLoopManagerUnchecked(), columns, colRegister, prevNode->GetVariations()),
        fHelpers(std::move(helpers)), fPrevNodes(MakePrevFilters(prevNode)), fInputValues(GetNSlots())
   {
      const auto &defines = colRegister.GetColumns();
      for (auto i = 0u; i < columns.size(); ++i) {
         auto it = defines.find(columns[i]);
         fIsDefine[i] = it != defines.end();
         if (fIsDefine[i])
            (it->second)->MakeVariations(GetVariations());
      }
   }

   RVariedAction(const RVariedAction &) = delete;
   RVariedAction &operator=(const RVariedAction &) = delete;

   void Initialize() final
   {
      std::for_each(fHelpers.begin(), fHelpers.end(), [](Helper &h) { h.Initialize(); });
   }

   void InitSlot(TTreeReader *r, unsigned int slot) final
   {
      RDFInternal::RColumnReadersInfo info{GetColumnNames(), GetColRegister(), fIsDefine.data(),
                                           fLoopManager->GetDSValuePtrs(), fLoopManager->GetDataSource()};

      // get readers for each systematic variation
      for (const auto &variation : GetVariations())
         fInputValues[slot].emplace_back(MakeColumnReaders(slot, r, ColumnTypes_t{}, info, variation));

      std::for_each(fHelpers.begin(), fHelpers.end(), [=](Helper &h) { h.InitTask(r, slot); });
   }

   template <typename... ColTypes, std::size_t... S>
   void
   CallExec(unsigned int slot, unsigned int varIdx, Long64_t entry, TypeList<ColTypes...>, std::index_sequence<S...>)
   {
      fHelpers[varIdx].Exec(slot, fInputValues[slot][varIdx][S]->template Get<ColTypes>(entry)...);
      (void)entry;
   }

   void Run(unsigned int slot, Long64_t entry) final
   {
      for (auto varIdx = 0u; varIdx < GetVariations().size(); ++varIdx) {
         if (fPrevNodes[varIdx]->CheckFilters(slot, entry))
            CallExec(slot, varIdx, entry, ColumnTypes_t{}, TypeInd_t{});
      }
   }

   void TriggerChildrenCount() final
   {
      std::for_each(fPrevNodes.begin(), fPrevNodes.end(), [](auto &f) { f->IncrChildrenCount(); });
   }

   /// Clean-up operations to be performed at the end of a task.
   void FinalizeSlot(unsigned int slot) final
   {
      fInputValues[slot].clear();
      std::for_each(fHelpers.begin(), fHelpers.end(), [=](Helper &h) { h.CallFinalizeTask(slot); });
   }

   /// Clean-up and finalize the action result (e.g. merging slot-local results).
   /// It invokes the helper's Finalize method.
   void Finalize() final
   {
      std::for_each(fHelpers.begin(), fHelpers.end(), [](Helper &h) { h.Finalize(); });
      SetHasRun();
   }

   /// Return the partially-updated value connected to the nominal result.
   void *PartialUpdate(unsigned int slot) final { return PartialUpdateImpl(slot); }

   /// Return the per-sample callback connected to the nominal result.
   ROOT::RDF::SampleCallback_t GetSampleCallback() final { return fHelpers[0].GetSampleCallback(); }

   std::shared_ptr<RDFGraphDrawing::GraphNode>
   GetGraph(std::unordered_map<void *, std::shared_ptr<RDFGraphDrawing::GraphNode>> &visitedMap) final
   {
      auto prevNode = fPrevNodes[0]->GetGraph(visitedMap);
      const auto &prevColumns = prevNode->GetDefinedColumns();

      // Action nodes do not need to go through CreateFilterNode: they are never common nodes between multiple branches
      const auto nodeType = HasRun() ? RDFGraphDrawing::ENodeType::kUsedAction : RDFGraphDrawing::ENodeType::kAction;
      auto thisNode = std::make_shared<RDFGraphDrawing::GraphNode>("Varied " + fHelpers[0].GetActionName(),
                                                                   visitedMap.size(), nodeType);
      visitedMap[(void *)this] = thisNode;

      auto upmostNode = AddDefinesToGraph(thisNode, GetColRegister(), prevColumns, visitedMap);

      thisNode->AddDefinedColumns(GetColRegister().GetNames());
      upmostNode->SetPrevNode(prevNode);
      return thisNode;
   }

   /**
      Retrieve a container holding the names and values of the variations. It
      knows how to merge with others of the same type.
   */
   std::unique_ptr<RMergeableValueBase> GetMergeableValue() const final
   {
      std::vector<std::string> keys{GetVariations()};

      std::vector<std::unique_ptr<RDFDetail::RMergeableValueBase>> values;
      values.reserve(fHelpers.size());
      for (auto &&h : fHelpers)
         values.emplace_back(h.GetMergeableValue());

      return std::make_unique<RDFDetail::RMergeableVariationsBase>(std::move(keys), std::move(values));
   }

   [[noreturn]] std::unique_ptr<RActionBase> MakeVariedAction(std::vector<void *> &&)
   {
      throw std::logic_error("Cannot produce a varied action from a varied action.");
   }

private:
   // this overload is SFINAE'd out if Helper does not implement `PartialUpdate`
   // the template parameter is required to defer instantiation of the method to SFINAE time
   template <typename H = Helper>
   auto PartialUpdateImpl(unsigned int slot) -> decltype(std::declval<H>().PartialUpdate(slot), (void *)(nullptr))
   {
      return &fHelpers[0].PartialUpdate(slot);
   }

   // this one is always available but has lower precedence thanks to `...`
   void *PartialUpdateImpl(...) { throw std::runtime_error("This action does not support callbacks!"); }
};

} // namespace RDF
} // namespace Internal
} // namespace ROOT

#endif // ROOT_RVARIEDACTION
