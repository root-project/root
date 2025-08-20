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
#include "ROOT/RDF/RSampleInfo.hxx"

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

/// Just like an RAction, but it has N action helpers and N previous nodes (N is the number of variations).
template <typename Helper, typename PrevNode, typename ColumnTypes_t>
class R__CLING_PTRCHECK(off) RVariedAction final : public RActionBase {
   using TypeInd_t = std::make_index_sequence<ColumnTypes_t::list_size>;
   // If the PrevNode is a RJittedFilter, our collection of previous nodes will have to use the RNodeBase type:
   // we'll have a RJittedFilter for the nominal case, but the others will be concrete filters.
   using PrevNodeType = std::conditional_t<std::is_same<PrevNode, RJittedFilter>::value, RFilterBase, PrevNode>;

   std::vector<Helper> fHelpers; ///< Action helpers per variation.
   /// Owning pointers to upstream nodes for each systematic variation.
   std::vector<std::shared_ptr<PrevNodeType>> fPrevNodes;

   /// Column readers per slot (outer dimension), per variation and per input column (inner dimension, std::array).
   std::vector<std::vector<std::array<RColumnReaderBase *, ColumnTypes_t::list_size>>> fInputValues;

   /// The nth flag signals whether the nth input column is a custom column or not.
   std::array<bool, ColumnTypes_t::list_size> fIsDefine;

   /// \brief Creates new filter nodes, one per variation, from the upstream nominal one.
   /// \param nominal The nominal filter
   /// \return The varied filters
   ///
   /// The nominal filter is not included in the return value.
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

   void SetupClass()
   {
      // The column register and names are private members of RActionBase
      const auto &colRegister = GetColRegister();
      const auto &columnNames = GetColumnNames();

      fLoopManager->Register(this);

      for (auto i = 0u; i < columnNames.size(); ++i) {
         auto *define = colRegister.GetDefine(columnNames[i]);
         fIsDefine[i] = define != nullptr;
         if (fIsDefine[i])
            define->MakeVariations(GetVariations());
      }
   }

   /// This constructor takes in input a vector of previous nodes, motivated by the CloneAction logic.
   RVariedAction(std::vector<Helper> &&helpers, const ColumnNames_t &columns,
                 const std::vector<std::shared_ptr<PrevNodeType>> &prevNodes, const RColumnRegister &colRegister)
      : RActionBase(prevNodes[0]->GetLoopManagerUnchecked(), columns, colRegister, prevNodes[0]->GetVariations()),
        fHelpers(std::move(helpers)),
        fPrevNodes(prevNodes),
        fInputValues(GetNSlots())
   {
      SetupClass();
   }

public:
   RVariedAction(std::vector<Helper> &&helpers, const ColumnNames_t &columns, std::shared_ptr<PrevNode> prevNode,
                 const RColumnRegister &colRegister)
      : RActionBase(prevNode->GetLoopManagerUnchecked(), columns, colRegister, prevNode->GetVariations()),
        fHelpers(std::move(helpers)),
        fPrevNodes(MakePrevFilters(prevNode)),
        fInputValues(GetNSlots())
   {
      SetupClass();
   }

   RVariedAction(const RVariedAction &) = delete;
   RVariedAction &operator=(const RVariedAction &) = delete;

   ~RVariedAction() { fLoopManager->Deregister(this); }

   void Initialize() final
   {
      std::for_each(fHelpers.begin(), fHelpers.end(), [](Helper &h) { h.Initialize(); });
   }

   void InitSlot(TTreeReader *r, unsigned int slot) final
   {
      RColumnReadersInfo info{GetColumnNames(), GetColRegister(), fIsDefine.data(), *fLoopManager};

      // get readers for each systematic variation
      for (const auto &variation : GetVariations())
         fInputValues[slot].emplace_back(GetColumnReaders(slot, r, ColumnTypes_t{}, info, variation));

      std::for_each(fHelpers.begin(), fHelpers.end(), [=](Helper &h) { h.InitTask(r, slot); });
   }

   template <typename ColType>
   auto GetValueChecked(unsigned int slot, unsigned int varIdx, std::size_t readerIdx, Long64_t entry) -> ColType &
   {
      if (auto *val = fInputValues[slot][varIdx][readerIdx]->template TryGet<ColType>(entry))
         return *val;

      throw std::out_of_range{"RDataFrame: Varied action (" + fHelpers[0].GetActionName() +
                              ") could not retrieve value for column '" + fColumnNames[readerIdx] + "' for entry " +
                              std::to_string(entry) +
                              ". You can use the DefaultValueFor operation to provide a default value, or "
                              "FilterAvailable/FilterMissing to discard/keep entries with missing values instead."};
   }

   template <typename... ColTypes, std::size_t... ReaderIdxs>
   void CallExec(unsigned int slot, unsigned int varIdx, Long64_t entry, TypeList<ColTypes...>,
                 std::index_sequence<ReaderIdxs...>)
   {
      fHelpers[varIdx].Exec(slot, GetValueChecked<ColTypes>(slot, varIdx, ReaderIdxs, entry)...);
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

   /// Return the partially-updated value connected to the first variation.
   void *PartialUpdate(unsigned int slot) final { return PartialUpdateImpl(slot); }

   /// Return a callback that in turn runs the callbacks of each variation's helper.
   ROOT::RDF::SampleCallback_t GetSampleCallback() final
   {
      if (fHelpers[0].GetSampleCallback()) {
         std::vector<ROOT::RDF::SampleCallback_t> callbacks;
         for (auto &h : fHelpers)
            callbacks.push_back(h.GetSampleCallback());

         auto callEachCallback = [cs = std::move(callbacks)](unsigned int slot, const RSampleInfo &info) {
            for (auto &c : cs)
               c(slot, info);
         };

         return callEachCallback;
      }

      return {};
   }

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

      thisNode->AddDefinedColumns(GetColRegister().GenerateColumnNames());
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

   [[noreturn]] std::unique_ptr<RActionBase> MakeVariedAction(std::vector<void *> &&) final
   {
      throw std::logic_error("Cannot produce a varied action from a varied action.");
   }

   std::unique_ptr<RActionBase> CloneAction(void *typeErasedResults) final
   {
      const auto &vectorOfTypeErasedResults = *reinterpret_cast<const std::vector<void *> *>(typeErasedResults);
      assert(vectorOfTypeErasedResults.size() == fHelpers.size() &&
             "The number of results and the number of helpers are not the same!");

      std::vector<Helper> clonedHelpers;
      clonedHelpers.reserve(fHelpers.size());
      for (std::size_t i = 0; i < fHelpers.size(); i++) {
         clonedHelpers.emplace_back(fHelpers[i].CallMakeNew(vectorOfTypeErasedResults[i]));
      }

      return std::unique_ptr<RVariedAction>(
         new RVariedAction(std::move(clonedHelpers), GetColumnNames(), fPrevNodes, GetColRegister()));
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
