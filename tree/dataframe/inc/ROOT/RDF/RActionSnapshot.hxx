// Author: Vincenzo Eduardo Padulano CERN 06/2025

/*************************************************************************
 * Copyright (C) 1995-2025, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RACTIONSNAPSHOT
#define ROOT_RACTIONSNAPSHOT

#include "ROOT/RDF/ColumnReaderUtils.hxx"
#include "ROOT/RDF/GraphNode.hxx"
#include "ROOT/RDF/RActionBase.hxx"
#include "ROOT/RDF/RLoopManager.hxx"

#include <cstddef> // std::size_t
#include <memory>
#include <string>
#include <vector>

namespace ROOT::Internal::RDF {

namespace GraphDrawing {
std::shared_ptr<GraphNode> AddDefinesToGraph(std::shared_ptr<GraphNode> node, const RColumnRegister &colRegister,
                                             const std::vector<std::string> &prevNodeDefines,
                                             std::unordered_map<void *, std::shared_ptr<GraphNode>> &visitedMap);
} // namespace GraphDrawing

template <typename Helper, typename PrevNode>
class R__CLING_PTRCHECK(off) RActionSnapshot final : public RActionBase {

   // Template needed to avoid dependency on ActionHelpers.hxx
   Helper fHelper;

   /// Pointer to the previous node in this branch of the computation graph
   std::shared_ptr<PrevNode> fPrevNode;

   /// Column readers per slot and per input column
   std::vector<std::vector<RColumnReaderBase *>> fValues;

   /// The nth flag signals whether the nth input column is a custom column or not.
   std::vector<bool> fIsDefine;

   /// Types of the columns to Snapshot
   std::vector<const std::type_info *> fColTypeIDs;

   ROOT::RDF::SampleCallback_t GetSampleCallback() final { return fHelper.GetSampleCallback(); }

public:
   RActionSnapshot(Helper &&h, const std::vector<std::string> &columns,
                   const std::vector<const std::type_info *> &colTypeIDs, std::shared_ptr<PrevNode> pd,
                   const RColumnRegister &colRegister)
      : RActionBase(pd->GetLoopManagerUnchecked(), columns, colRegister, pd->GetVariations()),
        fHelper(std::forward<Helper>(h)),
        fPrevNode(std::move(pd)),
        fValues(GetNSlots()),
        fColTypeIDs(colTypeIDs)
   {
      fLoopManager->Register(this);

      const auto nColumns = columns.size();
      fIsDefine.reserve(nColumns);
      for (auto i = 0u; i < nColumns; ++i)
         fIsDefine.push_back(colRegister.IsDefineOrAlias(columns[i]));
   }

   RActionSnapshot(const RActionSnapshot &) = delete;
   RActionSnapshot &operator=(const RActionSnapshot &) = delete;
   RActionSnapshot(RActionSnapshot &&) = delete;
   RActionSnapshot &operator=(RActionSnapshot &&) = delete;

   ~RActionSnapshot() final { fLoopManager->Deregister(this); }

   /**
      Retrieve a wrapper to the result of the action that knows how to merge
      with others of the same type.
   */
   std::unique_ptr<ROOT::Detail::RDF::RMergeableValueBase> GetMergeableValue() const final
   {
      return fHelper.GetMergeableValue();
   }

   void Initialize() final { fHelper.Initialize(); }

   void InitSlot(TTreeReader *r, unsigned int slot) final
   {
      fValues[slot] = GetUntypedColumnReaders(slot, r, RActionBase::GetColRegister(), *fLoopManager,
                                              RActionBase::GetColumnNames(), fColTypeIDs);
      fHelper.InitTask(r, slot);
   }

   void *GetValue(unsigned int slot, std::size_t readerIdx, Long64_t entry)
   {
      if (auto *val = fValues[slot][readerIdx]->template TryGet<void>(entry))
         return val;

      throw std::out_of_range{"RDataFrame: Action (" + fHelper.GetActionName() +
                              ") could not retrieve value for column '" + fColumnNames[readerIdx] + "' for entry " +
                              std::to_string(entry) +
                              ". You can use the DefaultValueFor operation to provide a default value, or "
                              "FilterAvailable/FilterMissing to discard/keep entries with missing values instead."};
   }

   void CallExec(unsigned int slot, Long64_t entry)
   {
      std::vector<void *> untypedValues;
      auto nReaders = fValues[slot].size();
      untypedValues.reserve(nReaders);
      for (decltype(nReaders) readerIdx{}; readerIdx < nReaders; readerIdx++)
         untypedValues.push_back(GetValue(slot, readerIdx, entry));

      fHelper.Exec(slot, untypedValues);
   }

   void Run(unsigned int slot, Long64_t entry) final
   {
      // check if entry passes all filters
      if (fPrevNode->CheckFilters(slot, entry))
         CallExec(slot, entry);
   }

   void TriggerChildrenCount() final { fPrevNode->IncrChildrenCount(); }

   /// Clean-up operations to be performed at the end of a task.
   void FinalizeSlot(unsigned int slot) final
   {
      fValues[slot].clear();
      fHelper.CallFinalizeTask(slot);
   }

   /// Clean-up and finalize the action result (e.g. merging slot-local results).
   /// It invokes the helper's Finalize method.
   void Finalize() final
   {
      fHelper.Finalize();
      SetHasRun();
   }

   std::shared_ptr<GraphDrawing::GraphNode>
   GetGraph(std::unordered_map<void *, std::shared_ptr<GraphDrawing::GraphNode>> &visitedMap) final
   {
      auto prevNode = fPrevNode->GetGraph(visitedMap);
      const auto &prevColumns = prevNode->GetDefinedColumns();

      // Action nodes do not need to go through CreateFilterNode: they are never common nodes between multiple branches
      const auto nodeType = HasRun() ? GraphDrawing::ENodeType::kUsedAction : GraphDrawing::ENodeType::kAction;
      auto thisNode = std::make_shared<GraphDrawing::GraphNode>(fHelper.GetActionName(), visitedMap.size(), nodeType);
      visitedMap[(void *)this] = thisNode;

      auto upmostNode = AddDefinesToGraph(thisNode, GetColRegister(), prevColumns, visitedMap);

      thisNode->AddDefinedColumns(GetColRegister().GenerateColumnNames());
      upmostNode->SetPrevNode(prevNode);
      return thisNode;
   }

   /// This method is invoked to update a partial result during the event loop, right before passing the result to a
   /// user-defined callback registered via RResultPtr::RegisterCallback
   void *PartialUpdate(unsigned int slot) final { return fHelper.CallPartialUpdate(slot); }

   [[maybe_unused]] std::unique_ptr<RActionBase> MakeVariedAction(std::vector<void *> && /*results*/) final
   {
      // TODO: Probably we also need an untyped RVariedAction
      throw std::runtime_error("RDataFrame::Snapshot: Snapshot with systematic variations is not supported yet.");
   }

   /**
    * \brief Returns a new action with a cloned helper.
    *
    * \param[in] newResult The result to be filled by the new action (needed to clone the helper).
    * \return A unique pointer to the new action.
    */
   std::unique_ptr<RActionBase> CloneAction(void *newResult) final
   {
      return std::make_unique<RActionSnapshot>(fHelper.CallMakeNew(newResult), GetColumnNames(), fColTypeIDs, fPrevNode,
                                               GetColRegister());
   }
};

} // namespace ROOT::Internal::RDF

#endif // ROOT_RACTIONSNAPSHOT
