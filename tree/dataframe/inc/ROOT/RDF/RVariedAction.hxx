// Author: Enrico Guiraud, CERN 11/2021

/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
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
template <typename Helper, typename PrevDataFrame, typename ColumnTypes_t>
class R__CLING_PTRCHECK(off) RVariedAction final : public RActionBase {
   using TypeInd_t = std::make_index_sequence<ColumnTypes_t::list_size>;

   std::vector<Helper> fHelpers; /// Action helpers per variation.
   std::shared_ptr<PrevDataFrame> fPrevNode;

   /// Column readers per slot (outer dimension), per variation and per input column (inner dimension, std::array).
   std::vector<std::vector<std::array<std::unique_ptr<RColumnReaderBase>, ColumnTypes_t::list_size>>> fInputValues;

   /// The nth flag signals whether the nth input column is a custom column or not.
   std::array<bool, ColumnTypes_t::list_size> fIsDefine;

public:
   RVariedAction(std::vector<Helper> &&helpers, const ColumnNames_t &columns, std::shared_ptr<PrevDataFrame> pd,
                 const RColumnRegister &colRegister)
      : RActionBase(pd->GetLoopManagerUnchecked(), columns, colRegister, pd->GetVariations()),
        fHelpers(std::move(helpers)), fPrevNode(std::move(pd)), fInputValues(GetNSlots())
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
   ~RVariedAction()
   {
      // must Deregister objects from the RLoopManager here, before the fPrevDataFrame data member is destroyed:
      // otherwise if fPrevDataFrame is the RLoopManager, it will be destroyed before the calls to Deregister happen.
      RActionBase::GetColRegister().Clear(); // triggers RDefine deregistration
      fLoopManager->Deregister(this);
   }

   void Initialize() final
   {
      std::for_each(fHelpers.begin(), fHelpers.end(), [](Helper &h) { h.Initialize(); });
   }

   void InitSlot(TTreeReader *r, unsigned int slot) final
   {
      RDFInternal::RColumnReadersInfo info{GetColumnNames(), GetColRegister(), fIsDefine.data(),
                                           fLoopManager->GetDSValuePtrs(), fLoopManager->GetDataSource()};

      // get readers for the nominal case + each systematic variation
      fInputValues[slot].emplace_back(MakeColumnReaders(slot, r, ColumnTypes_t{}, info /*, "nominal"*/));
      for (const auto &variation : GetVariations())
         fInputValues[slot].emplace_back(MakeColumnReaders(slot, r, ColumnTypes_t{}, info, variation));

      std::for_each(fHelpers.begin(), fHelpers.end(), [=](Helper &h) { h.InitTask(r, slot); });
   }

   template <typename... ColTypes, std::size_t... S>
   void CallExec(unsigned int slot, Long64_t entry, TypeList<ColTypes...>, std::index_sequence<S...>)
   {
      for (auto varIdx = 0u; varIdx < GetVariations().size() + 1; ++varIdx)
         fHelpers[varIdx].Exec(slot, fInputValues[slot][varIdx][S]->template Get<ColTypes>(entry)...);
      (void)entry;
   }

   void Run(unsigned int slot, Long64_t entry) final
   {
      // check if entry passes all filters
      if (fPrevNode->CheckFilters(slot, entry))
         CallExec(slot, entry, ColumnTypes_t{}, TypeInd_t{});
   }

   void TriggerChildrenCount() final { fPrevNode->IncrChildrenCount(); }

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

   std::shared_ptr<ROOT::Internal::RDF::GraphDrawing::GraphNode> GetGraph() final
   {
      auto prevNode = fPrevNode->GetGraph();
      auto prevColumns = prevNode->GetDefinedColumns();

      // Action nodes do not need to go through CreateFilterNode: they are never common nodes between multiple branches
      auto thisNode = std::make_shared<RDFGraphDrawing::GraphNode>("Varied " + fHelpers[0].GetActionName());

      auto upmostNode = AddDefinesToGraph(thisNode, GetColRegister(), prevColumns);

      thisNode->AddDefinedColumns(GetColRegister().GetNames());
      thisNode->SetAction(HasRun());
      upmostNode->SetPrevNode(prevNode);
      return thisNode;
   }

   [[noreturn]] std::unique_ptr<RMergeableValueBase> GetMergeableValue() const
   {
      throw std::logic_error("Varied actions cannot provide mergeable values");
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
