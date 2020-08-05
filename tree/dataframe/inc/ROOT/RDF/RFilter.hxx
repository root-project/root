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

#include "ROOT/RDF/ColumnReaders.hxx"
#include "ROOT/RDF/RCutFlowReport.hxx"
#include "ROOT/RDF/NodesUtils.hxx"
#include "ROOT/RDF/Utils.hxx"
#include "ROOT/RDF/RFilterBase.hxx"
#include "ROOT/RDF/RLoopManager.hxx"
#include "ROOT/RIntegerSequence.hxx"
#include "ROOT/TypeTraits.hxx"
#include "RtypesCore.h"

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

namespace ROOT {

namespace Internal {
namespace RDF {
using namespace ROOT::Detail::RDF;

// fwd decl for RFilter
namespace GraphDrawing {
std::shared_ptr<GraphNode> CreateFilterNode(const RFilterBase *filterPtr);

bool CheckIfDefaultOrDSColumn(const std::string &name, const std::shared_ptr<RCustomColumnBase> &column);

std::shared_ptr<GraphNode>
CreateDefineNode(const std::string &columnName, const RDFDetail::RCustomColumnBase *columnPtr);
} // ns GraphDrawing

} // ns RDF
} // ns Internal

namespace Detail {
namespace RDF {
using namespace ROOT::TypeTraits;
namespace RDFGraphDrawing = ROOT::Internal::RDF::GraphDrawing;

template <typename FilterF, typename PrevDataFrame>
class RFilter final : public RFilterBase {
   using ColumnTypes_t = typename CallableTraits<FilterF>::arg_types;
   using TypeInd_t = std::make_index_sequence<ColumnTypes_t::list_size>;

   FilterF fFilter;
   const ColumnNames_t fColumnNames;
   const std::shared_ptr<PrevDataFrame> fPrevDataPtr;
   PrevDataFrame &fPrevData;
   std::vector<RDFInternal::RDFValueTuple_t<ColumnTypes_t>> fValues;
   /// The nth flag signals whether the nth input column is a custom column or not.
   std::array<bool, ColumnTypes_t::list_size> fIsCustomColumn;

public:
   RFilter(FilterF f, const ColumnNames_t &columns, std::shared_ptr<PrevDataFrame> pd,
           const RDFInternal::RBookedCustomColumns &customColumns, std::string_view name = "")
      : RFilterBase(pd->GetLoopManagerUnchecked(), name, pd->GetLoopManagerUnchecked()->GetNSlots(), customColumns),
        fFilter(std::move(f)), fColumnNames(columns), fPrevDataPtr(std::move(pd)), fPrevData(*fPrevDataPtr),
        fValues(fNSlots), fIsCustomColumn()
   {
      const auto nColumns = fColumnNames.size();
      for (auto i = 0u; i < nColumns; ++i)
         fIsCustomColumn[i] = fCustomColumns.HasName(fColumnNames[i]);
   }

   RFilter(const RFilter &) = delete;
   RFilter &operator=(const RFilter &) = delete;
   // must call Deregister here, before fPrevDataFrame is destroyed,
   // otherwise if fPrevDataFrame is fLoopManager we get a use after delete
   ~RFilter() { fLoopManager->Deregister(this); }

   bool CheckFilters(unsigned int slot, Long64_t entry) final
   {
      if (entry != fLastCheckedEntry[slot]) {
         if (!fPrevData.CheckFilters(slot, entry)) {
            // a filter upstream returned false, cache the result
            fLastResult[slot] = false;
         } else {
            // evaluate this filter, cache the result
            auto passed = CheckFilterHelper(slot, entry, TypeInd_t());
            passed ? ++fAccepted[slot] : ++fRejected[slot];
            fLastResult[slot] = passed;
         }
         fLastCheckedEntry[slot] = entry;
      }
      return fLastResult[slot];
   }

   template <std::size_t... S>
   bool CheckFilterHelper(unsigned int slot, Long64_t entry, std::index_sequence<S...>)
   {
      // silence "unused parameter" warnings in gcc
      (void)slot;
      (void)entry;
      return fFilter(std::get<S>(fValues[slot])->Get(entry)...);
   }

   void InitSlot(TTreeReader *r, unsigned int slot) final
   {
      for (auto &bookedBranch : fCustomColumns.GetColumns())
         bookedBranch.second->InitSlot(r, slot);
      RDFInternal::InitColumnReaders(slot, fValues[slot], r, fColumnNames, fCustomColumns, TypeInd_t(),
                                     fIsCustomColumn);
   }

   // recursive chain of `Report`s
   void Report(ROOT::RDF::RCutFlowReport &rep) const final { PartialReport(rep); }

   void PartialReport(ROOT::RDF::RCutFlowReport &rep) const final
   {
      fPrevData.PartialReport(rep);
      FillReport(rep);
   }

   void StopProcessing() final
   {
      ++fNStopsReceived;
      if (fNStopsReceived == fNChildren)
         fPrevData.StopProcessing();
   }

   void IncrChildrenCount() final
   {
      ++fNChildren;
      // propagate "children activation" upstream. named filters do the propagation via `TriggerChildrenCount`.
      if (fNChildren == 1 && fName.empty())
         fPrevData.IncrChildrenCount();
   }

   void TriggerChildrenCount() final
   {
      R__ASSERT(!fName.empty()); // this method is to only be called on named filters
      fPrevData.IncrChildrenCount();
   }

   virtual void ClearValueReaders(unsigned int slot) final
   {
      RDFInternal::ResetColumnReaders(fValues[slot], TypeInd_t());
   }

   void AddFilterName(std::vector<std::string> &filters)
   {
      fPrevData.AddFilterName(filters);
      auto name = (HasName() ? fName : "Unnamed Filter");
      filters.push_back(name);
   }

   virtual void ClearTask(unsigned int slot) final
   {
      for (auto &column : fCustomColumns.GetColumns()) {
         column.second->ClearValueReaders(slot);
      }

      ClearValueReaders(slot);
   }

   std::shared_ptr<RDFGraphDrawing::GraphNode> GetGraph()
   {
      // Recursively call for the previous node.
      auto prevNode = fPrevData.GetGraph();
      auto prevColumns = prevNode->GetDefinedColumns();

      auto thisNode = RDFGraphDrawing::CreateFilterNode(this);

      /* If the returned node is not new, there is no need to perform any other operation.
       * This is a likely scenario when building the entire graph in which branches share
       * some nodes. */
      if (!thisNode->GetIsNew()) {
         return thisNode;
      }

      auto evaluatedNode = thisNode;
      /* Each column that this node has but the previous hadn't has been defined in between,
       * so it has to be built and appended. */

      for (auto &column : fCustomColumns.GetColumns()) {
         // Even if treated as custom columns by the Dataframe, datasource columns must not be in the graph.
         if (RDFGraphDrawing::CheckIfDefaultOrDSColumn(column.first, column.second))
            continue;
         if (std::find(prevColumns.begin(), prevColumns.end(), column.first) == prevColumns.end()) {
            auto defineNode = RDFGraphDrawing::CreateDefineNode(column.first, column.second.get());
            evaluatedNode->SetPrevNode(defineNode);
            evaluatedNode = defineNode;
         }
      }

      // Keep track of the columns defined up to this point.
      thisNode->AddDefinedColumns(fCustomColumns.GetNames());

      evaluatedNode->SetPrevNode(prevNode);
      return thisNode;
   }
};

} // ns RDF
} // ns Detail
} // ns ROOT

#endif // ROOT_RFILTER
