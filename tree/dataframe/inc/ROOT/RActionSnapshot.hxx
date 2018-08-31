#ifndef ROOT_RACTIONSNAPSHOT
#define ROOT_RACTIONSNAPSHOT

#include "ROOT/RDFNodes.hxx"
#include "ROOT/RDFActionHelpers.hxx"

namespace ROOT {
namespace Internal {
namespace RDF {

class RTypeErasedColumnValue {
   std::shared_ptr<void> fPtr; // shared_ptr correctly deletes the type-erased object

public:
   template <typename T>
   RTypeErasedColumnValue(std::unique_ptr<RColumnValue<T>> v) : fPtr(std::move(v))
   {
   }

   template <typename T>
   T &Get(ULong64_t e)
   {
      return std::static_pointer_cast<RColumnValue<T>>(fPtr)->Get(e);
   }

   template <typename T>
   RColumnValue<T> *Cast()
   {
      return static_cast<RColumnValue<T> *>(fPtr.get());
   }
};

template <std::size_t... S, typename... ColTypes>
void InitRDFValues(unsigned int slot, std::vector<RTypeErasedColumnValue> &values, TTreeReader *r,
                   const ColumnNames_t &bn, const RBookedCustomColumns &customCols, std::index_sequence<S...>,
                   ROOT::TypeTraits::TypeList<ColTypes...>)
{
   std::array<bool, sizeof...(S)> isTmpColumn;
   for (auto i = 0u; i < isTmpColumn.size(); ++i)
      isTmpColumn[i] = customCols.HasName(bn.at(i));

   using expander = int[];
   (void)expander{(values.emplace_back(std::make_unique<RColumnValue<ColTypes>>()), 0)..., 0};
   (void)expander{(isTmpColumn[S]
                      ? values[S].Cast<ColTypes>()->SetTmpColumn(slot, customCols.GetColumns().at(bn.at(S)).get())
                      : values[S].Cast<ColTypes>()->MakeProxy(r, bn.at(S)),
                   0)...,
                  0};
}

template <std::size_t... S, typename... ColTypes>
void ResetRDFValueTuple(std::vector<RTypeErasedColumnValue> &values, std::index_sequence<S...>,
                        ROOT::TypeTraits::TypeList<ColTypes...>)
{
   using expander = int[];
   (void)expander{(values[S].Cast<ColTypes>()->Reset(), 0)...};
}

template <typename PrevDataFrame, typename... ColTypes>
class RAction<SnapshotHelper<ColTypes...>, PrevDataFrame, ROOT::TypeTraits::TypeList<ColTypes...>> final
   : public RActionBase {
   using ColumnTypes_t = typename SnapshotHelper<ColTypes...>::ColumnTypes_t;
   using TypeInd_t = std::make_index_sequence<ColumnTypes_t::list_size>;

   SnapshotHelper<ColTypes...> fHelper;
   const std::shared_ptr<PrevDataFrame> fPrevDataPtr;
   PrevDataFrame &fPrevData;
   std::vector<std::vector<RTypeErasedColumnValue>> fValues;

public:
   RAction(SnapshotHelper<ColTypes...> &&h, const ColumnNames_t &bl, std::shared_ptr<PrevDataFrame> pd,
           const RBookedCustomColumns &customColumns)
      : RActionBase(pd->GetLoopManagerUnchecked(), pd->GetLoopManagerUnchecked()->GetNSlots(), bl, customColumns),
        fHelper(std::move(h)), fPrevDataPtr(std::move(pd)), fPrevData(*fPrevDataPtr), fValues(fNSlots)
   {
   }

   RAction(const RAction &) = delete;
   RAction &operator=(const RAction &) = delete;

   void Initialize() final { fHelper.Initialize(); }

   void InitSlot(TTreeReader *r, unsigned int slot) final
   {
      for (auto &customCol : fCustomColumns.GetColumns())
         customCol.second->InitSlot(r, slot);

      InitRDFValues(slot, fValues[slot], r, fColumnNames, fCustomColumns, TypeInd_t(), ColumnTypes_t());
      fHelper.InitTask(r, slot);
   }

   void Run(unsigned int slot, Long64_t entry) final
   {
      // check if entry passes all filters
      if (fPrevData.CheckFilters(slot, entry))
         Exec(slot, entry, TypeInd_t());
   }

   template <std::size_t... S>
   void Exec(unsigned int slot, Long64_t entry, std::index_sequence<S...>)
   {
      (void)entry; // avoid bogus 'unused parameter' warning in gcc4.9
      fHelper.Exec(slot, fValues[slot][S].template Get<ColTypes>(entry)...);
   }

   void TriggerChildrenCount() final { fPrevData.IncrChildrenCount(); }

   void FinalizeSlot(unsigned int slot) final
   {
      ClearValueReaders(slot);
      for (auto &column : fCustomColumns.GetColumns()) {
         column.second->ClearValueReaders(slot);
      }
      fHelper.CallFinalizeTask(slot);
   }

   void ClearValueReaders(unsigned int slot) { ResetRDFValueTuple(fValues[slot], TypeInd_t(), ColumnTypes_t()); }

   void Finalize() final
   {
      fHelper.Finalize();
      fHasRun = true;
   }

   std::shared_ptr<RDFGraphDrawing::GraphNode> GetGraph()
   {
      auto prevNode = fPrevData.GetGraph();
      auto prevColumns = prevNode->GetDefinedColumns();

      // Action nodes do not need to ask an helper to create the graph nodes. They are never common nodes between
      // multiple branches
      auto thisNode = std::make_shared<RDFGraphDrawing::GraphNode>(fHelper.GetActionName());
      auto evaluatedNode = thisNode;
      for (auto &column : fCustomColumns.GetColumns()) {
         /* Each column that this node has but the previous hadn't has been defined in between,
          * so it has to be built and appended. */
         if (RDFGraphDrawing::CheckIfDefaultOrDSColumn(column.first, column.second))
            continue;
         if (std::find(prevColumns.begin(), prevColumns.end(), column.first) == prevColumns.end()) {
            auto defineNode = RDFGraphDrawing::CreateDefineNode(column.first, column.second.get());
            evaluatedNode->SetPrevNode(defineNode);
            evaluatedNode = defineNode;
         }
      }

      thisNode->AddDefinedColumns(fCustomColumns.GetNames());
      thisNode->SetAction(HasRun());
      evaluatedNode->SetPrevNode(prevNode);
      return thisNode;
   }
};

template <typename PrevDataFrame, typename... ColTypes>
class RAction<SnapshotHelperMT<ColTypes...>, PrevDataFrame, ROOT::TypeTraits::TypeList<ColTypes...>> final
   : public RActionBase {
   using ColumnTypes_t = typename SnapshotHelperMT<ColTypes...>::ColumnTypes_t;
   using TypeInd_t = std::make_index_sequence<ColumnTypes_t::list_size>;

   SnapshotHelperMT<ColTypes...> fHelper;
   const std::shared_ptr<PrevDataFrame> fPrevDataPtr;
   PrevDataFrame &fPrevData;
   std::vector<std::vector<RTypeErasedColumnValue>> fValues;

public:
   RAction(SnapshotHelperMT<ColTypes...> &&h, const ColumnNames_t &bl, std::shared_ptr<PrevDataFrame> pd,
           const RBookedCustomColumns &customColumns)
      : RActionBase(pd->GetLoopManagerUnchecked(), pd->GetLoopManagerUnchecked()->GetNSlots(), bl, customColumns),
        fHelper(std::move(h)), fPrevDataPtr(std::move(pd)), fPrevData(*fPrevDataPtr), fValues(fNSlots)
   {
   }

   RAction(const RAction &) = delete;
   RAction &operator=(const RAction &) = delete;

   void Initialize() final { fHelper.Initialize(); }

   void InitSlot(TTreeReader *r, unsigned int slot) final
   {
      for (auto &bookedBranch : fCustomColumns.GetColumns())
         bookedBranch.second->InitSlot(r, slot);

      InitRDFValues(slot, fValues[slot], r, fColumnNames, fCustomColumns, TypeInd_t(), ColumnTypes_t());
      fHelper.InitTask(r, slot);
   }

   void Run(unsigned int slot, Long64_t entry) final
   {
      // check if entry passes all filters
      if (fPrevData.CheckFilters(slot, entry))
         Exec(slot, entry, TypeInd_t());
   }

   template <std::size_t... S>
   void Exec(unsigned int slot, Long64_t entry, std::index_sequence<S...>)
   {
      (void)entry; // avoid bogus 'unused parameter' warning in gcc4.9
      fHelper.Exec(slot, fValues[slot][S].template Get<ColTypes>(entry)...);
   }

   void TriggerChildrenCount() final { fPrevData.IncrChildrenCount(); }

   void FinalizeSlot(unsigned int slot) final
   {
      ClearValueReaders(slot);
      for (auto &column : fCustomColumns.GetColumns()) {
         column.second->ClearValueReaders(slot);
      }
      fHelper.CallFinalizeTask(slot);
   }

   void ClearValueReaders(unsigned int slot) { ResetRDFValueTuple(fValues[slot], TypeInd_t(), ColumnTypes_t()); }

   void Finalize() final
   {
      fHelper.Finalize();
      fHasRun = true;
   }

   std::shared_ptr<RDFGraphDrawing::GraphNode> GetGraph()
   {
      auto prevNode = fPrevData.GetGraph();
      auto prevColumns = prevNode->GetDefinedColumns();

      // Action nodes do not need to ask an helper to create the graph nodes. They are never common nodes between
      // multiple branches
      auto thisNode = std::make_shared<RDFGraphDrawing::GraphNode>(fHelper.GetActionName());
      auto evaluatedNode = thisNode;
      for (auto &column : fCustomColumns.GetColumns()) {
         /* Each column that this node has but the previous hadn't has been defined in between,
          * so it has to be built and appended. */
         if (RDFGraphDrawing::CheckIfDefaultOrDSColumn(column.first, column.second))
            continue;
         if (std::find(prevColumns.begin(), prevColumns.end(), column.first) == prevColumns.end()) {
            auto defineNode = RDFGraphDrawing::CreateDefineNode(column.first, column.second.get());
            evaluatedNode->SetPrevNode(defineNode);
            evaluatedNode = defineNode;
         }
      }

      thisNode->AddDefinedColumns(fCustomColumns.GetNames());
      thisNode->SetAction(HasRun());
      evaluatedNode->SetPrevNode(prevNode);
      return thisNode;
   }
   RAction(SnapshotHelperMT<ColTypes...> &&h, const ColumnNames_t &bl, std::shared_ptr<PrevDataFrame> pd)
      : RActionBase(pd->GetLoopManagerUnchecked(), pd->GetLoopManagerUnchecked()->GetNSlots(), bl),
        fHelper(std::move(h)), fPrevDataPtr(std::move(pd)), fPrevData(*fPrevDataPtr), fValues(fNSlots)
   {
   }

};
} // ns RDF
} // ns Internal
} // ns ROOT

#endif
