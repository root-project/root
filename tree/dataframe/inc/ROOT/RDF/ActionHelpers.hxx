/**
 \file ROOT/RDF/ActionHelpers.hxx
 \ingroup dataframe
 \author Enrico Guiraud, CERN
 \author Danilo Piparo, CERN
 \date 2016-12
 \author Vincenzo Eduardo Padulano
 \date 2020-06
*/

/*************************************************************************
 * Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RDFOPERATIONS
#define ROOT_RDFOPERATIONS

#include "Compression.h"
#include "ROOT/RStringView.hxx"
#include "ROOT/RVec.hxx"
#include "ROOT/TBufferMerger.hxx" // for SnapshotHelper
#include "ROOT/RDF/RCutFlowReport.hxx"
#include "ROOT/RDF/RSampleInfo.hxx"
#include "ROOT/RDF/Utils.hxx"
#include "ROOT/RSnapshotOptions.hxx"
#include "ROOT/TypeTraits.hxx"
#include "ROOT/RDF/RDisplay.hxx"
#include "RtypesCore.h"
#include "TBranch.h"
#include "TClassEdit.h"
#include "TClassRef.h"
#include "TDirectory.h"
#include "TError.h" // for R__ASSERT, Warning
#include "TFile.h" // for SnapshotHelper
#include "TH1.h"
#include "TGraph.h"
#include "TGraphAsymmErrors.h"
#include "TLeaf.h"
#include "TObject.h"
#include "TTree.h"
#include "TTreeReader.h" // for SnapshotHelper
#include "ROOT/RDF/RMergeableValue.hxx"

#include <algorithm>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility> // std::index_sequence
#include <vector>
#include <iomanip>
#include <numeric> // std::accumulate in MeanHelper

/// \cond HIDDEN_SYMBOLS

namespace ROOT {
namespace Detail {
namespace RDF {

/// Base class for action helpers, see RInterface::Book() for more information.
template <typename Helper>
class RActionImpl {
public:
   virtual ~RActionImpl() = default;
   // call Helper::FinalizeTask if present, do nothing otherwise
   template <typename T = Helper>
   auto CallFinalizeTask(unsigned int slot) -> decltype(std::declval<T>().FinalizeTask(slot))
   {
      static_cast<Helper *>(this)->FinalizeTask(slot);
   }

   template <typename... Args>
   void CallFinalizeTask(unsigned int, Args...) {}

   template <typename H = Helper>
   auto CallPartialUpdate(unsigned int slot) -> decltype(std::declval<H>().PartialUpdate(slot), (void *)(nullptr))
   {
      return &static_cast<Helper *>(this)->PartialUpdate(slot);
   }

   template <typename... Args>
   [[noreturn]] void *CallPartialUpdate(...)
   {
      throw std::logic_error("This action does not support callbacks!");
   }

   template <typename T = Helper>
   auto CallMakeNew(void *typeErasedResSharedPtr) -> decltype(std::declval<T>().MakeNew(typeErasedResSharedPtr))
   {
      return static_cast<Helper *>(this)->MakeNew(typeErasedResSharedPtr);
   }

   template <typename... Args>
   [[noreturn]] Helper CallMakeNew(void *, Args...)
   {
      const auto &actionName = static_cast<Helper *>(this)->GetActionName();
      throw std::logic_error("The MakeNew method is not implemented for this action helper (" + actionName +
                             "). Cannot Vary its result.");
   }

   // Helper functions for RMergeableValue
   virtual std::unique_ptr<RMergeableValueBase> GetMergeableValue() const
   {
      throw std::logic_error("`GetMergeableValue` is not implemented for this type of action.");
   }

   /// Override this method to register a callback that is executed before the processing a new data sample starts.
   /// The callback will be invoked in the same conditions as with DefinePerSample().
   virtual ROOT::RDF::SampleCallback_t GetSampleCallback() { return {}; }
};

} // namespace RDF
} // namespace Detail

namespace Internal {
namespace RDF {
using namespace ROOT::TypeTraits;
using namespace ROOT::VecOps;
using namespace ROOT::RDF;
using namespace ROOT::Detail::RDF;

using Hist_t = ::TH1D;

class RBranchSet {
   std::vector<TBranch *> fBranches;
   std::vector<std::string> fNames;

public:
   TBranch *Get(const std::string &name) const
   {
      auto it = std::find(fNames.begin(), fNames.end(), name);
      if (it == fNames.end())
         return nullptr;
      return fBranches[std::distance(fNames.begin(), it)];
   }

   void Insert(const std::string &name, TBranch *address)
   {
      if (address == nullptr) {
         throw std::logic_error("Trying to insert a null branch address.");
      }
      if (std::find(fBranches.begin(), fBranches.end(), address) != fBranches.end()) {
         throw std::logic_error("Trying to insert a branch address that's already present.");
      }
      if (std::find(fNames.begin(), fNames.end(), name) != fNames.end()) {
         throw std::logic_error("Trying to insert a branch name that's already present.");
      }
      fNames.emplace_back(name);
      fBranches.emplace_back(address);
   }

   void Clear()
   {
      fBranches.clear();
      fNames.clear();
   }
};

/// The container type for each thread's partial result in an action helper
// We have to avoid to instantiate std::vector<bool> as that makes it impossible to return a reference to one of
// the thread-local results. In addition, a common definition for the type of the container makes it easy to swap
// the type of the underlying container if e.g. we see problems with false sharing of the thread-local results..
template <typename T>
using Results = std::conditional_t<std::is_same<T, bool>::value, std::deque<T>, std::vector<T>>;

template <typename F>
class ForeachSlotHelper : public RActionImpl<ForeachSlotHelper<F>> {
   F fCallable;

public:
   using ColumnTypes_t = RemoveFirstParameter_t<typename CallableTraits<F>::arg_types>;
   ForeachSlotHelper(F &&f) : fCallable(f) {}
   ForeachSlotHelper(ForeachSlotHelper &&) = default;
   ForeachSlotHelper(const ForeachSlotHelper &) = delete;

   void InitTask(TTreeReader *, unsigned int) {}

   template <typename... Args>
   void Exec(unsigned int slot, Args &&... args)
   {
      // check that the decayed types of Args are the same as the branch types
      static_assert(std::is_same<TypeList<std::decay_t<Args>...>, ColumnTypes_t>::value, "");
      fCallable(slot, std::forward<Args>(args)...);
   }

   void Initialize() { /* noop */}

   void Finalize() { /* noop */}

   std::string GetActionName() { return "ForeachSlot"; }
};

class CountHelper : public RActionImpl<CountHelper> {
   const std::shared_ptr<ULong64_t> fResultCount;
   Results<ULong64_t> fCounts;

public:
   using ColumnTypes_t = TypeList<>;
   CountHelper(const std::shared_ptr<ULong64_t> &resultCount, const unsigned int nSlots);
   CountHelper(CountHelper &&) = default;
   CountHelper(const CountHelper &) = delete;
   void InitTask(TTreeReader *, unsigned int) {}
   void Exec(unsigned int slot);
   void Initialize() { /* noop */}
   void Finalize();

   // Helper functions for RMergeableValue
   std::unique_ptr<RMergeableValueBase> GetMergeableValue() const final
   {
      return std::make_unique<RMergeableCount>(*fResultCount);
   }

   ULong64_t &PartialUpdate(unsigned int slot);

   std::string GetActionName() { return "Count"; }

   CountHelper MakeNew(void *newResult)
   {
      auto &result = *static_cast<std::shared_ptr<ULong64_t> *>(newResult);
      return CountHelper(result, fCounts.size());
   }
};

template <typename ProxiedVal_t>
class ReportHelper : public RActionImpl<ReportHelper<ProxiedVal_t>> {
   const std::shared_ptr<RCutFlowReport> fReport;
   // Here we have a weak pointer since we need to keep track of the validity
   // of the proxied node. It can happen that the user does not trigger the
   // event loop by looking into the RResultPtr and the chain goes out of scope
   // before the Finalize method is invoked.
   std::weak_ptr<ProxiedVal_t> fProxiedWPtr;
   bool fReturnEmptyReport;

public:
   using ColumnTypes_t = TypeList<>;
   ReportHelper(const std::shared_ptr<RCutFlowReport> &report, const std::shared_ptr<ProxiedVal_t> &pp, bool emptyRep)
      : fReport(report), fProxiedWPtr(pp), fReturnEmptyReport(emptyRep){};
   ReportHelper(ReportHelper &&) = default;
   ReportHelper(const ReportHelper &) = delete;
   void InitTask(TTreeReader *, unsigned int) {}
   void Exec(unsigned int /* slot */) {}
   void Initialize() { /* noop */}
   void Finalize()
   {
      // We need the weak_ptr in order to avoid crashes at tear down
      if (!fReturnEmptyReport && !fProxiedWPtr.expired())
         fProxiedWPtr.lock()->Report(*fReport);
   }

   std::string GetActionName() { return "Report"; }

   // TODO implement MakeNew. Requires some smartness in passing the appropriate previous node.
};

class FillHelper : public RActionImpl<FillHelper> {
   // this sets a total initial size of 16 MB for the buffers (can increase)
   static constexpr unsigned int fgTotalBufSize = 2097152;
   using BufEl_t = double;
   using Buf_t = std::vector<BufEl_t>;

   std::vector<Buf_t> fBuffers;
   std::vector<Buf_t> fWBuffers;
   const std::shared_ptr<Hist_t> fResultHist;
   unsigned int fNSlots;
   unsigned int fBufSize;
   /// Histograms containing "snapshots" of partial results. Non-null only if a registered callback requires it.
   Results<std::unique_ptr<Hist_t>> fPartialHists;
   Buf_t fMin;
   Buf_t fMax;

   void UpdateMinMax(unsigned int slot, double v);

public:
   FillHelper(const std::shared_ptr<Hist_t> &h, const unsigned int nSlots);
   FillHelper(FillHelper &&) = default;
   FillHelper(const FillHelper &) = delete;
   void InitTask(TTreeReader *, unsigned int) {}
   void Exec(unsigned int slot, double v);
   void Exec(unsigned int slot, double v, double w);

   template <typename T, std::enable_if_t<IsDataContainer<T>::value || std::is_same<T, std::string>::value, int> = 0>
   void Exec(unsigned int slot, const T &vs)
   {
      auto &thisBuf = fBuffers[slot];
      // range-based for results in warnings on some compilers due to vector<bool>'s custom reference type
      for (auto v = vs.begin(); v != vs.end(); ++v) {
         UpdateMinMax(slot, *v);
         thisBuf.emplace_back(*v); // TODO: Can be optimised in case T == BufEl_t
      }
   }

   template <typename T, typename W, std::enable_if_t<IsDataContainer<T>::value && IsDataContainer<W>::value, int> = 0>
   void Exec(unsigned int slot, const T &vs, const W &ws)
   {
      auto &thisBuf = fBuffers[slot];

      for (auto &v : vs) {
         UpdateMinMax(slot, v);
         thisBuf.emplace_back(v);
      }

      auto &thisWBuf = fWBuffers[slot];
      for (auto &w : ws) {
         thisWBuf.emplace_back(w); // TODO: Can be optimised in case T == BufEl_t
      }
   }

   template <typename T, typename W, std::enable_if_t<IsDataContainer<T>::value && !IsDataContainer<W>::value, int> = 0>
   void Exec(unsigned int slot, const T &vs, const W w)
   {
      auto &thisBuf = fBuffers[slot];
      for (auto &v : vs) {
         UpdateMinMax(slot, v);
         thisBuf.emplace_back(v); // TODO: Can be optimised in case T == BufEl_t
      }

      auto &thisWBuf = fWBuffers[slot];
      thisWBuf.insert(thisWBuf.end(), vs.size(), w);
   }

   // ROOT-10092: Filling with a scalar as first column and a collection as second is not supported
   template <typename T, typename W, std::enable_if_t<IsDataContainer<W>::value && !IsDataContainer<T>::value, int> = 0>
   void Exec(unsigned int, const T &, const W &)
   {
      throw std::runtime_error(
        "Cannot fill object if the type of the first column is a scalar and the one of the second a container.");
   }

   Hist_t &PartialUpdate(unsigned int);

   void Initialize() { /* noop */}

   void Finalize();

   // Helper functions for RMergeableValue
   std::unique_ptr<RMergeableValueBase> GetMergeableValue() const final
   {
      return std::make_unique<RMergeableFill<Hist_t>>(*fResultHist);
   }

   std::string GetActionName()
   {
      return std::string(fResultHist->IsA()->GetName()) + "<BR/>" + std::string(fResultHist->GetName());
   }

   FillHelper MakeNew(void *newResult)
   {
      auto &result = *static_cast<std::shared_ptr<Hist_t> *>(newResult);
      result->Reset();
      result->SetDirectory(nullptr);
      return FillHelper(result, fNSlots);
   }
};

extern template void FillHelper::Exec(unsigned int, const std::vector<float> &);
extern template void FillHelper::Exec(unsigned int, const std::vector<double> &);
extern template void FillHelper::Exec(unsigned int, const std::vector<char> &);
extern template void FillHelper::Exec(unsigned int, const std::vector<int> &);
extern template void FillHelper::Exec(unsigned int, const std::vector<unsigned int> &);
extern template void FillHelper::Exec(unsigned int, const std::vector<float> &, const std::vector<float> &);
extern template void FillHelper::Exec(unsigned int, const std::vector<double> &, const std::vector<double> &);
extern template void FillHelper::Exec(unsigned int, const std::vector<char> &, const std::vector<char> &);
extern template void FillHelper::Exec(unsigned int, const std::vector<int> &, const std::vector<int> &);
extern template void
FillHelper::Exec(unsigned int, const std::vector<unsigned int> &, const std::vector<unsigned int> &);

template <typename HIST = Hist_t>
class FillParHelper : public RActionImpl<FillParHelper<HIST>> {
   std::vector<HIST *> fObjects;

   void UnsetDirectoryIfPossible(TH1 *h) {
      h->SetDirectory(nullptr);
   }

   void UnsetDirectoryIfPossible(...) {}

   // Merge overload for types with Merge(TCollection*), like TH1s
   template <typename H, typename = std::enable_if_t<std::is_base_of<TObject, H>::value, int>>
   auto Merge(std::vector<H *> &objs, int /*toincreaseoverloadpriority*/)
      -> decltype(objs[0]->Merge((TCollection *)nullptr), void())
   {
      TList l;
      for (auto it = ++objs.begin(); it != objs.end(); ++it)
         l.Add(*it);
      objs[0]->Merge(&l);
   }

   // Merge overload for types with Merge(const std::vector&)
   template <typename H>
   auto Merge(std::vector<H *> &objs, double /*toloweroverloadpriority*/)
      -> decltype(objs[0]->Merge(std::vector<HIST *>{}), void())
   {
      objs[0]->Merge({++objs.begin(), objs.end()});
   }

   // Merge overload to error out in case no valid HIST::Merge method was detected
   template <typename T>
   void Merge(T, ...)
   {
      static_assert(sizeof(T) < 0,
                    "The type passed to Fill does not provide a Merge(TCollection*) or Merge(const std::vector&) method.");
   }

   // class which wraps a pointer and implements a no-op increment operator
   template <typename T>
   class ScalarConstIterator {
      const T *obj_;

   public:
      ScalarConstIterator(const T *obj) : obj_(obj) {}
      const T &operator*() const { return *obj_; }
      ScalarConstIterator<T> &operator++() { return *this; }
   };

   // helper functions which provide one implementation for scalar types and another for containers
   // TODO these could probably all be replaced by inlined lambdas and/or constexpr if statements
   // in c++17 or later

   // return unchanged value for scalar
   template <typename T, typename std::enable_if<!IsDataContainer<T>::value, int>::type = 0>
   ScalarConstIterator<T> MakeBegin(const T &val)
   {
      return ScalarConstIterator<T>(&val);
   }

   // return iterator to beginning of container
   template <typename T, typename std::enable_if<IsDataContainer<T>::value, int>::type = 0>
   auto MakeBegin(const T &val)
   {
      return std::begin(val);
   }

   // return 1 for scalars
   template <typename T, typename std::enable_if<!IsDataContainer<T>::value, int>::type = 0>
   std::size_t GetSize(const T &)
   {
      return 1;
   }

   // return container size
   template <typename T, typename std::enable_if<IsDataContainer<T>::value, int>::type = 0>
   std::size_t GetSize(const T &val)
   {
#if __cplusplus >= 201703L
      return std::size(val);
#else
      return val.size();
#endif
   }

   template <std::size_t ColIdx, typename End_t, typename... Its>
   void ExecLoop(unsigned int slot, End_t end, Its... its)
   {
      auto *thisSlotH = fObjects[slot];
      // loop increments all of the iterators while leaving scalars unmodified
      // TODO this could be simplified with fold expressions or std::apply in C++17
      auto nop = [](auto &&...) {};
      for (; GetNthElement<ColIdx>(its...) != end; nop(++its...)) {
         thisSlotH->Fill(*its...);
      }
   }

public:
   FillParHelper(FillParHelper &&) = default;
   FillParHelper(const FillParHelper &) = delete;

   FillParHelper(const std::shared_ptr<HIST> &h, const unsigned int nSlots) : fObjects(nSlots, nullptr)
   {
      fObjects[0] = h.get();
      // Initialize all other slots
      for (unsigned int i = 1; i < nSlots; ++i) {
         fObjects[i] = new HIST(*fObjects[0]);
         UnsetDirectoryIfPossible(fObjects[i]);
      }
   }

   void InitTask(TTreeReader *, unsigned int) {}

   // no container arguments
   template <typename... ValTypes,
             typename std::enable_if<!Disjunction<IsDataContainer<ValTypes>...>::value, int>::type = 0>
   void Exec(unsigned int slot, const ValTypes &...x)
   {
      fObjects[slot]->Fill(x...);
   }

   // at least one container argument
   template <typename... Xs, typename std::enable_if<Disjunction<IsDataContainer<Xs>...>::value, int>::type = 0>
   void Exec(unsigned int slot, const Xs &...xs)
   {
      // array of bools keeping track of which inputs are containers
      constexpr std::array<bool, sizeof...(Xs)> isContainer{IsDataContainer<Xs>::value...};

      // index of the first container input
      constexpr std::size_t colidx = FindIdxTrue(isContainer);
      // if this happens, there is a bug in the implementation
      static_assert(colidx < sizeof...(Xs), "Error: index of collection-type argument not found.");

      // get the end iterator to the first container
      auto const xrefend = std::end(GetNthElement<colidx>(xs...));

      // array of container sizes (1 for scalars)
      std::array<std::size_t, sizeof...(xs)> sizes = {{GetSize(xs)...}};

      for (std::size_t i = 0; i < sizeof...(xs); ++i) {
         if (isContainer[i] && sizes[i] != sizes[colidx]) {
            throw std::runtime_error("Cannot fill histogram with values in containers of different sizes.");
         }
      }

      ExecLoop<colidx>(slot, xrefend, MakeBegin(xs)...);
   }

   void Initialize() { /* noop */}

   void Finalize()
   {
      if (fObjects.size() == 1)
         return;

      Merge(fObjects, /*toselectcorrectoverload=*/0);

      // delete the copies we created for the slots other than the first
      for (auto it = ++fObjects.begin(); it != fObjects.end(); ++it)
         delete *it;
   }

   HIST &PartialUpdate(unsigned int slot) { return *fObjects[slot]; }

   // Helper functions for RMergeableValue
   std::unique_ptr<RMergeableValueBase> GetMergeableValue() const final
   {
      return std::make_unique<RMergeableFill<HIST>>(*fObjects[0]);
   }

   // if the fObjects vector type is derived from TObject, return the name of the object
   template <typename T = HIST, typename std::enable_if<std::is_base_of<TObject, T>::value>::type * = nullptr>
   std::string GetActionName()
   {
      return std::string(fObjects[0]->IsA()->GetName()) + "<BR/>" + std::string(fObjects[0]->GetName());
   }

   // if fObjects is not derived from TObject, indicate it is some other object
   template <typename T = HIST, typename std::enable_if<!std::is_base_of<TObject, T>::value>::type * = nullptr>
   std::string GetActionName()
   {
      return "Fill custom object";
   }

   // generic objects might not have a Reset method, in which case we do the safe thing and disable MakeNew:
   // it would be hard to guarantee that the object copied from the original action is in a clean state, it
   // might have been copied _after_ the event loop that filled it already happened.
   template <typename H = HIST, typename = decltype(std::declval<H>().Reset())>
   FillParHelper MakeNew(void *newResult)
   {
      auto &result = *static_cast<std::shared_ptr<H> *>(newResult);
      result->Reset();
      UnsetDirectoryIfPossible(result.get());
      return FillParHelper(result, fObjects.size());
   }
};

class FillTGraphHelper : public ROOT::Detail::RDF::RActionImpl<FillTGraphHelper> {
public:
   using Result_t = ::TGraph;

private:
   std::vector<::TGraph *> fGraphs;

public:
   FillTGraphHelper(FillTGraphHelper &&) = default;
   FillTGraphHelper(const FillTGraphHelper &) = delete;

   // The last parameter is always false, as at the moment there is no way to propagate the parameter from the user to
   // this method
   FillTGraphHelper(const std::shared_ptr<::TGraph> &g, const unsigned int nSlots) : fGraphs(nSlots, nullptr)
   {
      fGraphs[0] = g.get();
      // Initialize all other slots
      for (unsigned int i = 1; i < nSlots; ++i) {
         fGraphs[i] = new TGraph(*fGraphs[0]);
      }
   }

   void Initialize() {}
   void InitTask(TTreeReader *, unsigned int) {}

   // case: both types are container types
   template <typename X0, typename X1,
             std::enable_if_t<IsDataContainer<X0>::value && IsDataContainer<X1>::value, int> = 0>
   void Exec(unsigned int slot, const X0 &x0s, const X1 &x1s)
   {
      if (x0s.size() != x1s.size()) {
         throw std::runtime_error("Cannot fill Graph with values in containers of different sizes.");
      }
      auto *thisSlotG = fGraphs[slot];
      auto x0sIt = std::begin(x0s);
      const auto x0sEnd = std::end(x0s);
      auto x1sIt = std::begin(x1s);
      for (; x0sIt != x0sEnd; x0sIt++, x1sIt++) {
         thisSlotG->SetPoint(thisSlotG->GetN(), *x0sIt, *x1sIt);
      }
   }

   // case: both types are non-container types, e.g. scalars
   template <typename X0, typename X1,
             std::enable_if_t<!IsDataContainer<X0>::value && !IsDataContainer<X1>::value, int> = 0>
   void Exec(unsigned int slot, X0 x0, X1 x1)
   {
      auto thisSlotG = fGraphs[slot];
      thisSlotG->SetPoint(thisSlotG->GetN(), x0, x1);
   }

   // case: types are combination of containers and non-containers
   // this is not supported, error out
   template <typename X0, typename X1, typename... ExtraArgsToLowerPriority>
   void Exec(unsigned int, X0, X1, ExtraArgsToLowerPriority...)
   {
      throw std::runtime_error("Graph was applied to a mix of scalar values and collections. This is not supported.");
   }

   void Finalize()
   {
      const auto nSlots = fGraphs.size();
      auto resGraph = fGraphs[0];
      TList l;
      l.SetOwner(); // The list will free the memory associated to its elements upon destruction
      for (unsigned int slot = 1; slot < nSlots; ++slot) {
         l.Add(fGraphs[slot]);
      }
      resGraph->Merge(&l);
   }

   // Helper functions for RMergeableValue
   std::unique_ptr<RMergeableValueBase> GetMergeableValue() const final
   {
      return std::make_unique<RMergeableFill<Result_t>>(*fGraphs[0]);
   }

   std::string GetActionName() { return "Graph"; }

   Result_t &PartialUpdate(unsigned int slot) { return *fGraphs[slot]; }

   FillTGraphHelper MakeNew(void *newResult)
   {
      auto &result = *static_cast<std::shared_ptr<TGraph> *>(newResult);
      result->Set(0);
      return FillTGraphHelper(result, fGraphs.size());
   }
};

class FillTGraphAsymmErrorsHelper : public ROOT::Detail::RDF::RActionImpl<FillTGraphAsymmErrorsHelper> {
public:
   using Result_t = ::TGraphAsymmErrors;

private:
   std::vector<::TGraphAsymmErrors *> fGraphAsymmErrors;

public:
   FillTGraphAsymmErrorsHelper(FillTGraphAsymmErrorsHelper &&) = default;
   FillTGraphAsymmErrorsHelper(const FillTGraphAsymmErrorsHelper &) = delete;

   FillTGraphAsymmErrorsHelper(const std::shared_ptr<::TGraphAsymmErrors> &g, const unsigned int nSlots)
      : fGraphAsymmErrors(nSlots, nullptr)
   {
      fGraphAsymmErrors[0] = g.get();
      // Initialize all other slots
      for (unsigned int i = 1; i < nSlots; ++i) {
         fGraphAsymmErrors[i] = new TGraphAsymmErrors(*fGraphAsymmErrors[0]);
      }
   }

   void Initialize() {}
   void InitTask(TTreeReader *, unsigned int) {}

   // case: all types are container types
   template <
      typename X, typename Y, typename EXL, typename EXH, typename EYL, typename EYH,
      std::enable_if_t<IsDataContainer<X>::value && IsDataContainer<Y>::value && IsDataContainer<EXL>::value &&
                          IsDataContainer<EXH>::value && IsDataContainer<EYL>::value && IsDataContainer<EYH>::value,
                       int> = 0>
   void
   Exec(unsigned int slot, const X &xs, const Y &ys, const EXL &exls, const EXH &exhs, const EYL &eyls, const EYH &eyhs)
   {
      if ((xs.size() != ys.size()) || (xs.size() != exls.size()) || (xs.size() != exhs.size()) ||
          (xs.size() != eyls.size()) || (xs.size() != eyhs.size())) {
         throw std::runtime_error("Cannot fill GraphAsymmErrors with values in containers of different sizes.");
      }
      auto *thisSlotG = fGraphAsymmErrors[slot];
      auto xsIt = std::begin(xs);
      auto ysIt = std::begin(ys);
      auto exlsIt = std::begin(exls);
      auto exhsIt = std::begin(exhs);
      auto eylsIt = std::begin(eyls);
      auto eyhsIt = std::begin(eyhs);
      while (xsIt != std::end(xs)) {
         const auto n = thisSlotG->GetN(); // must use the same `n` for SetPoint and SetPointError
         thisSlotG->SetPoint(n, *xsIt++, *ysIt++);
         thisSlotG->SetPointError(n, *exlsIt++, *exhsIt++, *eylsIt++, *eyhsIt++);
      }
   }

   // case: all types are non-container types, e.g. scalars
   template <
      typename X, typename Y, typename EXL, typename EXH, typename EYL, typename EYH,
      std::enable_if_t<!IsDataContainer<X>::value && !IsDataContainer<Y>::value && !IsDataContainer<EXL>::value &&
                          !IsDataContainer<EXH>::value && !IsDataContainer<EYL>::value && !IsDataContainer<EYH>::value,
                       int> = 0>
   void Exec(unsigned int slot, X x, Y y, EXL exl, EXH exh, EYL eyl, EYH eyh)
   {
      auto thisSlotG = fGraphAsymmErrors[slot];
      const auto n = thisSlotG->GetN();
      thisSlotG->SetPoint(n, x, y);
      thisSlotG->SetPointError(n, exl, exh, eyl, eyh);
   }

   // case: types are combination of containers and non-containers
   // this is not supported, error out
   template <typename X, typename Y, typename EXL, typename EXH, typename EYL, typename EYH,
             typename... ExtraArgsToLowerPriority>
   void Exec(unsigned int, X, Y, EXL, EXH, EYL, EYH, ExtraArgsToLowerPriority...)
   {
      throw std::runtime_error(
         "GraphAsymmErrors was applied to a mix of scalar values and collections. This is not supported.");
   }

   void Finalize()
   {
      const auto nSlots = fGraphAsymmErrors.size();
      auto resGraphAsymmErrors = fGraphAsymmErrors[0];
      TList l;
      l.SetOwner(); // The list will free the memory associated to its elements upon destruction
      for (unsigned int slot = 1; slot < nSlots; ++slot) {
         l.Add(fGraphAsymmErrors[slot]);
      }
      resGraphAsymmErrors->Merge(&l);
   }

   // Helper functions for RMergeableValue
   std::unique_ptr<RMergeableValueBase> GetMergeableValue() const final
   {
      return std::make_unique<RMergeableFill<Result_t>>(*fGraphAsymmErrors[0]);
   }

   std::string GetActionName() { return "GraphAsymmErrors"; }

   Result_t &PartialUpdate(unsigned int slot) { return *fGraphAsymmErrors[slot]; }
};

// In case of the take helper we have 4 cases:
// 1. The column is not an RVec, the collection is not a vector
// 2. The column is not an RVec, the collection is a vector
// 3. The column is an RVec, the collection is not a vector
// 4. The column is an RVec, the collection is a vector

template <typename V, typename COLL>
void FillColl(V&& v, COLL& c) {
   c.emplace_back(v);
}

// Use push_back for bool since some compilers do not support emplace_back.
template <typename COLL>
void FillColl(bool v, COLL& c) {
   c.push_back(v);
}

// Case 1.: The column is not an RVec, the collection is not a vector
// No optimisations, no transformations: just copies.
template <typename RealT_t, typename T, typename COLL>
class TakeHelper : public RActionImpl<TakeHelper<RealT_t, T, COLL>> {
   Results<std::shared_ptr<COLL>> fColls;

public:
   using ColumnTypes_t = TypeList<T>;
   TakeHelper(const std::shared_ptr<COLL> &resultColl, const unsigned int nSlots)
   {
      fColls.emplace_back(resultColl);
      for (unsigned int i = 1; i < nSlots; ++i)
         fColls.emplace_back(std::make_shared<COLL>());
   }
   TakeHelper(TakeHelper &&);
   TakeHelper(const TakeHelper &) = delete;

   void InitTask(TTreeReader *, unsigned int) {}

   void Exec(unsigned int slot, T &v) { FillColl(v, *fColls[slot]); }

   void Initialize() { /* noop */}

   void Finalize()
   {
      auto rColl = fColls[0];
      for (unsigned int i = 1; i < fColls.size(); ++i) {
         const auto &coll = fColls[i];
         const auto end = coll->end();
         // Use an explicit loop here to prevent compiler warnings introduced by
         // clang's range-based loop analysis and vector<bool> references.
         for (auto j = coll->begin(); j != end; j++) {
            FillColl(*j, *rColl);
         }
      }
   }

   COLL &PartialUpdate(unsigned int slot) { return *fColls[slot].get(); }

   std::string GetActionName() { return "Take"; }

   TakeHelper MakeNew(void *newResult)
   {
      auto &result = *static_cast<std::shared_ptr<COLL> *>(newResult);
      result->clear();
      return TakeHelper(result, fColls.size());
   }
};

// Case 2.: The column is not an RVec, the collection is a vector
// Optimisations, no transformations: just copies.
template <typename RealT_t, typename T>
class TakeHelper<RealT_t, T, std::vector<T>> : public RActionImpl<TakeHelper<RealT_t, T, std::vector<T>>> {
   Results<std::shared_ptr<std::vector<T>>> fColls;

public:
   using ColumnTypes_t = TypeList<T>;
   TakeHelper(const std::shared_ptr<std::vector<T>> &resultColl, const unsigned int nSlots)
   {
      fColls.emplace_back(resultColl);
      for (unsigned int i = 1; i < nSlots; ++i) {
         auto v = std::make_shared<std::vector<T>>();
         v->reserve(1024);
         fColls.emplace_back(v);
      }
   }
   TakeHelper(TakeHelper &&);
   TakeHelper(const TakeHelper &) = delete;

   void InitTask(TTreeReader *, unsigned int) {}

   void Exec(unsigned int slot, T &v) { FillColl(v, *fColls[slot]); }

   void Initialize() { /* noop */}

   // This is optimised to treat vectors
   void Finalize()
   {
      ULong64_t totSize = 0;
      for (auto &coll : fColls)
         totSize += coll->size();
      auto rColl = fColls[0];
      rColl->reserve(totSize);
      for (unsigned int i = 1; i < fColls.size(); ++i) {
         auto &coll = fColls[i];
         rColl->insert(rColl->end(), coll->begin(), coll->end());
      }
   }

   std::vector<T> &PartialUpdate(unsigned int slot) { return *fColls[slot]; }

   std::string GetActionName() { return "Take"; }

   TakeHelper MakeNew(void *newResult)
   {
      auto &result = *static_cast<std::shared_ptr<std::vector<T>> *>(newResult);
      result->clear();
      return TakeHelper(result, fColls.size());
   }
};

// Case 3.: The column is a RVec, the collection is not a vector
// No optimisations, transformations from RVecs to vectors
template <typename RealT_t, typename COLL>
class TakeHelper<RealT_t, RVec<RealT_t>, COLL> : public RActionImpl<TakeHelper<RealT_t, RVec<RealT_t>, COLL>> {
   Results<std::shared_ptr<COLL>> fColls;

public:
   using ColumnTypes_t = TypeList<RVec<RealT_t>>;
   TakeHelper(const std::shared_ptr<COLL> &resultColl, const unsigned int nSlots)
   {
      fColls.emplace_back(resultColl);
      for (unsigned int i = 1; i < nSlots; ++i)
         fColls.emplace_back(std::make_shared<COLL>());
   }
   TakeHelper(TakeHelper &&);
   TakeHelper(const TakeHelper &) = delete;

   void InitTask(TTreeReader *, unsigned int) {}

   void Exec(unsigned int slot, RVec<RealT_t> av) { fColls[slot]->emplace_back(av.begin(), av.end()); }

   void Initialize() { /* noop */}

   void Finalize()
   {
      auto rColl = fColls[0];
      for (unsigned int i = 1; i < fColls.size(); ++i) {
         auto &coll = fColls[i];
         for (auto &v : *coll) {
            rColl->emplace_back(v);
         }
      }
   }

   std::string GetActionName() { return "Take"; }

   TakeHelper MakeNew(void *newResult)
   {
      auto &result = *static_cast<std::shared_ptr<COLL> *>(newResult);
      result->clear();
      return TakeHelper(result, fColls.size());
   }
};

// Case 4.: The column is an RVec, the collection is a vector
// Optimisations, transformations from RVecs to vectors
template <typename RealT_t>
class TakeHelper<RealT_t, RVec<RealT_t>, std::vector<RealT_t>>
   : public RActionImpl<TakeHelper<RealT_t, RVec<RealT_t>, std::vector<RealT_t>>> {

   Results<std::shared_ptr<std::vector<std::vector<RealT_t>>>> fColls;

public:
   using ColumnTypes_t = TypeList<RVec<RealT_t>>;
   TakeHelper(const std::shared_ptr<std::vector<std::vector<RealT_t>>> &resultColl, const unsigned int nSlots)
   {
      fColls.emplace_back(resultColl);
      for (unsigned int i = 1; i < nSlots; ++i) {
         auto v = std::make_shared<std::vector<RealT_t>>();
         v->reserve(1024);
         fColls.emplace_back(v);
      }
   }
   TakeHelper(TakeHelper &&);
   TakeHelper(const TakeHelper &) = delete;

   void InitTask(TTreeReader *, unsigned int) {}

   void Exec(unsigned int slot, RVec<RealT_t> av) { fColls[slot]->emplace_back(av.begin(), av.end()); }

   void Initialize() { /* noop */}

   // This is optimised to treat vectors
   void Finalize()
   {
      ULong64_t totSize = 0;
      for (auto &coll : fColls)
         totSize += coll->size();
      auto rColl = fColls[0];
      rColl->reserve(totSize);
      for (unsigned int i = 1; i < fColls.size(); ++i) {
         auto &coll = fColls[i];
         rColl->insert(rColl->end(), coll->begin(), coll->end());
      }
   }

   std::string GetActionName() { return "Take"; }

   TakeHelper MakeNew(void *newResult)
   {
      auto &result = *static_cast<typename decltype(fColls)::value_type *>(newResult);
      result->clear();
      return TakeHelper(result, fColls.size());
   }
};

// Extern templates for TakeHelper
// NOTE: The move-constructor of specializations declared as extern templates
// must be defined out of line, otherwise cling fails to find its symbol.
template <typename RealT_t, typename T, typename COLL>
TakeHelper<RealT_t, T, COLL>::TakeHelper(TakeHelper<RealT_t, T, COLL> &&) = default;
template <typename RealT_t, typename T>
TakeHelper<RealT_t, T, std::vector<T>>::TakeHelper(TakeHelper<RealT_t, T, std::vector<T>> &&) = default;
template <typename RealT_t, typename COLL>
TakeHelper<RealT_t, RVec<RealT_t>, COLL>::TakeHelper(TakeHelper<RealT_t, RVec<RealT_t>, COLL> &&) = default;
template <typename RealT_t>
TakeHelper<RealT_t, RVec<RealT_t>, std::vector<RealT_t>>::TakeHelper(TakeHelper<RealT_t, RVec<RealT_t>, std::vector<RealT_t>> &&) = default;

// External templates are disabled for gcc5 since this version wrongly omits the C++11 ABI attribute
#if __GNUC__ > 5
extern template class TakeHelper<bool, bool, std::vector<bool>>;
extern template class TakeHelper<unsigned int, unsigned int, std::vector<unsigned int>>;
extern template class TakeHelper<unsigned long, unsigned long, std::vector<unsigned long>>;
extern template class TakeHelper<unsigned long long, unsigned long long, std::vector<unsigned long long>>;
extern template class TakeHelper<int, int, std::vector<int>>;
extern template class TakeHelper<long, long, std::vector<long>>;
extern template class TakeHelper<long long, long long, std::vector<long long>>;
extern template class TakeHelper<float, float, std::vector<float>>;
extern template class TakeHelper<double, double, std::vector<double>>;
#endif


template <typename ResultType>
class MinHelper : public RActionImpl<MinHelper<ResultType>> {
   const std::shared_ptr<ResultType> fResultMin;
   Results<ResultType> fMins;

public:
   MinHelper(MinHelper &&) = default;
   MinHelper(const std::shared_ptr<ResultType> &minVPtr, const unsigned int nSlots)
      : fResultMin(minVPtr), fMins(nSlots, std::numeric_limits<ResultType>::max())
   {
   }

   void Exec(unsigned int slot, ResultType v) { fMins[slot] = std::min(v, fMins[slot]); }

   void InitTask(TTreeReader *, unsigned int) {}

   template <typename T, std::enable_if_t<IsDataContainer<T>::value, int> = 0>
   void Exec(unsigned int slot, const T &vs)
   {
      for (auto &&v : vs)
         fMins[slot] = std::min(static_cast<ResultType>(v), fMins[slot]);
   }

   void Initialize() { /* noop */}

   void Finalize()
   {
      *fResultMin = std::numeric_limits<ResultType>::max();
      for (auto &m : fMins)
         *fResultMin = std::min(m, *fResultMin);
   }

   // Helper functions for RMergeableValue
   std::unique_ptr<RMergeableValueBase> GetMergeableValue() const final
   {
      return std::make_unique<RMergeableMin<ResultType>>(*fResultMin);
   }

   ResultType &PartialUpdate(unsigned int slot) { return fMins[slot]; }

   std::string GetActionName() { return "Min"; }

   MinHelper MakeNew(void *newResult)
   {
      auto &result = *static_cast<std::shared_ptr<ResultType> *>(newResult);
      return MinHelper(result, fMins.size());
   }
};

// TODO
// extern template void MinHelper::Exec(unsigned int, const std::vector<float> &);
// extern template void MinHelper::Exec(unsigned int, const std::vector<double> &);
// extern template void MinHelper::Exec(unsigned int, const std::vector<char> &);
// extern template void MinHelper::Exec(unsigned int, const std::vector<int> &);
// extern template void MinHelper::Exec(unsigned int, const std::vector<unsigned int> &);

template <typename ResultType>
class MaxHelper : public RActionImpl<MaxHelper<ResultType>> {
   const std::shared_ptr<ResultType> fResultMax;
   Results<ResultType> fMaxs;

public:
   MaxHelper(MaxHelper &&) = default;
   MaxHelper(const MaxHelper &) = delete;
   MaxHelper(const std::shared_ptr<ResultType> &maxVPtr, const unsigned int nSlots)
      : fResultMax(maxVPtr), fMaxs(nSlots, std::numeric_limits<ResultType>::lowest())
   {
   }

   void InitTask(TTreeReader *, unsigned int) {}
   void Exec(unsigned int slot, ResultType v) { fMaxs[slot] = std::max(v, fMaxs[slot]); }

   template <typename T, std::enable_if_t<IsDataContainer<T>::value, int> = 0>
   void Exec(unsigned int slot, const T &vs)
   {
      for (auto &&v : vs)
         fMaxs[slot] = std::max(static_cast<ResultType>(v), fMaxs[slot]);
   }

   void Initialize() { /* noop */}

   void Finalize()
   {
      *fResultMax = std::numeric_limits<ResultType>::lowest();
      for (auto &m : fMaxs) {
         *fResultMax = std::max(m, *fResultMax);
      }
   }

   // Helper functions for RMergeableValue
   std::unique_ptr<RMergeableValueBase> GetMergeableValue() const final
   {
      return std::make_unique<RMergeableMax<ResultType>>(*fResultMax);
   }

   ResultType &PartialUpdate(unsigned int slot) { return fMaxs[slot]; }

   std::string GetActionName() { return "Max"; }

   MaxHelper MakeNew(void *newResult)
   {
      auto &result = *static_cast<std::shared_ptr<ResultType> *>(newResult);
      return MaxHelper(result, fMaxs.size());
   }
};

// TODO
// extern template void MaxHelper::Exec(unsigned int, const std::vector<float> &);
// extern template void MaxHelper::Exec(unsigned int, const std::vector<double> &);
// extern template void MaxHelper::Exec(unsigned int, const std::vector<char> &);
// extern template void MaxHelper::Exec(unsigned int, const std::vector<int> &);
// extern template void MaxHelper::Exec(unsigned int, const std::vector<unsigned int> &);

template <typename ResultType>
class SumHelper : public RActionImpl<SumHelper<ResultType>> {
   const std::shared_ptr<ResultType> fResultSum;
   Results<ResultType> fSums;

   /// Evaluate neutral element for this type and the sum operation.
   /// This is assumed to be any_value - any_value if operator- is defined
   /// for the type, otherwise a default-constructed ResultType{} is used.
   template <typename T = ResultType>
   auto NeutralElement(const T &v, int /*overloadresolver*/) -> decltype(v - v)
   {
      return v - v;
   }

   template <typename T = ResultType, typename Dummy = int>
   ResultType NeutralElement(const T &, Dummy) // this overload has lower priority thanks to the template arg
   {
      return ResultType{};
   }

public:
   SumHelper(SumHelper &&) = default;
   SumHelper(const SumHelper &) = delete;
   SumHelper(const std::shared_ptr<ResultType> &sumVPtr, const unsigned int nSlots)
      : fResultSum(sumVPtr), fSums(nSlots, NeutralElement(*sumVPtr, -1))
   {
   }

   void InitTask(TTreeReader *, unsigned int) {}
   void Exec(unsigned int slot, ResultType v) { fSums[slot] += v; }

   template <typename T, std::enable_if_t<IsDataContainer<T>::value, int> = 0>
   void Exec(unsigned int slot, const T &vs)
   {
      for (auto &&v : vs)
         fSums[slot] += static_cast<ResultType>(v);
   }

   void Initialize() { /* noop */}

   void Finalize()
   {
      for (auto &m : fSums)
         *fResultSum += m;
   }

   // Helper functions for RMergeableValue
   std::unique_ptr<RMergeableValueBase> GetMergeableValue() const final
   {
      return std::make_unique<RMergeableSum<ResultType>>(*fResultSum);
   }

   ResultType &PartialUpdate(unsigned int slot) { return fSums[slot]; }

   std::string GetActionName() { return "Sum"; }

   SumHelper MakeNew(void *newResult)
   {
      auto &result = *static_cast<std::shared_ptr<ResultType> *>(newResult);
      *result = NeutralElement(*result, -1);
      return SumHelper(result, fSums.size());
   }
};

class MeanHelper : public RActionImpl<MeanHelper> {
   const std::shared_ptr<double> fResultMean;
   std::vector<ULong64_t> fCounts;
   std::vector<double> fSums;
   std::vector<double> fPartialMeans;

public:
   MeanHelper(const std::shared_ptr<double> &meanVPtr, const unsigned int nSlots);
   MeanHelper(MeanHelper &&) = default;
   MeanHelper(const MeanHelper &) = delete;
   void InitTask(TTreeReader *, unsigned int) {}
   void Exec(unsigned int slot, double v);

   template <typename T, std::enable_if_t<IsDataContainer<T>::value, int> = 0>
   void Exec(unsigned int slot, const T &vs)
   {
      for (auto &&v : vs) {
         fSums[slot] += v;
         fCounts[slot]++;
      }
   }

   void Initialize() { /* noop */}

   void Finalize();

   // Helper functions for RMergeableValue
   std::unique_ptr<RMergeableValueBase> GetMergeableValue() const final
   {
      const ULong64_t counts = std::accumulate(fCounts.begin(), fCounts.end(), 0ull);
      return std::make_unique<RMergeableMean>(*fResultMean, counts);
   }

   double &PartialUpdate(unsigned int slot);

   std::string GetActionName() { return "Mean"; }

   MeanHelper MakeNew(void *newResult)
   {
      auto &result = *static_cast<std::shared_ptr<double> *>(newResult);
      return MeanHelper(result, fSums.size());
   }
};

extern template void MeanHelper::Exec(unsigned int, const std::vector<float> &);
extern template void MeanHelper::Exec(unsigned int, const std::vector<double> &);
extern template void MeanHelper::Exec(unsigned int, const std::vector<char> &);
extern template void MeanHelper::Exec(unsigned int, const std::vector<int> &);
extern template void MeanHelper::Exec(unsigned int, const std::vector<unsigned int> &);

class StdDevHelper : public RActionImpl<StdDevHelper> {
   // Number of subsets of data
   const unsigned int fNSlots;
   const std::shared_ptr<double> fResultStdDev;
   // Number of element for each slot
   std::vector<ULong64_t> fCounts;
   // Mean of each slot
   std::vector<double> fMeans;
   // Squared distance from the mean
   std::vector<double> fDistancesfromMean;

public:
   StdDevHelper(const std::shared_ptr<double> &meanVPtr, const unsigned int nSlots);
   StdDevHelper(StdDevHelper &&) = default;
   StdDevHelper(const StdDevHelper &) = delete;
   void InitTask(TTreeReader *, unsigned int) {}
   void Exec(unsigned int slot, double v);

   template <typename T, std::enable_if_t<IsDataContainer<T>::value, int> = 0>
   void Exec(unsigned int slot, const T &vs)
   {
      for (auto &&v : vs) {
         Exec(slot, v);
      }
   }

   void Initialize() { /* noop */}

   void Finalize();

   // Helper functions for RMergeableValue
   std::unique_ptr<RMergeableValueBase> GetMergeableValue() const final
   {
      const ULong64_t counts = std::accumulate(fCounts.begin(), fCounts.end(), 0ull);
      const Double_t mean =
         std::inner_product(fMeans.begin(), fMeans.end(), fCounts.begin(), 0.) / static_cast<Double_t>(counts);
      return std::make_unique<RMergeableStdDev>(*fResultStdDev, counts, mean);
   }

   std::string GetActionName() { return "StdDev"; }

   StdDevHelper MakeNew(void *newResult)
   {
      auto &result = *static_cast<std::shared_ptr<double> *>(newResult);
      return StdDevHelper(result, fCounts.size());
   }
};

extern template void StdDevHelper::Exec(unsigned int, const std::vector<float> &);
extern template void StdDevHelper::Exec(unsigned int, const std::vector<double> &);
extern template void StdDevHelper::Exec(unsigned int, const std::vector<char> &);
extern template void StdDevHelper::Exec(unsigned int, const std::vector<int> &);
extern template void StdDevHelper::Exec(unsigned int, const std::vector<unsigned int> &);

template <typename PrevNodeType>
class DisplayHelper : public RActionImpl<DisplayHelper<PrevNodeType>> {
private:
   using Display_t = ROOT::RDF::RDisplay;
   const std::shared_ptr<Display_t> fDisplayerHelper;
   const std::shared_ptr<PrevNodeType> fPrevNode;

public:
   DisplayHelper(const std::shared_ptr<Display_t> &d, const std::shared_ptr<PrevNodeType> &prevNode)
      : fDisplayerHelper(d), fPrevNode(prevNode)
   {
   }
   DisplayHelper(DisplayHelper &&) = default;
   DisplayHelper(const DisplayHelper &) = delete;
   void InitTask(TTreeReader *, unsigned int) {}

   template <typename... Columns>
   void Exec(unsigned int, Columns &... columns)
   {
      fDisplayerHelper->AddRow(columns...);
      if (!fDisplayerHelper->HasNext()) {
         fPrevNode->StopProcessing();
      }
   }

   void Initialize() {}

   void Finalize() {}

   std::string GetActionName() { return "Display"; }
};

template <typename T>
void *GetData(ROOT::VecOps::RVec<T> &v)
{
   return v.data();
}

template <typename T>
void *GetData(T & /*v*/)
{
   return nullptr;
}

template <typename T>
void SetBranchesHelper(TTree *inputTree, TTree &outputTree, const std::string &inName,
                       const std::string &name, TBranch *&branch, void *&branchAddress, T *address,
                       RBranchSet &outputBranches)
{
   static TClassRef TBOClRef("TBranchObject");
   // FIXME we should be using FindBranch as a fallback if GetBranch fails
   TBranch *inputBranch = inputTree ? inputTree->GetBranch(inName.c_str()) : nullptr;

   auto *outputBranch = outputBranches.Get(name);
   if (outputBranch) {
      // the output branch was already created, we just need to (re)set its address
      if (inputBranch && inputBranch->IsA() == TBOClRef) {
         outputBranch->SetAddress(reinterpret_cast<T **>(inputBranch->GetAddress()));
      } else if (outputBranch->IsA() != TBranch::Class()) {
         branchAddress = address;
         outputBranch->SetAddress(&branchAddress);
      } else {
         outputBranch->SetAddress(address);
         branchAddress = address;
      }
      return;
   }

   if (inputBranch) {
      // Respect the original bufsize and splitlevel arguments
      // In particular, by keeping splitlevel equal to 0 if this was the case for `inputBranch`, we avoid
      // writing garbage when unsplit objects cannot be written as split objects (e.g. in case of a polymorphic
      // TObject branch, see https://bit.ly/2EjLMId ).
      const auto bufSize = inputBranch->GetBasketSize();
      const auto splitLevel = inputBranch->GetSplitLevel();

      if (inputBranch->IsA() == TBOClRef) {
         // Need to pass a pointer to pointer
         outputBranch =
            outputTree.Branch(name.c_str(), reinterpret_cast<T **>(inputBranch->GetAddress()), bufSize, splitLevel);
      } else {
         outputBranch = outputTree.Branch(name.c_str(), address, bufSize, splitLevel);
      }
   } else {
      outputBranch = outputTree.Branch(name.c_str(), address);
   }
   outputBranches.Insert(name, outputBranch);
   // This is not an array branch, so we don't register the address of the output branch here
   branch = nullptr;
   branchAddress = nullptr;
}

/// Helper function for SnapshotHelper and SnapshotHelperMT. It creates new branches for the output TTree of a Snapshot.
/// This overload is called for columns of type `RVec<T>`. For RDF, these can represent:
/// 1. c-style arrays in ROOT files, so we are sure that there are input trees to which we can ask the correct branch title
/// 2. RVecs coming from a custom column or a source
/// 3. vectors coming from ROOT files
/// 4. TClonesArray
///
/// In case of 1., we keep aside the pointer to the branch and the pointer to the input value (in `branch` and
/// `branchAddress`) so we can intercept changes in the address of the input branch and tell the output branch.
template <typename T>
void SetBranchesHelper(TTree *inputTree, TTree &outputTree, const std::string &inName,
                       const std::string &outName, TBranch *&branch, void *&branchAddress, RVec<T> *ab,
                       RBranchSet &outputBranches)
{
   TBranch *inputBranch = nullptr;
   if (inputTree) {
      inputBranch = inputTree->GetBranch(inName.c_str());
      if (!inputBranch) {
         // try harder
         inputBranch = inputTree->FindBranch(inName.c_str());
      }
   }
   auto *outputBranch = outputBranches.Get(outName);
   const bool isTClonesArray = inputBranch != nullptr && std::string(inputBranch->GetClassName()) == "TClonesArray";
   const auto mustWriteRVec = !inputBranch || isTClonesArray ||
                              ROOT::ESTLType::kSTLvector == TClassEdit::IsSTLCont(inputBranch->GetClassName());

   if (mustWriteRVec) {
      // Treat:
      // 2. RVec coming from a custom column or a source
      // 3. RVec coming from a column on disk of type vector (the RVec is adopting the data of that vector)
      // 4. TClonesArray written out as RVec<T>
      if (isTClonesArray) {
         Warning("Snapshot",
                 "Branch \"%s\" contains TClonesArrays but the type specified to Snapshot was RVec<T>. The branch will "
                 "be written out as a RVec instead of a TClonesArray. Specify that the type of the branch is "
                 "TClonesArray as a Snapshot template parameter to write out a TClonesArray instead.", inName.c_str());
      }
      if (outputBranch) {
         branchAddress = ab->data();
         outputBranch->SetAddress(&branchAddress);
      } else {
         auto *b = outputTree.Branch(outName.c_str(), ab);
         outputBranches.Insert(outName, b);
      }
      return;
   }

   // Treat 1, the C-array case
   auto *const leaf = static_cast<TLeaf *>(inputBranch->GetListOfLeaves()->UncheckedAt(0));
   const auto bname = leaf->GetName();
   const auto counterStr =
      leaf->GetLeafCount() ? std::string(leaf->GetLeafCount()->GetName()) : std::to_string(leaf->GetLenStatic());
   const auto btype = leaf->GetTypeName();
   const auto rootbtype = TypeName2ROOTTypeName(btype);
   const auto leaflist = std::string(bname) + "[" + counterStr + "]/" + rootbtype;

   auto dataPtr = ab->data();

   if (outputBranch) {
      if (outputBranch->IsA() != TBranch::Class()) {
         branchAddress = dataPtr;
         outputBranch->SetAddress(&branchAddress);
      } else {
         outputBranch->SetAddress(dataPtr);
      }
   } else {
      outputBranch = outputTree.Branch(outName.c_str(), dataPtr, leaflist.c_str());
      outputBranch->SetTitle(inputBranch->GetTitle());
      outputBranches.Insert(outName, outputBranch);
      branch = outputBranch;
      branchAddress = ab->data();
   }
}

void ValidateSnapshotOutput(const RSnapshotOptions &opts, const std::string &treeName, const std::string &fileName);

/// Helper object for a single-thread Snapshot action
template <typename... ColTypes>
class SnapshotHelper : public RActionImpl<SnapshotHelper<ColTypes...>> {
   const std::string fFileName;
   const std::string fDirName;
   const std::string fTreeName;
   const RSnapshotOptions fOptions;
   std::unique_ptr<TFile> fOutputFile;
   std::unique_ptr<TTree> fOutputTree; // must be a ptr because TTrees are not copy/move constructible
   bool fBranchAddressesNeedReset{true};
   const ColumnNames_t fInputBranchNames; // This contains the resolved aliases
   const ColumnNames_t fOutputBranchNames;
   TTree *fInputTree = nullptr; // Current input tree. Set at initialization time (`InitTask`)
   // TODO we might be able to unify fBranches, fBranchAddresses and fOutputBranches
   std::vector<TBranch *> fBranches; // Addresses of branches in output, non-null only for the ones holding C arrays
   std::vector<void *> fBranchAddresses; // Addresses of objects associated to output branches
   RBranchSet fOutputBranches;

public:
   using ColumnTypes_t = TypeList<ColTypes...>;
   SnapshotHelper(std::string_view filename, std::string_view dirname, std::string_view treename,
                  const ColumnNames_t &vbnames, const ColumnNames_t &bnames, const RSnapshotOptions &options)
      : fFileName(filename), fDirName(dirname), fTreeName(treename), fOptions(options), fInputBranchNames(vbnames),
        fOutputBranchNames(ReplaceDotWithUnderscore(bnames)), fBranches(vbnames.size(), nullptr),
        fBranchAddresses(vbnames.size(), nullptr)
   {
      ValidateSnapshotOutput(fOptions, fTreeName, fFileName);
   }

   SnapshotHelper(const SnapshotHelper &) = delete;
   SnapshotHelper(SnapshotHelper &&) = default;

   void InitTask(TTreeReader *r, unsigned int /* slot */)
   {
      if (r)
         fInputTree = r->GetTree();
      fBranchAddressesNeedReset = true;
   }

   void Exec(unsigned int /* slot */, ColTypes &... values)
   {
      using ind_t = std::index_sequence_for<ColTypes...>;
      if (!fBranchAddressesNeedReset) {
         UpdateCArraysPtrs(values..., ind_t{});
      } else {
         SetBranches(values..., ind_t{});
         fBranchAddressesNeedReset = false;
      }
      fOutputTree->Fill();
   }

   template <std::size_t... S>
   void UpdateCArraysPtrs(ColTypes &... values, std::index_sequence<S...> /*dummy*/)
   {
      // This code deals with branches which hold C arrays of variable size. It can happen that the buffers
      // associated to those is re-allocated. As a result the value of the pointer can change therewith
      // leaving associated to the branch of the output tree an invalid pointer.
      // With this code, we set the value of the pointer in the output branch anew when needed.
      // Nota bene: the extra ",0" after the invocation of SetAddress, is because that method returns void and
      // we need an int for the expander list.
      int expander[] = {(fBranches[S] && fBranchAddresses[S] != GetData(values)
                         ? fBranches[S]->SetAddress(GetData(values)),
                         fBranchAddresses[S] = GetData(values), 0 : 0, 0)...,
                        0};
      (void)expander; // avoid unused variable warnings for older compilers such as gcc 4.9
   }

   template <std::size_t... S>
   void SetBranches(ColTypes &... values, std::index_sequence<S...> /*dummy*/)
   {
      // create branches in output tree
      int expander[] = {
         (SetBranchesHelper(fInputTree, *fOutputTree, fInputBranchNames[S], fOutputBranchNames[S],
                            fBranches[S], fBranchAddresses[S], &values, fOutputBranches),
          0)...,
         0};
      (void)expander; // avoid unused variable warnings for older compilers such as gcc 4.9
   }

   void Initialize()
   {
      fOutputFile.reset(
         TFile::Open(fFileName.c_str(), fOptions.fMode.c_str(), /*ftitle=*/"",
                     ROOT::CompressionSettings(fOptions.fCompressionAlgorithm, fOptions.fCompressionLevel)));
      if(!fOutputFile)
         throw std::runtime_error("Snapshot: could not create output file " + fFileName);

      TDirectory *outputDir = fOutputFile.get();
      if (!fDirName.empty()) {
         TString checkupdate = fOptions.fMode;
         checkupdate.ToLower();
         if (checkupdate == "update")
            outputDir = fOutputFile->mkdir(fDirName.c_str(), "", true);  // do not overwrite existing directory
         else
            outputDir = fOutputFile->mkdir(fDirName.c_str());
      }

      fOutputTree =
         std::make_unique<TTree>(fTreeName.c_str(), fTreeName.c_str(), fOptions.fSplitLevel, /*dir=*/outputDir);

      if (fOptions.fAutoFlush)
         fOutputTree->SetAutoFlush(fOptions.fAutoFlush);
   }

   void Finalize()
   {
      assert(fOutputTree != nullptr);
      assert(fOutputFile != nullptr);

      // use AutoSave to flush TTree contents because TTree::Write writes in gDirectory, not in fDirectory
      fOutputTree->AutoSave("flushbaskets");
      // must destroy the TTree first, otherwise TFile will delete it too leading to a double delete
      fOutputTree.reset();
      fOutputFile->Close();
   }

   std::string GetActionName() { return "Snapshot"; }

   ROOT::RDF::SampleCallback_t GetSampleCallback() final
   {
      return [this](unsigned int, const RSampleInfo &) mutable { fBranchAddressesNeedReset = true; };
   }
};

/// Helper object for a multi-thread Snapshot action
template <typename... ColTypes>
class SnapshotHelperMT : public RActionImpl<SnapshotHelperMT<ColTypes...>> {
   const unsigned int fNSlots;
   std::unique_ptr<ROOT::TBufferMerger> fMerger; // must use a ptr because TBufferMerger is not movable
   std::vector<std::shared_ptr<ROOT::TBufferMergerFile>> fOutputFiles;
   std::vector<std::unique_ptr<TTree>> fOutputTrees;
   std::vector<int> fBranchAddressesNeedReset; // vector<bool> does not allow concurrent writing of different elements
   const std::string fFileName;           // name of the output file name
   const std::string fDirName;            // name of TFile subdirectory in which output must be written (possibly empty)
   const std::string fTreeName;           // name of output tree
   const RSnapshotOptions fOptions;       // struct holding options to pass down to TFile and TTree in this action
   const ColumnNames_t fInputBranchNames; // This contains the resolved aliases
   const ColumnNames_t fOutputBranchNames;
   std::vector<TTree *> fInputTrees; // Current input trees. Set at initialization time (`InitTask`)
   // Addresses of branches in output per slot, non-null only for the ones holding C arrays
   std::vector<std::vector<TBranch *>> fBranches;
   // Addresses associated to output branches per slot, non-null only for the ones holding C arrays
   std::vector<std::vector<void *>> fBranchAddresses;
   std::vector<RBranchSet> fOutputBranches;

public:
   using ColumnTypes_t = TypeList<ColTypes...>;
   SnapshotHelperMT(const unsigned int nSlots, std::string_view filename, std::string_view dirname,
                    std::string_view treename, const ColumnNames_t &vbnames, const ColumnNames_t &bnames,
                    const RSnapshotOptions &options)
      : fNSlots(nSlots), fOutputFiles(fNSlots), fOutputTrees(fNSlots), fBranchAddressesNeedReset(fNSlots, 1),
        fFileName(filename), fDirName(dirname), fTreeName(treename), fOptions(options), fInputBranchNames(vbnames),
        fOutputBranchNames(ReplaceDotWithUnderscore(bnames)), fInputTrees(fNSlots),
        fBranches(fNSlots, std::vector<TBranch *>(vbnames.size(), nullptr)),
        fBranchAddresses(fNSlots, std::vector<void *>(vbnames.size(), nullptr)), fOutputBranches(fNSlots)
   {
      ValidateSnapshotOutput(fOptions, fTreeName, fFileName);
   }
   SnapshotHelperMT(const SnapshotHelperMT &) = delete;
   SnapshotHelperMT(SnapshotHelperMT &&) = default;

   void InitTask(TTreeReader *r, unsigned int slot)
   {
      ::TDirectory::TContext c; // do not let tasks change the thread-local gDirectory
      if (!fOutputFiles[slot]) {
         // first time this thread executes something, let's create a TBufferMerger output directory
         fOutputFiles[slot] = fMerger->GetFile();
      }
      TDirectory *treeDirectory = fOutputFiles[slot].get();
      if (!fDirName.empty()) {
         // call returnExistingDirectory=true since MT can end up making this call multiple times
         treeDirectory = fOutputFiles[slot]->mkdir(fDirName.c_str(), "", true);
      }
      // re-create output tree as we need to create its branches again, with new input variables
      // TODO we could instead create the output tree and its branches, change addresses of input variables in each task
      fOutputTrees[slot] =
         std::make_unique<TTree>(fTreeName.c_str(), fTreeName.c_str(), fOptions.fSplitLevel, /*dir=*/treeDirectory);
      fOutputTrees[slot]->SetBit(TTree::kEntriesReshuffled);
      // TODO can be removed when RDF supports interleaved TBB task execution properly, see ROOT-10269
      fOutputTrees[slot]->SetImplicitMT(false);
      if (fOptions.fAutoFlush)
         fOutputTrees[slot]->SetAutoFlush(fOptions.fAutoFlush);
      if (r) {
         // not an empty-source RDF
         fInputTrees[slot] = r->GetTree();
      }
      fBranchAddressesNeedReset[slot] = 1; // reset first event flag for this slot
   }

   void FinalizeTask(unsigned int slot)
   {
      if (fOutputTrees[slot]->GetEntries() > 0)
         fOutputFiles[slot]->Write();
      // clear now to avoid concurrent destruction of output trees and input tree (which has them listed as fClones)
      fOutputTrees[slot].reset(nullptr);
      fOutputBranches[slot].Clear();
   }

   void Exec(unsigned int slot, ColTypes &... values)
   {
      using ind_t = std::index_sequence_for<ColTypes...>;
      if (fBranchAddressesNeedReset[slot] == 0) {
         UpdateCArraysPtrs(slot, values..., ind_t{});
      } else {
         SetBranches(slot, values..., ind_t{});
         fBranchAddressesNeedReset[slot] = 0;
      }
      fOutputTrees[slot]->Fill();
      auto entries = fOutputTrees[slot]->GetEntries();
      auto autoFlush = fOutputTrees[slot]->GetAutoFlush();
      if ((autoFlush > 0) && (entries % autoFlush == 0))
         fOutputFiles[slot]->Write();
   }

   template <std::size_t... S>
   void UpdateCArraysPtrs(unsigned int slot, ColTypes &... values, std::index_sequence<S...> /*dummy*/)
   {
      // This code deals with branches which hold C arrays of variable size. It can happen that the buffers
      // associated to those is re-allocated. As a result the value of the pointer can change therewith
      // leaving associated to the branch of the output tree an invalid pointer.
      // With this code, we set the value of the pointer in the output branch anew when needed.
      // Nota bene: the extra ",0" after the invocation of SetAddress, is because that method returns void and
      // we need an int for the expander list.
      (void)slot; // avoid bogus 'unused parameter' warning
      int expander[] = {(fBranches[slot][S] && fBranchAddresses[slot][S] != GetData(values)
                         ? fBranches[slot][S]->SetAddress(GetData(values)),
                         fBranchAddresses[slot][S] = GetData(values), 0 : 0, 0)...,
                        0};
      (void)expander; // avoid unused variable warnings for older compilers such as gcc 4.9
   }

   template <std::size_t... S>
   void SetBranches(unsigned int slot, ColTypes &... values, std::index_sequence<S...> /*dummy*/)
   {
         // hack to call TTree::Branch on all variadic template arguments
         int expander[] = {
            (SetBranchesHelper(fInputTrees[slot], *fOutputTrees[slot], fInputBranchNames[S], fOutputBranchNames[S],
                               fBranches[slot][S], fBranchAddresses[slot][S], &values, fOutputBranches[slot]),
             0)...,
            0};
         (void)expander; // avoid unused variable warnings for older compilers such as gcc 4.9
         (void)slot;     // avoid unused variable warnings in gcc6.2
   }

   void Initialize()
   {
      const auto cs = ROOT::CompressionSettings(fOptions.fCompressionAlgorithm, fOptions.fCompressionLevel);
      auto out_file = TFile::Open(fFileName.c_str(), fOptions.fMode.c_str(), /*ftitle=*/fFileName.c_str(), cs);
      if(!out_file)
         throw std::runtime_error("Snapshot: could not create output file " + fFileName);
      fMerger = std::make_unique<ROOT::TBufferMerger>(std::unique_ptr<TFile>(out_file));
   }

   void Finalize()
   {
      assert(std::any_of(fOutputFiles.begin(), fOutputFiles.end(), [](const auto &ptr) { return ptr != nullptr; }));

      auto fileWritten = false;
      for (auto &file : fOutputFiles) {
         if (file) {
            file->Write();
            file->Close();
            fileWritten = true;
         }
      }

      if (!fileWritten) {
         Warning("Snapshot",
                 "No input entries (input TTree was empty or no entry passed the Filters). Output TTree is empty.");
      }

      // flush all buffers to disk by destroying the TBufferMerger
      fOutputFiles.clear();
      fMerger.reset();
   }

   std::string GetActionName() { return "Snapshot"; }

   ROOT::RDF::SampleCallback_t GetSampleCallback() final
   {
      return [this](unsigned int slot, const RSampleInfo &) mutable { fBranchAddressesNeedReset[slot] = 1; };
   }
};

template <typename Acc, typename Merge, typename R, typename T, typename U,
          bool MustCopyAssign = std::is_same<R, U>::value>
class AggregateHelper : public RActionImpl<AggregateHelper<Acc, Merge, R, T, U, MustCopyAssign>> {
   Acc fAggregate;
   Merge fMerge;
   const std::shared_ptr<U> fResult;
   Results<U> fAggregators;

public:
   using ColumnTypes_t = TypeList<T>;

   AggregateHelper(Acc &&f, Merge &&m, const std::shared_ptr<U> &result, const unsigned int nSlots)
      : fAggregate(std::move(f)), fMerge(std::move(m)), fResult(result), fAggregators(nSlots, *result)
   {
   }

   AggregateHelper(Acc &f, Merge &m, const std::shared_ptr<U> &result, const unsigned int nSlots)
      : fAggregate(f), fMerge(m), fResult(result), fAggregators(nSlots, *result)
   {
   }

   AggregateHelper(AggregateHelper &&) = default;
   AggregateHelper(const AggregateHelper &) = delete;

   void InitTask(TTreeReader *, unsigned int) {}

   template <bool MustCopyAssign_ = MustCopyAssign, std::enable_if_t<MustCopyAssign_, int> = 0>
   void Exec(unsigned int slot, const T &value)
   {
      fAggregators[slot] = fAggregate(fAggregators[slot], value);
   }

   template <bool MustCopyAssign_ = MustCopyAssign, std::enable_if_t<!MustCopyAssign_, int> = 0>
   void Exec(unsigned int slot, const T &value)
   {
      fAggregate(fAggregators[slot], value);
   }

   void Initialize() { /* noop */}

   template <typename MergeRet = typename CallableTraits<Merge>::ret_type,
             bool MergeAll = std::is_same<void, MergeRet>::value>
   std::enable_if_t<MergeAll, void> Finalize()
   {
      fMerge(fAggregators);
      *fResult = fAggregators[0];
   }

   template <typename MergeRet = typename CallableTraits<Merge>::ret_type,
             bool MergeTwoByTwo = std::is_same<U, MergeRet>::value>
   std::enable_if_t<MergeTwoByTwo, void> Finalize(...) // ... needed to let compiler distinguish overloads
   {
      for (const auto &acc : fAggregators)
         *fResult = fMerge(*fResult, acc);
   }

   U &PartialUpdate(unsigned int slot) { return fAggregators[slot]; }

   std::string GetActionName() { return "Aggregate"; }

   AggregateHelper MakeNew(void *newResult)
   {
      auto &result = *static_cast<std::shared_ptr<U> *>(newResult);
      return AggregateHelper(fAggregate, fMerge, result, fAggregators.size());
   }
};

} // end of NS RDF
} // end of NS Internal
} // end of NS ROOT

/// \endcond

#endif
