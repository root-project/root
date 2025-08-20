// Author: Enrico Guiraud, Danilo Piparo CERN  03/2017

/*************************************************************************
 * Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RRESULTPTR
#define ROOT_RRESULTPTR

#include "ROOT/RDF/RActionBase.hxx"
#include "RtypesCore.h"
#include "ROOT/RDF/RLoopManager.hxx"
#include "ROOT/TypeTraits.hxx"
#include "TError.h" // Warning

#include <memory>
#include <functional>
#include <type_traits> // std::is_constructible

namespace ROOT {
namespace RDF {
template <typename T>
class RResultPtr;

namespace Experimental {
template <typename T>
class RResultMap;

template <typename T>
RResultMap<T> VariationsFor(RResultPtr<T> resPtr);
} // namespace Experimental

template <typename Proxied, typename DataSource>
class RInterface;
} // namespace RDF

namespace Internal {
namespace RDF {
class GraphCreatorHelper;
/**
 * \brief Creates a new RResultPtr with a cloned action.
 *
 * \tparam T The type of the result held by the RResultPtr.
 * \param inptr The pointer.
 * \return A new pointer with a cloned action.
 */
template <typename T>
ROOT::RDF::RResultPtr<T> CloneResultAndAction(const ROOT::RDF::RResultPtr<T> &inptr)
{
   // We call the copy constructor, to copy also the metadata of certain
   // result types, e.g. a for a TH1D we have to create a new histogram with
   // the same binning and axis limits.
   std::shared_ptr<T> copiedResult{new T{*inptr.fObjPtr}};
   return ROOT::RDF::RResultPtr<T>(copiedResult, inptr.fLoopManager,
                                   inptr.fActionPtr->CloneAction(reinterpret_cast<void *>(&copiedResult)));
}

using SnapshotPtr_t = ROOT::RDF::RResultPtr<ROOT::RDF::RInterface<ROOT::Detail::RDF::RLoopManager, void>>;
/**
 * \brief Creates a new RResultPtr with a cloned Snapshot action.
 *
 * \param inptr The pointer.
 * \param outputFileName A new name for the output file of the cloned action.
 * \return A new pointer with a cloned action.
 *
 * This overload is needed since cloning a Snapshot node usually also involves
 * changing the name of the output file, otherwise the cloned Snapshot would
 * overwrite the same file.
 */
SnapshotPtr_t CloneResultAndAction(const SnapshotPtr_t &inptr, const std::string &outputFileName);
} // namespace RDF
} // namespace Internal

namespace Detail {
namespace RDF {
using ROOT::RDF::RResultPtr;
// Fwd decl for RResultPtr
template <typename T>
RResultPtr<T> MakeResultPtr(const std::shared_ptr<T> &r, RLoopManager &df,
                            std::shared_ptr<ROOT::Internal::RDF::RActionBase> actionPtr);

// Fwd decl for GetMergeableValue
template <typename T>
class RMergeableValue;

template <typename T>
std::unique_ptr<RMergeableValue<T>> GetMergeableValue(RResultPtr<T> &rptr);
} // namespace RDF
} // namespace Detail
namespace RDF {
namespace RDFInternal = ROOT::Internal::RDF;
namespace RDFDetail = ROOT::Detail::RDF;
namespace TTraits = ROOT::TypeTraits;

/// Smart pointer for the return type of actions.
/**
\class ROOT::RDF::RResultPtr
\ingroup dataframe
\brief A wrapper around the result of RDataFrame actions able to trigger calculations lazily.
\tparam T Type of the action result

A wrapper around a shared_ptr which allows to access the result of RDataFrame actions.
The underlying object can be accessed by dereferencing the RResultPtr:
~~~{.cpp}
ROOT::RDF::RResultPtr<TH1D> histo = rdf.Histo1D(...);
histo->Draw(); // Starts running the event loop
~~~
Upon invocation of the arrow operator or dereferencing (`operator*`), the
loop on the events and calculations of all scheduled actions are executed
if needed.
It is possible to iterate on the result proxy if the proxied object is a collection.
~~~{.cpp}
for (auto& myItem : myResultProxy) { ... };
~~~
If iteration is not supported by the type of the proxied object, a compilation error is thrown.

When shared ownership to the result is desired, a copy of the underlying shared_ptr can be obtained:
~~~{.cpp}
std::shared_ptr<TH1D> ProduceResult(const char *columnname) {
   auto ht = rdf.Histo1D(*h, columname);
   return ht.GetSharedPtr();
}
~~~
Note that this will run the event loop. If this is not desired, the RResultPtr can be copied.
*/
template <typename T>
class RResultPtr {
   // friend declarations
   template <typename T1>
   friend class RResultPtr;

   template <typename T1>
   friend RResultPtr<T1> RDFDetail::MakeResultPtr(const std::shared_ptr<T1> &, ::ROOT::Detail::RDF::RLoopManager &,
                                                  std::shared_ptr<RDFInternal::RActionBase>);

   template <typename T1>
   friend ROOT::RDF::Experimental::RResultMap<T1> ROOT::RDF::Experimental::VariationsFor(RResultPtr<T1> resPtr);

   template <class T1, class T2>
   friend bool operator==(const RResultPtr<T1> &lhs, const RResultPtr<T2> &rhs);
   template <class T1, class T2>
   friend bool operator!=(const RResultPtr<T1> &lhs, const RResultPtr<T2> &rhs);
   template <class T1>
   friend bool operator==(const RResultPtr<T1> &lhs, std::nullptr_t rhs);
   template <class T1>
   friend bool operator==(std::nullptr_t lhs, const RResultPtr<T1> &rhs);
   template <class T1>
   friend bool operator!=(const RResultPtr<T1> &lhs, std::nullptr_t rhs);
   template <class T1>
   friend bool operator!=(std::nullptr_t lhs, const RResultPtr<T1> &rhs);
   friend std::unique_ptr<RDFDetail::RMergeableValue<T>> RDFDetail::GetMergeableValue<T>(RResultPtr<T> &rptr);

   friend class ROOT::Internal::RDF::GraphDrawing::GraphCreatorHelper;

   friend class RResultHandle;

   friend RResultPtr<T> ROOT::Internal::RDF::CloneResultAndAction<T>(const RResultPtr<T> &inptr);
   friend ROOT::Internal::RDF::SnapshotPtr_t
   ROOT::Internal::RDF::CloneResultAndAction(const ROOT::Internal::RDF::SnapshotPtr_t &inptr,
                                             const std::string &outputFileName);
   /// \cond HIDDEN_SYMBOLS
   template <typename V, bool hasBeginEnd = TTraits::HasBeginAndEnd<V>::value>
   struct RIterationHelper {
      using Iterator_t = void;
      void GetBegin(const V &) { static_assert(sizeof(V) == 0, "It does not make sense to ask begin for this class."); }
      void GetEnd(const V &) { static_assert(sizeof(V) == 0, "It does not make sense to ask end for this class."); }
   };

   template <typename V>
   struct RIterationHelper<V, true> {
      using Iterator_t = decltype(std::begin(std::declval<V>()));
      static Iterator_t GetBegin(const V &v) { return std::begin(v); };
      static Iterator_t GetEnd(const V &v) { return std::end(v); };
   };
   /// \endcond

   /// Non-owning pointer to the RLoopManager at the root of this computation graph.
   /// The RLoopManager is guaranteed to be always in scope if fLoopManager is not a nullptr.
   RDFDetail::RLoopManager *fLoopManager = nullptr;
   std::shared_ptr<T> fObjPtr; ///< Shared pointer encapsulating the wrapped result
   /// Owning pointer to the action that will produce this result.
   /// Ownership is shared with other copies of this ResultPtr.
   std::shared_ptr<RDFInternal::RActionBase> fActionPtr;

   /// Triggers the event loop in the RLoopManager
   void TriggerRun();

   void ThrowIfNull()
   {
      if (fObjPtr == nullptr)
         throw std::runtime_error("Trying to access the contents of a null RResultPtr.");
   }

   RResultPtr(std::shared_ptr<T> objPtr, RDFDetail::RLoopManager *lm,
              std::shared_ptr<RDFInternal::RActionBase> actionPtr)
      : fLoopManager(lm), fObjPtr(std::move(objPtr)), fActionPtr(std::move(actionPtr))
   {
   }

public:
   using Value_t = T;                       ///< Convenience alias to simplify access to proxied type
   static constexpr ULong64_t kOnce = 0ull; ///< Convenience definition to express a callback must be executed once

   RResultPtr() = default;
   RResultPtr(const RResultPtr &) = default;
   RResultPtr(RResultPtr &&) = default;
   RResultPtr &operator=(const RResultPtr &) = default;
   RResultPtr &operator=(RResultPtr &&) = default;
   explicit operator bool() const { return bool(fObjPtr); }

   /// Convert a RResultPtr<T2> to a RResultPtr<T>.
   ///
   /// Useful e.g. to store a number of RResultPtr<TH1D> and RResultPtr<TH2D> in a std::vector<RResultPtr<TH1>>.
   /// The requirements on T2 and T are the same as for conversion between std::shared_ptr<T2> and std::shared_ptr<T>.
   template <typename T2,
             std::enable_if_t<std::is_constructible<std::shared_ptr<T>, std::shared_ptr<T2>>::value, int> = 0>
   RResultPtr(const RResultPtr<T2> &r) : fLoopManager(r.fLoopManager), fObjPtr(r.fObjPtr), fActionPtr(r.fActionPtr)
   {
   }

   /// Produce the encapsulated result, and return a shared pointer to it.
   /// If RDataFrame hasn't produced the result yet, triggers the event loop and execution
   /// of all actions booked in the associated RLoopManager.
   /// \note To share a "lazy" handle to the result without running the event loop, copy the RResultPtr.
   std::shared_ptr<T> GetSharedPtr()
   {
      if (fActionPtr != nullptr && !fActionPtr->HasRun())
         TriggerRun();
      return fObjPtr;
   }

   /// Get a const reference to the encapsulated object.
   /// Triggers event loop and execution of all actions booked in the associated RLoopManager.
   const T &GetValue()
   {
      ThrowIfNull();
      return *GetSharedPtr();
   }

   /// Get the pointer to the encapsulated object.
   /// Triggers event loop and execution of all actions booked in the associated RLoopManager.
   /// \note Ownership is not transferred to the caller.
   T *GetPtr() { return GetSharedPtr().get(); }

   /// Get a reference to the encapsulated object.
   /// Triggers event loop and execution of all actions booked in the associated RLoopManager.
   T &operator*()
   {
      ThrowIfNull();
      return *GetSharedPtr();
   }

   /// Get a pointer to the encapsulated object.
   /// Triggers event loop and execution of all actions booked in the associated RLoopManager.
   /// \note Ownership is not transferred to the caller.
   T *operator->()
   {
      ThrowIfNull();
      return GetSharedPtr().get();
   }

   /// Return an iterator to the beginning of the contained object if this makes
   /// sense, throw a compilation error otherwise
   typename RIterationHelper<T>::Iterator_t begin()
   {
      ThrowIfNull();
      if (!fActionPtr->HasRun())
         TriggerRun();
      return RIterationHelper<T>::GetBegin(*fObjPtr);
   }

   /// Return an iterator to the end of the contained object if this makes
   /// sense, throw a compilation error otherwise
   typename RIterationHelper<T>::Iterator_t end()
   {
      ThrowIfNull();
      if (!fActionPtr->HasRun())
         TriggerRun();
      return RIterationHelper<T>::GetEnd(*fObjPtr);
   }

   // clang-format off
   /// Register a callback that RDataFrame will execute "everyNEvents" on a partial result.
   ///
   /// \param[in] everyNEvents Frequency at which the callback will be called, as a number of events processed
   /// \param[in] callback a callable with signature `void(Value_t&)` where Value_t is the type of the value contained in this RResultPtr
   /// \return this RResultPtr, to allow chaining of OnPartialResultSlot with other calls
   ///
   /// The callback must be a callable (lambda, function, functor class...) that takes a reference to the result type as
   /// argument and returns nothing. RDataFrame will invoke registered callbacks passing partial action results as
   /// arguments to them (e.g. a histogram filled with a part of the selected events, a counter incremented only up to a
   /// certain point, a mean over a subset of the events and so forth).
   ///
   /// Callbacks can be used e.g. to inspect partial results of the analysis while the event loop is running. For
   /// example one can draw an up-to-date version of a result histogram every 100 entries like this:
   /// \code{.cpp}
   /// auto h = tdf.Histo1D("x");
   /// TCanvas c("c","x hist");
   /// h.OnPartialResult(100, [&c](TH1D &h_) { c.cd(); h_.Draw(); c.Update(); });
   /// h->Draw(); // event loop runs here, this `Draw` is executed after the event loop is finished
   /// \endcode
   ///
   /// A value of 0 for everyNEvents indicates the callback must be executed only once, before running the event loop.
   /// A conveniece definition `kOnce` is provided to make this fact more expressive in user code (see snippet below).
   /// Multiple callbacks can be registered with the same RResultPtr (i.e. results of RDataFrame actions) and will
   /// be executed sequentially. Callbacks are executed in the order they were registered.
   /// The type of the value contained in a RResultPtr is also available as RResultPtr<T>::Value_t, e.g.
   /// \code{.cpp}
   /// auto h = tdf.Histo1D("x");
   /// // h.kOnce is 0
   /// // decltype(h)::Value_t is TH1D
   /// \endcode
   ///
   /// When implicit multi-threading is enabled, the callback:
   /// - will never be executed by multiple threads concurrently: it needs not be thread-safe. For example the snippet
   ///   above that draws the partial histogram on a canvas works seamlessly in multi-thread event loops.
   /// - will be executed by the first worker that arrives at the callback, and then only be executed when the same result
   ///   is updated. For example, if dataframe uses N internal copies of a result, it will always be the `i`th < N object
   ///   that is passed into the callback.
   /// - will always be executed "everyNEvents": the partial result passed into the callback will have accumulated N more
   ///   events, irrespective of the progress that other worker threads make.
   /// - might be executed by a different worker thread at different times: the value of `std::this_thread::get_id()`
   ///   might change between calls.
   ///
   /// To register a callback that is called by _each_ worker thread (concurrently) every N events one can use
   /// OnPartialResultSlot().
   // clang-format on
   RResultPtr<T> &OnPartialResult(ULong64_t everyNEvents, std::function<void(T &)> callback)
   {
      ThrowIfNull();
      auto actionPtr = fActionPtr;
      constexpr auto kUninit = std::numeric_limits<unsigned int>::max();
      auto activeSlot = std::make_shared<std::atomic_uint>(kUninit);
      auto c = [=](unsigned int slot) {
         if (activeSlot->load() == kUninit) {
            // Try to grab the right to run the callback for our slot:
            unsigned int expected = kUninit;
            activeSlot->compare_exchange_strong(expected, slot);
         }
         if (activeSlot->load() != slot)
            return;

         auto partialResult = static_cast<Value_t *>(actionPtr->PartialUpdate(slot));
         callback(*partialResult);
      };
      fLoopManager->RegisterCallback(everyNEvents, std::move(c));
      return *this;
   }

   // clang-format off
   /// Register a callback that RDataFrame will execute in each worker thread concurrently on that thread's partial result.
   ///
   /// \param[in] everyNEvents Frequency at which the callback will be called by each thread, as a number of events processed
   /// \param[in] callback A callable with signature `void(unsigned int, Value_t&)` where Value_t is the type of the value contained in this RResultPtr
   /// \return this RResultPtr, to allow chaining of OnPartialResultSlot with other calls
   ///
   /// See `OnPartialResult` for a generic explanation of the callback mechanism.
   /// Compared to `OnPartialResult`, this method has two major differences:
   /// - all worker threads invoke the callback once every specified number of events. The event count is per-thread,
   ///   and callback invocation might happen concurrently (i.e. the callback must be thread-safe)
   /// - the callable must take an extra `unsigned int` parameter corresponding to a multi-thread "processing slot":
   ///   this is a "helper value" to simplify writing thread-safe callbacks: different worker threads might invoke the
   ///   callback concurrently but always with different `slot` numbers.
   /// - a value of 0 for everyNEvents indicates the callback must be executed once _per slot_.
   ///
   /// For example, the following snippet prints out a thread-safe progress bar of the events processed by RDataFrame
   /// \code
   /// auto c = tdf.Count(); // any action would do, but `Count` is the most lightweight
   /// std::string progress;
   /// std::mutex bar_mutex;
   /// c.OnPartialResultSlot(nEvents / 100, [&progress, &bar_mutex](unsigned int, ULong64_t &) {
   ///    std::lock_guard<std::mutex> lg(bar_mutex);
   ///    progress.push_back('#');
   ///    std::cout << "\r[" << std::left << std::setw(100) << progress << ']' << std::flush;
   /// });
   /// std::cout << "Analysis running..." << std::endl;
   /// *c; // trigger the event loop by accessing an action's result
   /// std::cout << "\nDone!" << std::endl;
   /// \endcode
   // clang-format on
   RResultPtr<T> &OnPartialResultSlot(ULong64_t everyNEvents, std::function<void(unsigned int, T &)> callback)
   {
      ThrowIfNull();
      auto actionPtr = fActionPtr;
      auto c = [actionPtr, callback](unsigned int slot) {
         auto partialResult = static_cast<Value_t *>(actionPtr->PartialUpdate(slot));
         callback(slot, *partialResult);
      };
      fLoopManager->RegisterCallback(everyNEvents, std::move(c));
      return *this;
   }

   // clang-format off
   /// Check whether the result has already been computed
   ///
   /// ~~~{.cpp}
   /// auto res = df.Count();
   /// res.IsReady(); // false, access will trigger event loop
   /// std::cout << *res << std::endl; // triggers event loop
   /// res.IsReady(); // true
   /// ~~~
   // clang-format on
   bool IsReady() const
   {
      if (fActionPtr == nullptr)
         return false;
      return fActionPtr->HasRun();
   }
};

template <typename T>
void RResultPtr<T>::TriggerRun()
{
   fLoopManager->Run();
}

template <class T1, class T2>
bool operator==(const RResultPtr<T1> &lhs, const RResultPtr<T2> &rhs)
{
   return lhs.fObjPtr == rhs.fObjPtr;
}

template <class T1, class T2>
bool operator!=(const RResultPtr<T1> &lhs, const RResultPtr<T2> &rhs)
{
   return lhs.fObjPtr != rhs.fObjPtr;
}

template <class T1>
bool operator==(const RResultPtr<T1> &lhs, std::nullptr_t rhs)
{
   return lhs.fObjPtr == rhs;
}

template <class T1>
bool operator==(std::nullptr_t lhs, const RResultPtr<T1> &rhs)
{
   return lhs == rhs.fObjPtr;
}

template <class T1>
bool operator!=(const RResultPtr<T1> &lhs, std::nullptr_t rhs)
{
   return lhs.fObjPtr != rhs;
}

template <class T1>
bool operator!=(std::nullptr_t lhs, const RResultPtr<T1> &rhs)
{
   return lhs != rhs.fObjPtr;
}

} // namespace RDF

namespace Detail {
namespace RDF {
/// Create a RResultPtr and set its pointer to the corresponding RAction
/// This overload is invoked by non-jitted actions, as they have access to RAction before constructing RResultPtr.
template <typename T>
RResultPtr<T>
MakeResultPtr(const std::shared_ptr<T> &r, RLoopManager &lm, std::shared_ptr<RDFInternal::RActionBase> actionPtr)
{
   return RResultPtr<T>(r, &lm, std::move(actionPtr));
}

////////////////////////////////////////////////////////////////////////////////
/// \brief Retrieve a mergeable value from an RDataFrame action.
/// \param[in] rptr lvalue reference of an RResultPtr object.
/// \returns An RMergeableValue holding the result of the action, wrapped in an
///          `std::unique_ptr`.
///
/// This function triggers the execution of the RDataFrame computation graph.
/// Then retrieves an RMergeableValue object created with the result wrapped by
/// the RResultPtr argument. The user obtains ownership of the mergeable, which
/// in turn holds a copy of the result of the action. The RResultPtr is not
/// destroyed in the process and will still retain (shared) ownership of the
/// original result.
///
/// Example usage:
/// ~~~{.cpp}
/// using namespace ROOT::Detail::RDF;
/// ROOT::RDataFrame d("myTree", "file_*.root");
/// auto h = d.Histo1D("Branch_A");
/// auto mergeablehisto = GetMergeableValue(h);
/// ~~~
template <typename T>
std::unique_ptr<RMergeableValue<T>> GetMergeableValue(RResultPtr<T> &rptr)
{
   rptr.ThrowIfNull();
   if (!rptr.fActionPtr->HasRun())
      rptr.TriggerRun(); // Prevents from using `const` specifier in parameter
   return std::unique_ptr<RMergeableValue<T>>{
      static_cast<RMergeableValue<T> *>(rptr.fActionPtr->GetMergeableValue().release())};
}
} // namespace RDF
} // namespace Detail
} // namespace ROOT

#endif // ROOT_TRESULTPROXY
