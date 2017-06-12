// Author: Enrico Guiraud, Danilo Piparo CERN  03/2017

/*************************************************************************
 * Copyright (C) 1995-2016, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TRESULTPROXY
#define ROOT_TRESULTPROXY

#include "ROOT/TDFNodes.hxx"
#include "ROOT/TDFUtils.hxx"

#include <memory>

namespace ROOT {

namespace Experimental {
namespace TDF {
// Fwd decl for MakeResultProxy
template <typename T>
class TResultProxy;
}
}

namespace Detail {
namespace TDF {
using ROOT::Experimental::TDF::TResultProxy;
// Fwd decl for TResultProxy
template <typename T>
TResultProxy<T> MakeResultProxy(const std::shared_ptr<T> &r, const std::shared_ptr<TLoopManager> &df);
} // ns TDF
} // ns Detail

namespace Experimental {
namespace TDF {
namespace TDFInternal = ROOT::Internal::TDF;
namespace TDFDetail = ROOT::Detail::TDF;

/// Smart pointer for the return type of actions
/**
\class ROOT::Experimental::TDF::TResultProxy
\ingroup dataframe
\brief A wrapper around the result of TDataFrame actions able to trigger calculations lazily.
\tparam T Type of the action result

A smart pointer which allows to access the result of a TDataFrame action. The
methods of the encapsulated object can be accessed via the arrow operator.
Upon invocation of the arrow operator or dereferencing (`operator*`), the
loop on the events and calculations of all scheduled actions are executed
if needed.
It is possible to iterate on the result proxy if the proxied object is a collection.
~~~{.cpp}
for (auto& myItem : myResultProxy) { ... };
~~~
If iteration is not supported by the type of the proxied object, a compilation error is thrown.

*/
template <typename T>
class TResultProxy {
   /// \cond HIDDEN_SYMBOLS
   template <typename V, bool isCont = TDFInternal::TIsContainer<V>::fgValue>
   struct TIterationHelper {
      using Iterator_t = void;
      void GetBegin(const V &) { static_assert(sizeof(V) == 0, "It does not make sense to ask begin for this class."); }
      void GetEnd(const V &) { static_assert(sizeof(V) == 0, "It does not make sense to ask end for this class."); }
   };

   template <typename V>
   struct TIterationHelper<V, true> {
      using Iterator_t = decltype(std::begin(std::declval<V>()));
      static Iterator_t GetBegin(const V &v) { return std::begin(v); };
      static Iterator_t GetEnd(const V &v) { return std::end(v); };
   };
   /// \endcond
   using SPT_t = std::shared_ptr<T>;
   using SPTLM_t = std::shared_ptr<TDFDetail::TLoopManager>;
   using WPTLM_t = std::weak_ptr<TDFDetail::TLoopManager>;
   using ShrdPtrBool_t = std::shared_ptr<bool>;
   template <typename W>
   friend TResultProxy<W> TDFDetail::MakeResultProxy(const std::shared_ptr<W> &, const SPTLM_t &);

   ShrdPtrBool_t fReadiness =
      std::make_shared<bool>(false); ///< State registered also in the TLoopManager until the event loop is executed
   WPTLM_t fImplWeakPtr;             ///< Points to the TLoopManager at the root of the functional graph
   SPT_t fObjPtr;                    ///< Shared pointer encapsulating the wrapped result

   /// Triggers the event loop in the TLoopManager instance to which it's associated via the fImplWeakPtr
   void TriggerRun();

   /// Get the pointer to the encapsulated result.
   /// Ownership is not transferred to the caller.
   /// Triggers event loop and execution of all actions booked in the associated TLoopManager.
   T *Get()
   {
      if (!*fReadiness) TriggerRun();
      return fObjPtr.get();
   }

   TResultProxy(const SPT_t &objPtr, const ShrdPtrBool_t &readiness, const SPTLM_t &firstData)
      : fReadiness(readiness), fImplWeakPtr(firstData), fObjPtr(objPtr)
   {
   }

public:
   TResultProxy() = delete;

   /// Get a const reference to the encapsulated object.
   /// Triggers event loop and execution of all actions booked in the associated TLoopManager.
   const T &GetValue()
   {
      return *Get();
   }

   /// Get a pointer to the encapsulated object.
   /// Triggers event loop and execution of all actions booked in the associated TLoopManager.
   T &operator*() { return *Get(); }

   /// Get a pointer to the encapsulated object.
   /// Ownership is not transferred to the caller.
   /// Triggers event loop and execution of all actions booked in the associated TLoopManager.
   T *operator->() { return Get(); }

   /// Return an iterator to the beginning of the contained object if this makes
   /// sense, throw a compilation error otherwise
   typename TIterationHelper<T>::Iterator_t begin()
   {
      if (!*fReadiness) TriggerRun();
      return TIterationHelper<T>::GetBegin(*fObjPtr);
   }

   /// Return an iterator to the end of the contained object if this makes
   /// sense, throw a compilation error otherwise
   typename TIterationHelper<T>::Iterator_t end()
   {
      if (!*fReadiness) TriggerRun();
      return TIterationHelper<T>::GetEnd(*fObjPtr);
   }
};

template <typename T>
void TResultProxy<T>::TriggerRun()
{
   auto df = fImplWeakPtr.lock();
   if (!df) {
      throw std::runtime_error("The main TDataFrame is not reachable: did it go out of scope?");
   }
   df->Run();
}
} // end NS TDF
} // end NS Experimental

namespace Detail {
namespace TDF {
template <typename T>
TResultProxy<T> MakeResultProxy(const std::shared_ptr<T> &r, const std::shared_ptr<TLoopManager> &df)
{
   auto readiness = std::make_shared<bool>(false);
   auto resPtr = TResultProxy<T>(r, readiness, df);
   df->Book(readiness);
   return resPtr;
}
} // end NS TDF
} // end NS Detail
} // end NS ROOT

#endif // ROOT_TRESULTPROXY
