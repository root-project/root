// Author: Enrico Guiraud, Danilo Piparo CERN  03/2017

/*************************************************************************
 * Copyright (C) 1995-2016, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TACTIONRESULTPROXY
#define ROOT_TACTIONRESULTPROXY

#include "TDFUtils.hxx"

#include <memory>

namespace ROOT {

// Fwd declarations
namespace Detail {
class TDataFrameImpl;
}

namespace Experimental {

/// Smart pointer for the return type of actions
/**
\class ROOT::Experimental::TActionResultProxy
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
class TActionResultProxy {
   /// \cond HIDDEN_SYMBOLS
   template <typename V, bool isCont = ROOT::Internal::TDFTraitsUtils::TIsContainer<V>::fgValue>
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
   using SPT_t         = std::shared_ptr<T>;
   using SPTDFI_t      = std::shared_ptr<ROOT::Detail::TDataFrameImpl>;
   using WPTDFI_t      = std::weak_ptr<ROOT::Detail::TDataFrameImpl>;
   using ShrdPtrBool_t = std::shared_ptr<bool>;
   friend class ROOT::Detail::TDataFrameImpl;

   ShrdPtrBool_t fReadiness =
      std::make_shared<bool>(false); ///< State registered also in the TDataFrameImpl until the event loop is executed
   WPTDFI_t fImplWeakPtr;            ///< Points to the TDataFrameImpl at the root of the functional graph
   SPT_t    fObjPtr;                 ///< Shared pointer encapsulating the wrapped result

   /// Triggers the event loop in the TDataFrameImpl instance to which it's associated via the fImplWeakPtr
   void TriggerRun();

   /// Get the pointer to the encapsulated result.
   /// Ownership is not transferred to the caller.
   /// Triggers event loop and execution of all actions booked in the associated TDataFrameImpl.
   T *Get()
   {
      if (!*fReadiness) TriggerRun();
      return fObjPtr.get();
   }

   TActionResultProxy(const SPT_t &objPtr, const ShrdPtrBool_t &readiness, const SPTDFI_t &firstData)
      : fReadiness(readiness), fImplWeakPtr(firstData), fObjPtr(objPtr)
   {
   }

   /// Factory to allow to keep the constructor private
   static TActionResultProxy<T> MakeActionResultProxy(const SPT_t &objPtr, const ShrdPtrBool_t &readiness,
                                                      const SPTDFI_t &firstData)
   {
      return TActionResultProxy(objPtr, readiness, firstData);
   }

public:
   TActionResultProxy() = delete;

   /// Get a reference to the encapsulated object.
   /// Triggers event loop and execution of all actions booked in the associated TDataFrameImpl.
   T &operator*() { return *Get(); }

   /// Get a pointer to the encapsulated object.
   /// Ownership is not transferred to the caller.
   /// Triggers event loop and execution of all actions booked in the associated TDataFrameImpl.
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

} // end NS Experimental

} // end NS ROOT

#endif // ROOT_TACTIONRESULTPROXY
