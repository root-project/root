// @(#)root/thread:$Id$
// Author: Danilo Piparo August 2017

/*************************************************************************
 * Copyright (C) 1995-2017, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TFuture
#define ROOT_TFuture

#include "RConfigure.h"

#include "ROOT/TTaskGroup.hxx"
#include "ROOT/TypeTraits.hxx"

#include <type_traits>
#include <future>

// exclude in case ROOT does not have IMT support
#ifndef R__USE_IMT
// No need to error out for dictionaries.
#if !defined(__ROOTCLING__) && !defined(G__DICTIONARY)
#error "Cannot use ROOT::Experimental::Async without defining R__USE_IMT."
#endif
#else

namespace ROOT {

// fwd declaration
namespace Experimental {
template <typename T>
class TFuture;
}

namespace Detail {
template <typename T>
class TFutureImpl {
   template <typename V>
   friend class Experimental::TFuture;

protected:
   using TTaskGroup = Experimental::TTaskGroup;
   std::future<T> fStdFut;
   std::unique_ptr<TTaskGroup> fTg{nullptr};

   TFutureImpl(std::future<T> &&fut, std::unique_ptr<TTaskGroup> &&tg) : fStdFut(std::move(fut))
   {
      fTg = std::move(tg);
   };
   TFutureImpl(){};

   TFutureImpl(std::future<T> &&fut) : fStdFut(std::move(fut)) {}

   TFutureImpl(TFutureImpl<T> &&other) : fStdFut(std::move(other.fStdFut)), fTg(std::move(other.fTg)) {}

   TFutureImpl &operator=(std::future<T> &&other) { fStdFut = std::move(other); }

   TFutureImpl<T> &operator=(TFutureImpl<T> &&other) = default;

public:
   TFutureImpl<T> &operator=(TFutureImpl<T> &other) = delete;

   TFutureImpl(const TFutureImpl<T> &other) = delete;

   void wait()
   {
      if (fTg)
         fTg->Wait();
   }

   bool valid() const { return fStdFut.valid(); };
};
}

namespace Experimental {

////////////////////////////////////////////////////////////////////////////////
/// A TFuture class. It can wrap an std::future.
template <typename T>
class TFuture final : public ROOT::Detail::TFutureImpl<T> {
   template <class Function, class... Args>
   friend TFuture<
      ROOT::TypeTraits::InvokeResult_t<typename std::decay<Function>::type, typename std::decay<Args>::type...>>
   Async(Function &&f, Args &&...args);

private:
   TFuture(std::future<T> &&fut, std::unique_ptr<TTaskGroup> &&tg)
      : ROOT::Detail::TFutureImpl<T>(std::forward<std::future<T>>(fut), std::move(tg)){};

public:
   TFuture(std::future<T> &&fut) : ROOT::Detail::TFutureImpl<T>(std::forward<std::future<T>>(fut)){};

   T get()
   {
      this->wait();
      return this->fStdFut.get();
   }
};
/// \cond
// Two specialisations, for void and T& as for std::future
template <>
class TFuture<void> final : public ROOT::Detail::TFutureImpl<void> {
   template <class Function, class... Args>
   friend TFuture<
      ROOT::TypeTraits::InvokeResult_t<typename std::decay<Function>::type, typename std::decay<Args>::type...>>
   Async(Function &&f, Args &&...args);

private:
   TFuture(std::future<void> &&fut, std::unique_ptr<TTaskGroup> &&tg)
      : ROOT::Detail::TFutureImpl<void>(std::forward<std::future<void>>(fut), std::move(tg)){};

public:
   TFuture(std::future<void> &&fut) : ROOT::Detail::TFutureImpl<void>(std::forward<std::future<void>>(fut)){};

   void get()
   {
      this->wait();
      fStdFut.get();
   }
};

template <typename T>
class TFuture<T &> final : public ROOT::Detail::TFutureImpl<T &> {
   template <class Function, class... Args>
   friend TFuture<
      ROOT::TypeTraits::InvokeResult_t<typename std::decay<Function>::type, typename std::decay<Args>::type...>>
   Async(Function &&f, Args &&...args);

private:
   TFuture(std::future<T &> &&fut, std::unique_ptr<TTaskGroup> &&tg)
      : ROOT::Detail::TFutureImpl<T &>(std::forward<std::future<T &>>(fut), std::move(tg)){};

public:
   TFuture(std::future<T &> &&fut) : ROOT::Detail::TFutureImpl<T &>(std::forward<std::future<T &>>(fut)){};

   T &get()
   {
      this->wait();
      return this->fStdFut.get();
   }
};
/// \endcond

////////////////////////////////////////////////////////////////////////////////
/// Runs a function asynchronously potentially in a new thread and returns a
/// ROOT TFuture that will hold the result.
template <class Function, class... Args>
TFuture<ROOT::TypeTraits::InvokeResult_t<typename std::decay<Function>::type, typename std::decay<Args>::type...>>
Async(Function &&f, Args &&...args)
{
   // The return type according to the standard implementation of std::future
   using Ret_t = ROOT::TypeTraits::InvokeResult_t<std::decay_t<Function>, std::decay_t<Args>...>;

   auto thisPt = std::make_shared<std::packaged_task<Ret_t()>>(std::bind(f, args...));
   std::unique_ptr<ROOT::Experimental::TTaskGroup> tg(new ROOT::Experimental::TTaskGroup());
   tg->Run([thisPt]() { (*thisPt)(); });

   return ROOT::Experimental::TFuture<Ret_t>(thisPt->get_future(), std::move(tg));
}
}
}

#endif
#endif
