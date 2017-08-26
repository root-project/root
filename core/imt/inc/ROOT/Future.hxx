// @(#)root/thread:$Id$
// Author: Danilo Piparo August 2017

/*************************************************************************
 * Copyright (C) 1995-2017, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_Future
#define ROOT_Future

#include "RConfigure.h"

#include "ROOT/TTaskGroup.hxx"

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
class Future;
}

namespace Detail {
template <typename T>
class FutureImpl {
   template<typename V> friend class Experimental::Future;
protected:
   using TTaskGroup = Experimental::TTaskGroup;
   std::future<T> fStdFut;
   std::unique_ptr<TTaskGroup> fTg {nullptr};

   FutureImpl(std::future<T> &&fut, std::unique_ptr<TTaskGroup> &&tg) : fStdFut(std::move(fut)) {
      fTg = std::move(tg);
   };
   FutureImpl(){};

   FutureImpl(std::future<T> &&fut) : fStdFut(std::move(fut)) {}

   FutureImpl(FutureImpl<T> &&other) { *this = std::move(other); }

   FutureImpl &operator=(std::future<T> &&other)
   {
      fStdFut = std::move(other);
   }

   FutureImpl<T> &operator=(FutureImpl<T> &&other) = default;

public:

   FutureImpl<T> &operator=(FutureImpl<T> &other) = delete;

   FutureImpl(const FutureImpl<T> &other) = delete;

   void wait() { if (fTg) fTg->Wait(); }

   bool valid() const { return fStdFut.valid(); };
};
}

namespace Experimental {

////////////////////////////////////////////////////////////////////////////////
/// A future class. It can wrap an std::future.
template <typename T>
class Future final : public ROOT::Detail::FutureImpl<T> {
template <class Function, class... Args>
   friend Future<typename std::result_of<typename std::decay<Function>::type(typename std::decay<Args>::type...)>::type> Async(Function &&f, Args &&... args);
private:
   Future(std::future<T> &&fut, std::unique_ptr<TTaskGroup> &&tg)
      : ROOT::Detail::FutureImpl<T>(std::forward<std::future<T>>(fut), std::move(tg)){};
public:
   Future(std::future<T> &&fut) : ROOT::Detail::FutureImpl<T>(std::forward<std::future<T>>(fut)) {};

   T get()
   {
      this->wait();
      return this->fStdFut.get();
   }
};
/// \cond
// Two specialisations, for void and T& as for std::future
template <>
class Future<void> final : public ROOT::Detail::FutureImpl<void> {
template <class Function, class... Args>
   friend Future<typename std::result_of<typename std::decay<Function>::type(typename std::decay<Args>::type...)>::type> Async(Function &&f, Args &&... args);
private:
   Future(std::future<void> &&fut, std::unique_ptr<TTaskGroup> &&tg)
      : ROOT::Detail::FutureImpl<void>(std::forward<std::future<void>>(fut), std::move(tg)){};
public:
   Future(std::future<void> &&fut) : ROOT::Detail::FutureImpl<void>(std::forward<std::future<void>>(fut)) {};

   void get()
   {
      this->wait();
      fStdFut.get();
   }
};

template <typename T>
class Future<T &> final : public ROOT::Detail::FutureImpl<T &> {
template <class Function, class... Args>
   friend Future<typename std::result_of<typename std::decay<Function>::type(typename std::decay<Args>::type...)>::type> Async(Function &&f, Args &&... args);
private:
   Future(std::future<T &> &&fut, std::unique_ptr<TTaskGroup> &&tg)
      : ROOT::Detail::FutureImpl<T &>(std::forward<std::future<T &>>(fut), std::move(tg)){};
public:
   Future(std::future<T&> &&fut) : ROOT::Detail::FutureImpl<T&>(std::forward<std::future<T&>>(fut)) {};

   T &get()
   {
      this->wait();
      return this->fStdFut.get();
   }
};
/// \endcond

////////////////////////////////////////////////////////////////////////////////
/// Runs a function asynchronously potentially in a new thread and returns a
/// ROOT Future that will hold the result.
template <class Function, class... Args>
Future<typename std::result_of<typename std::decay<Function>::type(typename std::decay<Args>::type...)>::type>
Async(Function &&f, Args &&... args)
{
   // The return type according to the standard implementation of std::future
   // the long type is due to the fact that we want to be c++11 compatible.
   // A more elegant version would be:
   // std::future<std::result_of_t<std::decay_t<Function>(std::decay_t<Args>...)>>
   using Ret_t = typename std::result_of<typename std::decay<Function>::type(typename std::decay<Args>::type...)>::type;

   auto thisPt = std::make_shared<std::packaged_task<Ret_t()>>(std::bind(f, args...));
   std::unique_ptr<ROOT::Experimental::TTaskGroup> tg(new ROOT::Experimental::TTaskGroup());
   tg->Run([thisPt]() { (*thisPt)(); });

   return ROOT::Experimental::Future<Ret_t>(thisPt->get_future(), std::move(tg));
}
}
}

#endif
#endif