/// \file ROOT/RError.h
/// \ingroup Base ROOT7
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2019-12-11
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RError
#define ROOT7_RError

#include <ROOT/RConfig.hxx> // for R__[un]likely
#include <ROOT/RLogger.hxx> // for R__LOG_PRETTY_FUNCTION

#include <cstddef>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

// The RResult<T> class and their related classes are used for call chains that can throw exceptions,
// such as I/O code paths.  Throwing of the exception is deferred to allow for `if (result)` style error
// checking where it makes sense.
//
// A function returning an RResult might look like this:
//
//     RResult<int> MyIOFunc()
//     {
//        int rv = syscall(...);
//        if (rv == -1)
//           R__FAIL("user-facing error message");
//        if (rv == kShortcut)
//           return 42;
//        R__FORWARD_RESULT(FuncThatReturnsRResultOfInt());
//     }
//
// Code using MyIOFunc might look like this:
//
//     auto result = MyIOOperation();
//     if (!result) {
//        /* custom error handling or result.Throw() */
//     }
//     switch (result.Get()) {
//        ...
//     }
//
// Note that RResult<void> can be used for a function without return value, like this
//
//     RResult<void> DoSomething()
//     {
//        if (failure)
//           R__FAIL("user-facing error messge");
//        R__SUCCESS
//     }


namespace ROOT {
namespace Experimental {

// clang-format off
/**
\class ROOT::Experimental::RError
\ingroup Base
\brief Captures diagnostics related to a ROOT runtime error
*/
// clang-format on
class RError {
public:
   struct RLocation {
      RLocation() = default;
      RLocation(const std::string &func, const std::string &file, int line)
         : fFunction(func), fSourceFile(file), fSourceLine(line) {}

      // TODO(jblomer) use std::source_location once available
      std::string fFunction;
      std::string fSourceFile;
      int fSourceLine;
   };

private:
   /// User-facing error message
   std::string fMessage;
   /// The location of the error related to fMessage plus upper frames if the error is forwarded through the call stack
   std::vector<RLocation> fStackTrace;

public:
   /// Used by R__FAIL
   RError(const std::string &message, RLocation &&sourceLocation);
   /// Used by R__FORWARD_RESULT
   void AddFrame(RLocation &&sourceLocation);
   /// Add more information to the diagnostics
   void AppendMessage(const std::string &info) { fMessage += info; }
   /// Format a dignostics report, e.g. for an exception message
   std::string GetReport() const;
   const std::vector<RLocation> &GetStackTrace() const { return fStackTrace; }
};

// clang-format off
/**
\class ROOT::Experimental::RException
\ingroup Base
\brief Base class for all ROOT issued exceptions
*/
// clang-format on
class RException : public std::runtime_error {
   RError fError;
public:
   explicit RException(const RError &error) : std::runtime_error(error.GetReport()), fError(error) {}
   const RError &GetError() const { return fError; }
};


/// Wrapper class that generates a data member of type T in RResult<T> for all Ts except T == void
namespace Internal {
template <typename T>
struct RResultType {
   RResultType() = default;
   explicit RResultType(const T &value) : fValue(value) {}
   T fValue;
};

template <>
struct RResultType<void> {};
} // namespace Internal


// clang-format off
/**
\class ROOT::Experimental::RResult
\ingroup Base
\brief The class is used as a return type for operations that can fail; wraps a value of type T or an RError

The RResult enforces checking whether it contains a valid value or an error state. If the RResult leaves the scope
unchecked, it will throw an exception.  RResult should only be allocated on the stack, which is helped by deleting the
new operator.  RResult is movable but not copyable to avoid throwing exceptions multiple times.
*/
// clang-format on
template <typename T>
class RResult : public Internal::RResultType<T> {
private:
   /// This is the nullptr for an RResult representing success
   std::unique_ptr<RError> fError;
   /// This is safe because checking an RResult is not a multi-threaded operation
   mutable bool fIsChecked{false};

public:
   /// Constructor is _not_ explicit in order to allow for `return T();` for functions returning RResult<T>
   /// Only available if T is not void
   template <typename Dummy = T, typename = typename std::enable_if_t<!std::is_void<T>::value, Dummy>>
   RResult(const Dummy &value) : Internal::RResultType<T>(value) {}
   /// Constructor is _not_ explicit such that the RError returned by R__FAIL can be converted into an RResult<T>
   /// for any T
   RResult(const RError &error) : fError(std::make_unique<RError>(error)) {}

   /// RResult<void> has a default contructor that creates an object representing success
   template <typename Dummy = T, typename = typename std::enable_if_t<std::is_void<T>::value, Dummy>>
   RResult() {}

   RResult(const RResult &other) = delete;
   RResult(RResult &&other) = default;
   RResult &operator =(const RResult &other) = delete;
   RResult &operator =(RResult &&other) = default;

   ~RResult() noexcept(false)
   {
      if (R__unlikely(fError && !fIsChecked)) {
         // Prevent from throwing if the object is deconstructed in the course of stack unwinding for another exception
#if __cplusplus >= 201703L
         if (std::uncaught_exceptions() == 0)
#else
         if (!std::uncaught_exception())
#endif
         {
            throw RException(*fError);
         }
      }
   }

   /// Only available if T is not void
   template <typename Dummy = T>
   typename std::enable_if_t<!std::is_void<T>::value, const Dummy &>
   Get()
   {
      if (R__unlikely(fError)) {
         fError->AppendMessage(" (invalid access)");
         throw RException(*fError);
      }
      return Internal::RResultType<T>::fValue;
   }

   explicit operator bool() const
   {
      fIsChecked = true;
      return !fError;
   }

   RError *GetError() { return fError.get(); }

   void Throw() { throw RException(*fError); }

   // Help to prevent heap construction of RResult objects. Unchecked RResult objects in failure state should throw
   // an exception close to the error location. For stack allocated RResult objects, an exception is thrown
   // the latest when leaving the scope. Heap allocated ill RResult objects can live much longer making it difficult
   // to trace back the original failure.
   void *operator new(std::size_t size) = delete;
   void *operator new(std::size_t, void *) = delete;
   void *operator new[](std::size_t) = delete;
   void *operator new[](std::size_t, void *) = delete;
};

/// Short-hand to return an RResult<void> indicating success
#define R__SUCCESS return ROOT::Experimental::RResult<void>();
/// Short-hand to return an RResult<T> in an error state; the RError is implicitly converted into RResult<T>
#define R__FAIL(msg) return ROOT::Experimental::RError(msg, {R__LOG_PRETTY_FUNCTION, __FILE__, __LINE__})
/// Short-hand to return an RResult<T> value from a subroutine to the calling stack frame
#define R__FORWARD_RESULT(res) if (res.GetError()) \
      { res.GetError()->AddFrame({R__LOG_PRETTY_FUNCTION, __FILE__, __LINE__}); } \
   return res

} // namespace Experimental
} // namespace ROOT

#endif
