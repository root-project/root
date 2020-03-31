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
#include <ROOT/RLogger.hxx> // for R__LOG_PRETTY_FUNCTION, R__WARNING_HERE

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
//        return R__FORWARD_RESULT(FuncThatReturnsRResultOfInt());
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
//        return RResult<void>::Success();
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
      RLocation(const char *func, const char *file, int line)
         : fFunction(func), fSourceFile(file), fSourceLine(line) {}

      // TODO(jblomer) use std::source_location once available
      const char *fFunction;
      const char *fSourceFile;
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
   void AppendToMessage(const std::string &info) { fMessage += info; }
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
class RResultType {
protected:
   T fValue;
   explicit RResultType() = default;
   explicit RResultType(const T &value) : fValue(value) {}
   explicit RResultType(T &&value) : fValue(std::move(value)) {}
};

template <>
class RResultType<void> { };
} // namespace Internal


// clang-format off
/**
\class ROOT::Experimental::RResult
\ingroup Base
\brief The class is used as a return type for operations that can fail; wraps a value of type T or an RError

RResult enforces checking whether it contains a valid value or an error state. If the RResult leaves the scope
unchecked, it will throw an exception.  RResult should only be allocated on the stack, which is helped by deleting the
new operator.  RResult is movable but not copyable to avoid throwing multiple exceptions about the same failure.
*/
// clang-format on
template <typename T>
class RResult : public Internal::RResultType<T> {
private:
   /// This is the nullptr for an RResult representing success
   std::unique_ptr<RError> fError;
   /// Switches to true once the user of an RResult object checks the object status
   /// Declaring it mutable is safe because checking an RResult is not a multi-threaded operation
   /// The alternative, making the bool operator non-const, has unwanted effects when using an RResult, e.g.
   ///     auto res = Func();
   ///     ASSERT_TRUE(res);
   /// would not work anymore
   mutable bool fIsChecked{false};

   RResult() = default;

public:
   /// Returns a RResult<void> that captures the successful execution of the function
   template <typename Dummy = T, typename = typename std::enable_if_t<std::is_void<T>::value, Dummy>>
   static RResult Success() { return RResult(); }
   /// Constructor is _not_ explicit in order to allow for `return T();` for functions returning RResult<T>
   /// Only available if T is not void
   template <typename Dummy = T, typename = typename std::enable_if_t<!std::is_void<T>::value, Dummy>>
   RResult(const Dummy &value) : Internal::RResultType<T>(value) { }
   template <typename Dummy = T, typename = typename std::enable_if_t<!std::is_void<T>::value, Dummy>>
   RResult(Dummy &&value) : Internal::RResultType<T>(std::move(value)) { }
   /// Constructor is _not_ explicit such that the RError returned by R__FAIL can be converted into an RResult<T>
   /// for any T
   RResult(RError &&error) : fError(std::make_unique<RError>(std::move(error))) {}

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
         } else {
            R__WARNING_HERE("RError") << "unhandled RResult exception during stack unwinding";
         }
      }
   }

   /// Used by R__FORWARD_RESULT in order to keep track of the stack trace in case of errors
   static RResult &Forward(RResult &result, RError::RLocation &&sourceLocation) {
      if (result.fError)
         result.fError->AddFrame(std::move(sourceLocation));
      return result;
   }

   /// Only available if T is not void
   template <typename Dummy = T>
   typename std::enable_if_t<!std::is_void<T>::value, const Dummy &>
   Get()
   {
      if (R__unlikely(fError)) {
         // Get() can be wrapped in a try-catch block, so throwing the exception here is akin to checking the error.
         // Setting fIsChecked to true also avoids a spurious warning in the RResult destructor
         fIsChecked = true;

         fError->AppendToMessage(" (unchecked RResult access!)");
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
   // the latest when leaving the scope. Heap allocated RResult objects in failure state can live much longer making it
   // difficult to trace back the original error.
   void *operator new(std::size_t size) = delete;
   void *operator new(std::size_t, void *) = delete;
   void *operator new[](std::size_t) = delete;
   void *operator new[](std::size_t, void *) = delete;
};

/// Short-hand to return an RResult<T> in an error state; the RError is implicitly converted into RResult<T>
#define R__FAIL(msg) return ROOT::Experimental::RError(msg, {R__LOG_PRETTY_FUNCTION, __FILE__, __LINE__})
/// Short-hand to return an RResult<T> value from a subroutine to the calling stack frame
#define R__FORWARD_RESULT(res) std::move(res.Forward(res, {R__LOG_PRETTY_FUNCTION, __FILE__, __LINE__}))
} // namespace Experimental
} // namespace ROOT

#endif
