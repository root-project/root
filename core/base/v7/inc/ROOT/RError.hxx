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

#include <ROOT/RConfig.hxx>

#include <cstddef>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

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
   RError(const std::string &message, const std::string &func, const std::string &file, int line);
   /// Used by R__FORWARD_RESULT
   void AddFrame(const std::string &func, const std::string &file, int line);
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

// clang-format off
/**
\class ROOT::Experimental::RResult
\ingroup Base
\brief The class is used as a return type for operations that can fail; wraps a value of type T or an RError

The RResult enforces checking whether it contains a valid value or an error state. If the RResult leaves the scope
unchecked, it will throw an exception.  RResult can only be allocated on the stack, which is enforced by deleting the
new operator.  RResult is movable but not copyable to avoid throwing exceptions multiple times.
*/
// clang-format on
template <typename T>
class RResult {
private:
   T fValue;
   std::unique_ptr<RError> fError;
   /// This is safe because checking an RResult is not a multi-threaded operation
   mutable bool fIsChecked{false};

public:
   RResult(const T &value) : fValue(value) {}
   RResult(std::unique_ptr<RError> error) : fError(std::move(error)) {}

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

   const T &Get()
   {
      if (R__unlikely(fError)) {
         fError->AppendMessage(" (invalid access)");
         throw RException(*fError);
      }
      return fValue;
   }
   explicit operator bool() const
   {
      fIsChecked = true;
      return !fError;
   }
   RError *GetError() { return fError.get(); }
   void Throw() { throw RException(*fError); }

   // Prevent heap construction of RStatus objects
   void *operator new(std::size_t size) = delete;
   void *operator new(std::size_t, void *) = delete;
   void *operator new[](std::size_t) = delete;
   void *operator new[](std::size_t, void *) = delete;
};

/// Short-hand to return an RResult<T> in an error state
#define R__FAIL(msg) return std::make_unique<ROOT::Experimental::RError>(msg, __func__, __FILE__, __LINE__)
/// Short-hand to return an RResult<T> value from a subroutine to the calling stack frame
#define R__FORWARD_RESULT(res) if (res.GetError()) { res.GetError()->AddFrame(__func__, __FILE__, __LINE__); } \
   return res

} // namespace Experimental
} // namespace ROOT

#endif
