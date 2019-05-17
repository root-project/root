/// \file ROOT/TLogger.h
/// \ingroup Base ROOT7
/// \author Axel Naumann <axel@cern.ch>
/// \date 2015-03-29
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2015, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RLogger
#define ROOT7_RLogger

#include <array>
#include <memory>
#include <sstream>
#include <vector>

#include "ROOT/RStringView.hxx"

namespace ROOT {
namespace Experimental {

/**
 Kinds of diagnostics.
 */
enum class ELogLevel {
   kDebug,   ///< Debug information; only useful for developers
   kInfo,    ///< Informational messages; used for instance for tracing
   kWarning, ///< Warnings about likely unexpected behavior
   kError,
   kFatal
};

class RLogEntry;

/**
 Abstract RLogHandler base class. ROOT logs everything from info to error
 to entities of this class.
 */
class RLogHandler {
public:
   virtual ~RLogHandler();
   /// Emit a log entry.
   /// \param entry - the RLogEntry to be emitted.
   /// \returns false if further emission of this Log should be suppressed.
   ///
   /// \note This function is called concurrently; log emission must be locked
   /// if needed. (The default log handler using ROOT's DefaultErrorHandler is locked.)
   virtual bool Emit(const RLogEntry &entry) = 0;
};


/**
 A RLogHandler that multiplexes diagnostics to different client `RLogHandler`s.
 `RLogHandler::Get()` returns the process's (static) log manager.
 */

class RLogManager: public RLogHandler {
private:
   std::vector<std::unique_ptr<RLogHandler>> fHandlers;

   long long fNumWarnings{0};
   long long fNumErrors{0};

   /// Initialize taking a RLogHandlerDefault.
   RLogManager(std::unique_ptr<RLogHandler> &&lh) { fHandlers.emplace_back(std::move(lh)); }

public:
   static RLogManager &Get();

   /// Add a RLogHandler in the front - to be called before all others.
   void PushFront(std::unique_ptr<RLogHandler> handler) { fHandlers.insert(fHandlers.begin(), std::move(handler)); }

   /// Add a RLogHandler in the back - to be called after all others.
   void PushBack(std::unique_ptr<RLogHandler> handler) { fHandlers.emplace_back(std::move(handler)); }

   // Emit a `RLogEntry` to the RLogHandlers.
   // Returns false if further emission of this Log should be suppressed.
   bool Emit(const RLogEntry &entry) override
   {
      for (auto &&handler: fHandlers)
         if (!handler->Emit(entry))
            return false;
      return true;
   }

   /// Returns the current number of warnings seen by this log manager.
   long long GetNumWarnings() const { return fNumWarnings; }

   /// Returns the current number of errors seen by this log manager.
   long long GetNumErrors() const { return fNumErrors; }
};

/**
 Object to count the number of warnings and errors emitted by a section of code,
 after construction of this type.
 */
class RLogDiagCounter {
private:
   /// The number of the RLogManager's emitted warnings at construction time of *this.
   long long fInitialWarnings{RLogManager::Get().GetNumWarnings()};
   /// The number of the RLogManager's emitted errors at construction time.
   long long fInitialErrors{RLogManager::Get().GetNumErrors()};

public:
   /// Get the number of warnings that the RLogManager has emitted since construction of *this.
   long long GetAccumulatedWarnings() const { return RLogManager::Get().GetNumWarnings() - fInitialWarnings; }

   /// Get the number of errors that the RLogManager has emitted since construction of *this.
   long long GetAccumulatedErrors() const { return RLogManager::Get().GetNumErrors() - fInitialErrors; }

   /// Whether the RLogManager has emitted a warnings since construction time of *this.
   bool HasWarningOccurred() const { return GetAccumulatedWarnings(); }

   /// Whether the RLogManager has emitted an error since construction time of *this.
   bool HasErrorOccurred() const { return GetAccumulatedErrors(); }

   /// Whether the RLogManager has emitted an error or a warning since construction time of *this.
   bool HasErrorOrWarningOccurred() const { return HasWarningOccurred() || HasErrorOccurred(); }
};

/**
 A diagnostic, emitted by the RLogManager upon destruction of the RLogEntry.
 One can construct a RLogEntry through the utility preprocessor macros R__ERROR_HERE, R__WARNING_HERE etc
 like this:
     R__INFO_HERE("CodeGroupForInstanceLibrary") << "All we know is " << 42;
 This will automatically capture the current class and function name, the file and line number.
 */

class RLogEntry: public std::ostringstream {
public:
   std::string fGroup;
   std::string fFile;
   std::string fFuncName;
   int fLine = 0;
   ELogLevel fLevel;

public:
   RLogEntry() = default;
   RLogEntry(ELogLevel level, std::string_view group): fGroup(group), fLevel(level) {}
   RLogEntry(ELogLevel level, std::string_view group, std::string_view filename, int line, std::string_view funcname)
      : fGroup(group), fFile(filename), fFuncName(funcname), fLine(line), fLevel(level)
   {}

   RLogEntry &SetFile(const std::string &file)
   {
      fFile = file;
      return *this;
   }
   RLogEntry &SetFunction(const std::string &func)
   {
      fFuncName = func;
      return *this;
   }
   RLogEntry &SetLine(int line)
   {
      fLine = line;
      return *this;
   }

   ~RLogEntry() { RLogManager::Get().Emit(*this); }
};

} // namespace Experimental
} // namespace ROOT

#if defined(_MSC_VER)
#define R__LOG_PRETTY_FUNCTION __FUNCSIG__
#else
#define R__LOG_PRETTY_FUNCTION __PRETTY_FUNCTION__
#endif

#define R__LOG_HERE(LEVEL, GROUP) \
   ROOT::Experimental::RLogEntry(LEVEL, GROUP).SetFile(__FILE__).SetLine(__LINE__).SetFunction(R__LOG_PRETTY_FUNCTION)

#define R__FATAL_HERE(GROUP) R__LOG_HERE(ROOT::Experimental::ELogLevel::kFatal, GROUP)
#define R__ERROR_HERE(GROUP) R__LOG_HERE(ROOT::Experimental::ELogLevel::kError, GROUP)
#define R__WARNING_HERE(GROUP) R__LOG_HERE(ROOT::Experimental::ELogLevel::kWarning, GROUP)
#define R__INFO_HERE(GROUP) R__LOG_HERE(ROOT::Experimental::ELogLevel::kInfo, GROUP)
#define R__DEBUG_HERE(GROUP) R__LOG_HERE(ROOT::Experimental::ELogLevel::kDebug, GROUP)

#endif
