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

#ifndef ROOT7_TLog
#define ROOT7_TLog

#include <array>
#include <memory>
#include <sstream>
#include "RStringView.h"
#include <vector>

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

class TLogEntry;

/**
 Abstract TLogHandler base class. ROOT logs everything from info to error
 to entities of this class.
 */
class TLogHandler {
public:
   virtual ~TLogHandler();
   // Returns false if further emission of this Log should be suppressed.
   virtual bool Emit(const TLogEntry &entry) = 0;
};

class TLogManager: public TLogHandler {
private:
   std::vector<std::unique_ptr<TLogHandler>> fHandlers;

   /// Initialize taking a TLogHandlerDefault.
   TLogManager(std::unique_ptr<TLogHandler> &&lh) { fHandlers.emplace_back(std::move(lh)); }

public:
   static TLogManager &Get();

   /// Add a TLogHandler in the front - to be called before all others.
   void PushFront(std::unique_ptr<TLogHandler> handler) { fHandlers.insert(fHandlers.begin(), std::move(handler)); }

   /// Add a TLogHandler in the back - to be called after all others.
   void PushBack(std::unique_ptr<TLogHandler> handler) { fHandlers.emplace_back(std::move(handler)); }

   // Emit a `TLogEntry` to the TLogHandlers.
   // Returns false if further emission of this Log should be suppressed.
   bool Emit(const TLogEntry &entry) override
   {
      for (auto &&handler: fHandlers)
         if (!handler->Emit(entry))
            return false;
      return true;
   }
};

class TLogEntry: public std::ostringstream {
public:
   std::string fGroup;
   std::string fFile;
   std::string fFuncName;
   int fLine = 0;
   ELogLevel fLevel;

public:
   TLogEntry() = default;
   TLogEntry(ELogLevel level, std::string_view group): fGroup(group), fLevel(level) {}
   TLogEntry(ELogLevel level, std::string_view group, std::string_view filename, int line, std::string_view funcname)
      : fGroup(group), fFile(filename), fFuncName(funcname), fLine(line), fLevel(level)
   {}

   TLogEntry &SetFile(const std::string &file)
   {
      fFile = file;
      return *this;
   }
   TLogEntry &SetFunction(const std::string &func)
   {
      fFuncName = func;
      return *this;
   }
   TLogEntry &SetLine(int line)
   {
      fLine = line;
      return *this;
   }

   ~TLogEntry() { TLogManager::Get().Emit(*this); }
};

} // namespace Experimental
} // namespace ROOT

#define R__LOG_HERE(LEVEL, GROUP) \
   ROOT::Experimental::TLogEntry(LEVEL, GROUP).SetFile(__FILE__).SetLine(__LINE__).SetFunction(__PRETTY_FUNCTION__)

#define R__FATAL_HERE(GROUP) R__LOG_HERE(ROOT::Experimental::ELogLevel::kFatal, GROUP)
#define R__ERROR_HERE(GROUP) R__LOG_HERE(ROOT::Experimental::ELogLevel::kError, GROUP)
#define R__WARNING_HERE(GROUP) R__LOG_HERE(ROOT::Experimental::ELogLevel::kWarning, GROUP)
#define R__INFO_HERE(GROUP) R__LOG_HERE(ROOT::Experimental::ELogLevel::kInfo, GROUP)
#define R__DEBUG_HERE(GROUP) R__LOG_HERE(ROOT::Experimental::ELogLevel::kDebug, GROUP)

#endif
