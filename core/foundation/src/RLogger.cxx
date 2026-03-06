/// \file RLogger.cxx
/// \ingroup Base ROOT7
/// \author Axel Naumann <axel@cern.ch>
/// \date 2015-07-07
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/RLogger.hxx"

#include "TError.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

// pin vtable
ROOT::RLogHandler::~RLogHandler() {}

namespace {

class RLogHandlerDefault : public ROOT::RLogHandler {
public:
   // Returns false if further emission of this log entry should be suppressed.
   bool Emit(const ROOT::RLogEntry &entry) override;
};

inline bool RLogHandlerDefault::Emit(const ROOT::RLogEntry &entry)
{
   constexpr static int numLevels = static_cast<int>(ROOT::ELogLevel::kDebug) + 1;
   int cappedLevel = std::min(static_cast<int>(entry.fLevel), numLevels - 1);
   constexpr static std::array<const char *, numLevels> sTag{
      {"{unset-error-level please report}", "FATAL", "Error", "Warning", "Info", "Debug"}};

   std::stringstream strm;
   auto channel = entry.fChannel;
   if (channel && !channel->GetName().empty())
      strm << '[' << channel->GetName() << "] ";
   strm << sTag[cappedLevel];

   if (!entry.fLocation.fFile.empty())
      strm << " " << entry.fLocation.fFile << ':' << entry.fLocation.fLine;
   if (!entry.fLocation.fFuncName.empty())
      strm << " in " << entry.fLocation.fFuncName;

   static constexpr const int errorLevelOld[] = {kFatal /*unset*/, kFatal, kError, kWarning, kInfo, kInfo /*debug*/};
   (*::GetErrorHandler())(errorLevelOld[cappedLevel], entry.fLevel == ROOT::ELogLevel::kFatal, strm.str().c_str(),
                          entry.fMessage.c_str());
   return true;
}

/// Trim leading and trailing whitespace from a string.
std::string_view TrimWhitespace(std::string_view s)
{
   const auto begin = s.find_first_not_of(" \t\r\n");
   if (begin == std::string_view::npos)
      return {};
   const auto end = s.find_last_not_of(" \t\r\n");
   return s.substr(begin, end - begin + 1);
}

/// Parse a level string such as "Debug", "Debug(3)", "Info", "Warning", "Error", "Fatal".
/// Returns the corresponding ELogLevel. For Debug(N), the returned level is kDebug + N.
ROOT::ELogLevel ParseLogLevel(std::string_view levelStr)
{
   if (levelStr.compare(0, 5, "Debug") == 0) {
      int extra = 0;
      auto parenOpen = levelStr.find('(');
      if (parenOpen != std::string::npos) {
         auto parenClose = levelStr.find(')', parenOpen);
         if (parenClose != std::string::npos) {
            try {
               extra = std::stoi(std::string(levelStr.substr(parenOpen + 1, parenClose - parenOpen - 1)));
            } catch (...) {
               extra = 0;
               ::Warning("ROOT_LOG", "Cannot parse verbosity level in '%s', defaulting to Debug", levelStr.data());
            }
         }
      }
      return ROOT::ELogLevel::kDebug + extra;
   }
   if (levelStr == "Info")
      return ROOT::ELogLevel::kInfo;
   if (levelStr == "Warning")
      return ROOT::ELogLevel::kWarning;
   if (levelStr == "Error")
      return ROOT::ELogLevel::kError;
   if (levelStr == "Fatal")
      return ROOT::ELogLevel::kFatal;

   // Unrecognised string: warn the user and return kUnset so the channel falls back to global.
   ::Warning("ROOT_LOG", "Unrecognized log level '%s', ignoring", levelStr.data());
   return ROOT::ELogLevel::kUnset;
}

/// Parse ROOT_LOG and return a map of channel-name -> verbosity level.
/// Format: "Channel1=Level1,Channel2=Debug(N),..."
std::unordered_map<std::string, ROOT::ELogLevel> ParseRootLogEnvVar()
{
   std::unordered_map<std::string, ROOT::ELogLevel> result;

   const char *envVal = std::getenv("ROOT_LOG");
   if (!envVal)
      return result;

   std::stringstream ss(envVal);
   std::string token;
   while (std::getline(ss, token, ',')) {
      token = TrimWhitespace(token);
      if (token.empty())
         continue;

      auto eq = token.find('=');
      if (eq == std::string::npos)
         continue;

      std::string_view channelName = TrimWhitespace(std::string_view(token).substr(0, eq));
      std::string_view levelStr = TrimWhitespace(std::string_view(token).substr(eq + 1));

      if (channelName.empty() || levelStr.empty())
         continue;

      ROOT::ELogLevel level = ParseLogLevel(levelStr);
      if (level != ROOT::ELogLevel::kUnset)
         result[std::string(channelName)] = level;
   }
   return result;
}

} // unnamed namespace

/// Construct the RLogManager, install the default handler, then apply
/// gDebug and the ROOT_LOG environment variable.
ROOT::RLogManager::RLogManager(std::unique_ptr<RLogHandler> lh) : RLogChannel(ELogLevel::kWarning)
{
   fHandlers.emplace_back(std::move(lh));

   // Apply gDebug as a global verbosity floor.
   // gDebug == 1 maps to kDebug, gDebug == 2 to kDebug+1, etc.
   if (gDebug > 0)
      SetVerbosity(ELogLevel::kDebug + (gDebug - 1));

   // Parse ROOT_LOG and store per-channel overrides for lazy application.
   fEnvVerbosity = ParseRootLogEnvVar();
}

ROOT::RLogManager &ROOT::RLogManager::Get()
{
   static RLogManager instance(std::make_unique<RLogHandlerDefault>());
   return instance;
}

std::unique_ptr<ROOT::RLogHandler> ROOT::RLogManager::Remove(RLogHandler *handler)
{
   auto iter = std::find_if(fHandlers.begin(), fHandlers.end(), [&](const std::unique_ptr<RLogHandler> &handlerPtr) {
      return handlerPtr.get() == handler;
   });
   if (iter != fHandlers.end()) {
      std::unique_ptr<RLogHandler> ret;
      swap(*iter, ret);
      fHandlers.erase(iter);
      return ret;
   }
   return {};
}

bool ROOT::RLogManager::Emit(const ROOT::RLogEntry &entry)
{
   auto channel = entry.fChannel;

   Increment(entry.fLevel);
   if (channel != this)
      channel->Increment(entry.fLevel);

   // Is there a specific level for the channel? If so, take that,
   // overruling the global one.
   if (channel->GetEffectiveVerbosity(*this) < entry.fLevel)
      return true;

   // Lock-protected extraction of handlers, such that they don't get added during the
   // handler iteration.
   std::vector<RLogHandler *> handlers;
   {
      std::lock_guard<std::mutex> lock(fMutex);
      handlers.resize(fHandlers.size());
      std::transform(fHandlers.begin(), fHandlers.end(), handlers.begin(),
                     [](const std::unique_ptr<RLogHandler> &handlerUPtr) { return handlerUPtr.get(); });
   }

   for (auto &&handler : handlers)
      if (!handler->Emit(entry))
         return false;
   return true;
}