/// \file RLogger.cxx
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
#include <cctype>
#include <cstdlib>
#include <memory>
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

std::string_view TrimWhitespace(std::string_view sv)
{
   while (!sv.empty() && (sv.front() == ' ' || sv.front() == '\t' || sv.front() == '\r' || sv.front() == '\n'))
      sv.remove_prefix(1);
   while (!sv.empty() && (sv.back() == ' ' || sv.back() == '\t' || sv.back() == '\r' || sv.back() == '\n'))
      sv.remove_suffix(1);
   return sv;
}
} // unnamed namespace

ROOT::RLogManager::RLogManager(std::unique_ptr<RLogHandler> lh) : RLogChannel(ELogLevel::kWarning)
{
   fHandlers.emplace_back(std::move(lh));
   ConfigureFromEnv();
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

ROOT::ELogLevel ROOT::RLogManager::GetChannelVerbosity(const std::string &name) const
{
   std::lock_guard<std::mutex> lock(fChannelVerbosityMutex);
   auto it = fChannelVerbosity.find(name);
   if (it != fChannelVerbosity.end())
      return it->second;
   return ELogLevel::kUnset;
}

void ROOT::RLogManager::SetChannelVerbosity(const std::string &name, ELogLevel level)
{
   std::lock_guard<std::mutex> lock(fChannelVerbosityMutex);
   if (level == ELogLevel::kUnset) {
      fChannelVerbosity.erase(name);
   } else {
      fChannelVerbosity[name] = level;
   }
}

void ROOT::RLogManager::ResetChannelVerbosity(const std::string &name)
{
   std::lock_guard<std::mutex> lock(fChannelVerbosityMutex);
   fChannelVerbosity.erase(name);
}

void ROOT::RLogManager::ResetAllChannelVerbosities()
{
   std::lock_guard<std::mutex> lock(fChannelVerbosityMutex);
   fChannelVerbosity.clear();
}

ROOT::ELogLevel ROOT::RLogManager::ParseLogLevel(std::string_view str)
{
   str = TrimWhitespace(str);
   if (str.empty())
      return ELogLevel::kUnset;

   int offset = 0;
   std::string_view base = str;

   auto openParen = str.find('(');
   auto closeParen = str.find(')', openParen);
   if (openParen != std::string_view::npos && closeParen != std::string_view::npos && closeParen > openParen) {
      base = TrimWhitespace(str.substr(0, openParen));
      std::string_view numStr = TrimWhitespace(str.substr(openParen + 1, closeParen - openParen - 1));
      try {
         offset = std::stoi(std::string(numStr));
      } catch (...) {
         offset = 0;
      }
   } else {
      auto plusPos = str.find('+');
      if (plusPos != std::string_view::npos) {
         base = TrimWhitespace(str.substr(0, plusPos));
         std::string_view numStr = TrimWhitespace(str.substr(plusPos + 1));
         try {
            offset = std::stoi(std::string(numStr));
         } catch (...) {
            offset = 0;
         }
      } else {
         auto colonPos = str.find(':');
         if (colonPos != std::string_view::npos) {
            base = TrimWhitespace(str.substr(0, colonPos));
            std::string_view numStr = TrimWhitespace(str.substr(colonPos + 1));
            try {
               offset = std::stoi(std::string(numStr));
            } catch (...) {
               offset = 0;
            }
         }
      }
   }

   std::string lowerBase;
   lowerBase.reserve(base.size());
   for (char c : base)
      lowerBase.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));

   if (lowerBase.size() > 1 && lowerBase.front() == 'k') {
      std::string_view stripped(lowerBase.c_str() + 1, lowerBase.size() - 1);
      if (stripped == "fatal" || stripped == "error" || stripped == "warning" ||
          stripped == "warn" || stripped == "info" || stripped == "debug" || stripped == "unset")
         lowerBase = std::string(stripped);
   }

   if (lowerBase == "fatal")
      return ELogLevel::kFatal;
   if (lowerBase == "error")
      return ELogLevel::kError;
   if (lowerBase == "warning" || lowerBase == "warn")
      return ELogLevel::kWarning;
   if (lowerBase == "info")
      return ELogLevel::kInfo;
   if (lowerBase == "debug") {
      int lvl = static_cast<int>(ELogLevel::kDebug) + offset;
      if (lvl > 255)
         lvl = 255;
      if (lvl < 0)
         lvl = 0;
      return static_cast<ELogLevel>(lvl);
   }
   if (lowerBase == "unset")
      return ELogLevel::kUnset;

   return ELogLevel::kUnset;
}

void ROOT::RLogManager::ConfigureFromString(std::string_view config)
{
   while (!config.empty()) {
      auto commaPos = config.find(',');
      std::string_view entry = (commaPos == std::string_view::npos) ? config : config.substr(0, commaPos);
      if (commaPos == std::string_view::npos)
         config = {};
      else
         config = config.substr(commaPos + 1);

      entry = TrimWhitespace(entry);
      if (entry.empty())
         continue;

      auto eqPos = entry.find('=');
      if (eqPos != std::string_view::npos) {
         std::string_view key = TrimWhitespace(entry.substr(0, eqPos));
         std::string_view val = TrimWhitespace(entry.substr(eqPos + 1));
         ELogLevel level = ParseLogLevel(val);
         if (key.empty() || key == "*" || key == "default" || key == "ROOT") {
            SetVerbosity(level);
         } else {
            SetChannelVerbosity(std::string(key), level);
         }
      } else {
         ELogLevel level = ParseLogLevel(entry);
         SetVerbosity(level);
      }
   }
}

void ROOT::RLogManager::ConfigureFromEnv()
{
   const char *env = std::getenv("ROOT_LOG");
   if (!env || !*env)
      env = std::getenv("ROOT_LOGGERS");
   if (env && *env)
      ConfigureFromString(env);
}
