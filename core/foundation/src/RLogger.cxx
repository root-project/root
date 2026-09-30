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
#include <memory>
#include <vector>
#include <map>
#include <cstdlib>
#include <string>

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

static ROOT::ELogLevel ParseVerbosityStr(const std::string &str) {
   if (str == "Fatal") return ROOT::ELogLevel::kFatal;
   if (str == "Error") return ROOT::ELogLevel::kError;
   if (str == "Warning") return ROOT::ELogLevel::kWarning;
   if (str == "Info") return ROOT::ELogLevel::kInfo;
   if (str.compare(0, 5, "Debug") == 0) {
      if (str.length() > 6 && str[5] == '(' && str.back() == ')') {
         int level = std::stoi(str.substr(6, str.length() - 7));
         return static_cast<ROOT::ELogLevel>(static_cast<int>(ROOT::ELogLevel::kDebug) + level);
      }
      return ROOT::ELogLevel::kDebug;
   }
   return ROOT::ELogLevel::kUnset;
}

namespace ROOT {
namespace Internal {

void ParseRootLogStr(const std::string& env, std::map<std::string, ROOT::ELogLevel>& sChannelVerbosities) {
   std::string s = env;
   size_t start = 0;
   size_t end = s.find(',');
   while (start != std::string::npos) {
      std::string part = s.substr(start, end - start);
      size_t eq = part.find('=');
      if (eq != std::string::npos) {
         sChannelVerbosities[part.substr(0, eq)] = ParseVerbosityStr(part.substr(eq + 1));
      } else {
         sChannelVerbosities[""] = ParseVerbosityStr(part);
      }
      if (end == std::string::npos) break;
      start = end + 1;
      end = s.find(',', start);
   }
}

} // namespace Internal
} // namespace ROOT

static std::map<std::string, ROOT::ELogLevel>& GetChannelVerbosities() {
   static std::map<std::string, ROOT::ELogLevel> sChannelVerbosities;
   static bool parsed = false;
   if (!parsed) {
      parsed = true;
      const char *env = std::getenv("ROOT_LOG");
      if (env) {
         ROOT::Internal::ParseRootLogStr(env, sChannelVerbosities);
      }
   }
   return sChannelVerbosities;
}

} // unnamed namespace

ROOT::RLogManager &ROOT::RLogManager::Get()
{
   static RLogManager instance(std::make_unique<RLogHandlerDefault>());
   static bool configured = false;
   if (!configured) {
      configured = true;
      auto& cfg = GetChannelVerbosities();
      if (cfg.count("")) {
         instance.SetVerbosity(cfg[""]);
      }
   }
   return instance;
}

ROOT::ELogLevel ROOT::RLogManager::GetConfiguredVerbosity(const std::string &name) const
{
   auto& cfg = GetChannelVerbosities();
   auto it = cfg.find(name);
   if (it != cfg.end())
      return it->second;
   return ROOT::ELogLevel::kUnset;
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
