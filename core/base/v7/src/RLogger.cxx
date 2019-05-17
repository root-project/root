/// \file TLogger.cxx
/// \ingroup Base ROOT7
/// \author Axel Naumann <axel@cern.ch>
/// \date 2015-07-07
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2015, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/RLogger.hxx"
#include <iostream>
#include <sstream>

#include "TError.h"

// pin vtable
ROOT::Experimental::TLogHandler::~TLogHandler() {}

namespace {
class TLogHandlerDefault: public ROOT::Experimental::TLogHandler {
public:
   // Returns false if further emission of this log entry should be suppressed.
   bool Emit(const ROOT::Experimental::TLogEntry &entry) override;
};

bool TLogHandlerDefault::Emit(const ROOT::Experimental::TLogEntry &entry)
{
   constexpr static std::array<const char *, 5> sTag{{"Debug", "Info", "Warning", "Log", "FATAL"}};
   std::stringstream strm;
   strm << "ROOT ";
   if (!entry.fGroup.empty())
      strm << '[' << entry.fGroup << "] ";
   strm << sTag[static_cast<int>(entry.fLevel)];

   if (!entry.fFile.empty())
      strm << " " << entry.fFile << ':' << entry.fLine;
   if (!entry.fFuncName.empty())
      strm << " in " << entry.fFuncName;

   static constexpr const int errorLevelOld[] = {0, 1000, 2000, 3000, 6000};
   (*::GetErrorHandler())(errorLevelOld[static_cast<int>(entry.fLevel)],
                           entry.fLevel == ROOT::Experimental::ELogLevel::kFatal,
                           strm.str().c_str(), entry.str().c_str());
   return true;
}
} // unnamed namespace

ROOT::Experimental::TLogManager &ROOT::Experimental::TLogManager::Get()
{
   static TLogManager instance(std::make_unique<TLogHandlerDefault>());
   return instance;
}
