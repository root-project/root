#include "TClass.h"
#include "TClassEdit.h"
#include "TError.h"
#include "TInterpreter.h"
#include "TStreamerInfo.h"
#include "TSystem.h"
#include "TVirtualCollectionProxy.h"

#include <cstring>
#include <string>
#include <vector>

namespace {
ErrorHandlerFunc_t gPreviousHandler = nullptr;
std::vector<std::string> gInvalidPairs;
bool gSawAutoParseProbe = false;

void CaptureAutoParse(Int_t level, Bool_t abort, const char *location, const char *message)
{
   const char *prefix = "Trying to autoparse for pair<";
   if (level == kInfo && std::strstr(location, "TCling::AutoParse") &&
       std::strcmp(message, "Trying to autoparse for MapPairNameAutoParseProbe") == 0)
      gSawAutoParseProbe = true;
   if (level == kInfo && std::strstr(location, "TCling::AutoParse") &&
       std::strncmp(message, prefix, std::strlen(prefix)) == 0 && std::strstr(message, "MapPairKey")) {
      const char *name = message + std::strlen("Trying to autoparse for ");
      std::vector<std::string> parts;
      int nested = 0;
      if (TClassEdit::GetSplit(name, parts, nested) > 4)
         gInvalidPairs.emplace_back(name);
   }
   if (level > kInfo || abort)
      gPreviousHandler(level, abort, location, message);
}
}

int mapPairName()
{
   const auto previousDebug = gDebug;
   const auto previousIgnoreLevel = gErrorIgnoreLevel;
   const auto previousAutoParse = gInterpreter->SetClassAutoparsing(true);
   gPreviousHandler = SetErrorHandler(CaptureAutoParse);
   gErrorIgnoreLevel = kPrint;
   gDebug = 2;

   gInterpreter->AutoParse("MapPairNameAutoParseProbe");
   const auto loadResult = gSystem->Load("libMapPairName");
   auto holder = loadResult >= 0 ? TClass::GetClass("MapPairHolder") : nullptr;
   auto info = holder ? holder->GetStreamerInfo() : nullptr;

   gDebug = previousDebug;
   gErrorIgnoreLevel = previousIgnoreLevel;
   gInterpreter->SetClassAutoparsing(previousAutoParse);
   SetErrorHandler(gPreviousHandler);

   if (!gSawAutoParseProbe) {
      Error("mapPairName", "Could not observe AutoParse diagnostics");
      return 1;
   }
   if (!holder || !info) {
      Error("mapPairName", "Could not load libMapPairName or the holder and its streamer info");
      return 1;
   }
   if (!gInvalidPairs.empty()) {
      for (const auto &name : gInvalidPairs)
         Error("mapPairName", "AutoParse was called for invalid %s", name.c_str());
      return 2;
   }

   const char *collections[] = {
      "map<MapPairKey,string,MapPairLess>",
      "multimap<MapPairKey,string,MapPairLess>",
      "unordered_map<MapPairKey,vector<string>,MapPairHash,MapPairEqual>",
      "unordered_multimap<MapPairKey,vector<string>,MapPairHash,MapPairEqual>"
   };
   for (auto name : collections) {
      auto cl = TClass::GetClass(name);
      auto proxy = cl ? cl->GetCollectionProxy() : nullptr;
      auto value = proxy ? proxy->GetValueClass() : nullptr;
      if (!value) {
         Error("mapPairName", "No pair value class for %s", name);
         return 3;
      }
      std::vector<std::string> parts;
      int nested = 0;
      if (TClassEdit::GetSplit(value->GetName(), parts, nested) != 4 ||
          parts[1] != "MapPairKey" ||
          parts[2] != (std::strstr(name, "unordered_") ? "vector<string>" : "string")) {
         Error("mapPairName", "Unexpected value class %s for %s", value->GetName(), name);
         return 4;
      }
   }
   return 0;
}
