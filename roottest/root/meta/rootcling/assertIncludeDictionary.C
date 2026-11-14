#include "TClass.h"
#include "TClassTable.h"
#include "TError.h"
#include "TSystem.h"

#include <cstring>

int assertIncludeDictionary(const char *library, const char *className)
{
   // Each CTest invocation starts a fresh ROOT process. Do not let a dictionary
   // loaded by another case hide a missing selection in the requested library.
   if (TClassTable::GetDictNorm(className)) {
      Error("assertIncludeDictionary", "Dictionary for %s was already registered", className);
      return 1;
   }
   if (gSystem->Load(library) < 0) {
      Error("assertIncludeDictionary", "Could not load %s", library);
      return 2;
   }

   // Look up the compiled registration directly, without interpreter fallback,
   // header autoparsing or autoloading a different dictionary library.
   auto dictionary = TClassTable::GetDictNorm(className);
   if (!dictionary) {
      Error("assertIncludeDictionary", "%s did not register a dictionary for %s", library, className);
      return 3;
   }
   auto cl = dictionary();
   if (!cl || !cl->HasDictionary() || std::strcmp(cl->GetName(), className) != 0) {
      Error("assertIncludeDictionary", "Invalid compiled dictionary for %s from %s", className, library);
      return 4;
   }
   return 0;
}
