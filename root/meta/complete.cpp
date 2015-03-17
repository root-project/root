// By Oliver Freyermuth; see ROOT-7103

#include <string.h>
#include <set>

#include "TClass.h"
#include "TError.h"
#include "TClassTable.h"

int main(int argc, char** argv) {
   std::set<std::string> classes;
   for (Int_t i = 0; ; i++) {
      char* className = gClassTable->At(i);
      if (className == NULL) {
         // ID out of range.
         break;
      }
      classes.insert(className);
   }

   for (auto it = classes.begin(); it != classes.end(); ++it) {
      // Silent, autoload:
      // Ugly hack needed due to https://sft.its.cern.ch/jira/browse/ROOT-6225 .
      gErrorIgnoreLevel = kError;
      TClass* cls = TClass::GetClass(it->c_str(), kTRUE, kTRUE);
      gErrorIgnoreLevel = kPrint;
      // Skip classes without dictionary.
      if (cls == NULL) {
         continue;
      }
      // Filtering by abstract-ness - remove anything we can not "New()":
      if (cls->GetNew() == NULL) {
         //
      } else if (cls->InheritsFrom(TNamed::Class())) {
         //
      }
   }

   return 0;
}

