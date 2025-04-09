#include "infoDumper.h"

int modelReadNoDict(const char* filename) {
   // This is without dictionaries
   unique_ptr<TFile> f (TFile::Open (filename));
   auto className = "edm2::A";
   dumpInfo(className);

   return 0;
}

