#include "Template.h"

#ifdef __ROOTCLING__
#pragma link C++ class Template<short, 0, TagClassWithoutDefinition, int, 1>+;
#endif

extern "C" int printf(const char*,...);

// Signal that this library was loaded.
struct ThisLibraryHasBeenLoaded_t {
   ThisLibraryHasBeenLoaded_t() {
      printf("The library TmpltNoSpec_C has been loaded!\n");
   }
} gThisLibraryHasBeenLoaded;
