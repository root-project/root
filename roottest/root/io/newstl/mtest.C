#include "mapHolder.h"
#include "test.C"

void mtest(const char *dirname = "") {
   const char* testname = "map";
   typedef mapHolder holder;

   if (dirname && *dirname) {
      std::cout << "Running reading test " << testname << std::endl;
      read<holder>(dirname, testname);
   } else {
      std::cout << "Running test " << testname << std::endl;
      checkHolder<holder>(testname);
      write<holder>(testname);
      read<holder>("", testname);
   }
}
