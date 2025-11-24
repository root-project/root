#include "vectorHolder.h"
#include "test.C"

void vtest(const char *dirname = "") {
   const char* testname = "vector";
   typedef vectorHolder holder;

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
