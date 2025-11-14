#include "multimapHolder.h"
#include "test.C"

void ntest(const char *dirname = "") {
   const char* testname = "multimap";
   typedef multimapHolder holder;

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
