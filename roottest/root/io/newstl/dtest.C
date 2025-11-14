#include "dequeHolder.h"
#include "test.C"

void dtest(const char *dirname = "") {
   const char* testname = "deque";
   typedef dequeHolder holder;

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
