#include "multisetHolder.h"
#include "test.C"

void ttest() {
   const char* testname = "multiset";
   typedef multisetHolder holder;   

   std::cout << "Running test " << testname << std::endl;
   checkHolder<holder>(testname);
   write<holder>(testname);
   read<holder>(testname);
}
