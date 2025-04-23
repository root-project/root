#include "dequeHolder.h"
#include "test.C"

void dtest(bool readother=false) {
   const char* testname = "deque";
   typedef dequeHolder holder;

   std::cout << "Running test " << testname << std::endl;
   checkHolder<holder>(testname);
   write<holder>(testname);
   read<holder>(testname,0,readother);
}
