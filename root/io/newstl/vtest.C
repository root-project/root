#include "vectorHolder.h"
#include "test.C"

void vtest() {
   const char* testname = "vector";
   typedef vectorHolder holder;

   std::cout << "Running test " << testname << std::endl;
   checkHolder<holder>(testname);
   write<holder>(testname);
   read<holder>(testname);
}
