#include "setHolder.h"
#include "test.C"

void stest() {
   const char* testname = "set";
   typedef setHolder holder;

   std::cout << "Running test " << testname << std::endl;
   checkHolder<holder>(testname);
   write<holder>(testname);
   read<holder>(testname);
}
