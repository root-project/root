#include "listHolder.h"
#include "test.C"

void ltest() {
   const char* testname = "list";
   typedef listHolder holder;

   std::cout << "Running test " << testname << std::endl;
   checkHolder<holder>(testname);
   write<holder>(testname);
   read<holder>(testname);
}
