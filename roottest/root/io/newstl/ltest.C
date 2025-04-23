#include "listHolder.h"
#include "test.C"

void ltest(bool readother=false) {
   const char* testname = "list";
   typedef listHolder holder;

   std::cout << "Running test " << testname << std::endl;
   checkHolder<holder>(testname);
   write<holder>(testname);
   read<holder>(testname,0,readother);
}
