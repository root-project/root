#include "vectorHolder.h"
#include "test.C"

void vtest(bool readother=false) {
   const char* testname = "vector";
   typedef vectorHolder holder;

   std::cout << "Running test " << testname << std::endl;
   checkHolder<holder>(testname);
   write<holder>(testname);
   read<holder>(testname,0,readother);
}
